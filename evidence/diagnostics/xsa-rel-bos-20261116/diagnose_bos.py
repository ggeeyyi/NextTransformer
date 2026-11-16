"""Post-run BOS diagnostics using original CORE prompts and answer scoring.

Run with torchrun and a completed native NextTransformer checkpoint. This writes
new diagnostic records only; it never edits a run or changes the shared evaluator.
"""
import argparse
import collections
import gc
import json
import os
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', default='squad,boolq,bigbench_language_identification,agi_eval_lsat_ar')
    parser.add_argument('--all', action='store_true', help='All examples, original and cropped BOS restoration')
    parser.add_argument('--controls', action='store_true', help='Add space replacement and BOS deletion')
    parser.add_argument('--squad-limit', type=int, default=1024)
    parser.add_argument('--per-stratum', type=int, default=128)
    args = parser.parse_args()
    if args.all and args.controls:
        parser.error('--all and --controls are separate experiment scopes')
    if min(args.squad_limit, args.per_stratum) < 1:
        parser.error('sample limits must be positive')
    # Isolate compiler caches across ranks, including when called outside train.
    for key in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
        if os.environ.get(key):
            os.environ[key] = str(Path(os.environ[key])/('bos-rank-'+os.environ.get('RANK', '0')))
    import torch
    import torch.distributed as dist
    import yaml
    from nanochat import core_eval
    from nanochat.tokenizer import RustBPETokenizer
    from nexttransformer.models import load_model
    from nexttransformer.runtime.process import initialize

    rank, world, device = initialize(42)
    torch.set_num_threads(1)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    tok = RustBPETokenizer.from_directory(args.resources/'tokenizer')
    bos, space = tok.get_bos_token_id(), tok.encode(' ')[0]
    model, saved = load_model(args.checkpoint, device)
    if saved['state']['step'] != 46875 or model.config.sequence_len != 2048:
        raise ValueError('Use a completed fixed-protocol 2k checkpoint')
    model.eval()
    method = model.config.method
    del saved
    gc.collect()
    tasks = yaml.safe_load((args.resources/'eval_bundle/core.yaml').read_text())['icl_tasks']
    selected = args.tasks.split(',')
    if set(selected) - {t['label'] for t in tasks}:
        raise ValueError('Unknown CORE task')

    def encode(text, **kwargs):
        return tok.encode(text, num_threads=1, **kwargs)
    encode.get_bos_token_id = tok.get_bos_token_id
    records = []
    for task in tasks:
        label = task['label']
        if label not in selected:
            continue
        data = [json.loads(line) for line in (args.resources/'eval_bundle/eval_data'/task['dataset_uri']).read_text().splitlines()]
        random.Random(1337).shuffle(data)
        kind = task['icl_task_type']
        render, batch = {
            'language_modeling': (core_eval.render_prompts_lm, core_eval.batch_sequences_lm),
            'multiple_choice': (core_eval.render_prompts_mc, core_eval.batch_sequences_mc),
            'schema': (core_eval.render_prompts_schema, core_eval.batch_sequences_schema),
        }[kind]
        counts = collections.Counter()
        for index, item in enumerate(data):
            if not args.all and label == 'squad' and index >= args.squad_limit:
                break
            indices = random.Random(1234+index).sample([i for i in range(len(data)) if i != index], task['num_fewshot'][0])
            tokens, starts, ends = batch(encode, render(item, task.get('continuation_delimiter', ' '), [data[i] for i in indices]))
            crops = [max(0, len(t)-2048) for t in tokens]
            group = 'cropped' if any(crops) else 'uncropped'
            counts[group] += 1
            if not args.all and label != 'squad' and counts[group] > args.per_stratum:
                continue
            if index % world != rank:
                continue
            tokens = [t[c:] for t, c in zip(tokens, crops)]
            starts = [s-c for s, c in zip(starts, crops)]
            ends = [e-c for e, c in zip(ends, crops)]
            variants = {'original': (tokens, starts, ends)}
            if group == 'cropped':
                variants['restore_bos'] = ([[bos]+t[1:] if c else t for t, c in zip(tokens, crops)], starts, ends)
                if args.controls:
                    variants['replace_first_with_space'] = ([[space]+t[1:] if c else t for t, c in zip(tokens, crops)], starts, ends)
            elif args.controls:
                assert all(t[0] == bos for t in tokens)
                variants['replace_bos_with_space'] = ([[space]+t[1:] for t in tokens], starts, ends)
                variants['remove_bos'] = ([t[1:] for t in tokens], [s-1 for s in starts], [e-1 for e in ends])
            for variant, (changed, ss, ee) in variants.items():
                assert all(0 < s < e <= len(t) <= 2048 for t, s, e in zip(changed, ss, ee))
                assert all(t[s:e] == original[a:b] for t, s, e, original, a, b in zip(changed, ss, ee, tokens, starts, ends))
                with torch.inference_mode():
                    ids = core_eval.stack_sequences(changed, bos).to(device)
                    model.set_row_offset(index)
                    if kind == 'language_modeling':
                        s, e = ss[0], ee[0]
                        output = model(ids)
                        logits = output[0, s-1:e-1].float()
                        del output
                        gold = ids[0, s:e]
                        matches = logits.argmax(-1).eq(gold)
                        nll = -logits.log_softmax(-1).gather(1, gold[:, None]).sum()
                        correct = bool(matches.all())
                        extra = dict(token_correct=int(matches.sum()), answer_tokens=e-s, nll_sum=float(nll), first_correct=bool(matches[0]))
                        del logits, gold, matches, nll
                    else:
                        losses, predictions = core_eval.forward_model(model, ids)
                        means = [float(losses[i, s-1:e-1].mean()) for i, (s, e) in enumerate(zip(ss, ee))]
                        chosen = min(range(len(means)), key=lambda i: means[i])
                        correct = chosen == item['gold']
                        extra = dict(chosen=chosen, gold=item['gold'], choice_losses=means)
                        del losses, predictions
                    records.append(dict(method=method, task=label, idx=index, group=group, variant=variant, correct=correct, crops=crops, **extra))
                    del ids
    (args.output/f'rank-{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    if world > 1:
        dist.barrier()
    if rank == 0:
        rows = [json.loads(line) for i in range(world) for line in (args.output/f'rank-{i}.jsonl').read_text().splitlines()]
        groups = collections.defaultdict(list)
        for row in rows:
            groups[(row['task'], row['group'], row['variant'])].append(row)
        summary = [dict(task=t, group=g, variant=v, examples=len(rs), correct=sum(r['correct'] for r in rs), accuracy=sum(r['correct'] for r in rs)/len(rs)) for (t, g, v), rs in sorted(groups.items())]
        (args.output/'summary.json').write_text(json.dumps(dict(scope='diagnostic only', method=method, groups=summary), indent=2)+'\n')
        print(json.dumps(dict(status='passed', method=method, records=len(rows))), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
