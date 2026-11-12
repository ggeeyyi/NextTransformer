"""Export a finished scaling-stage run as a results/scaling record, or check every record.

    python -m nexttransformer.scaling.export RUN_DIR --config configs/scaling/dense_200b.json --id dense_200b \\
        --name 'Dense · 200B tokens · 4K' --family dense --wandb URL --checkpoint URL
    python -m nexttransformer.scaling.export --check

Export reads only rank zero's records in RUN_DIR: provenance.json, run_status.json, metrics.jsonl, bpb_final.json
and core_final.json. It refuses a run unless training completed with this configuration, every scheduled validation
and full 22-task CORE evaluation is present, and the final records agree with the event history. A record keeps
scientific quantities only, never host names, paths, devices or commands. Scaling-stage records are not leaderboard
entries, and existing records are immutable: an identical export is a no-op.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

from nexttransformer.scaling.config import config_hash, load_config

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO/'results/scaling/runs'
FIELDS = ('id', 'name', 'family', 'protocol', 'config', 'configSha256', 'model', 'parameters', 'training', 'source',
          'val', 'core', 'finalBpb', 'finalCore', 'links')
CORE_TASKS = 22


def by_step(rows, kind):
    return {row['step']: row for row in rows if row.get('event') == kind}


def record(run, config, run_id, name, family, wandb=None, checkpoint=None):
    run, config = Path(run), Path(config)
    mc, tc = load_config(config)
    provenance = json.loads((run/'provenance.json').read_text())
    status = json.loads((run/'run_status.json').read_text())
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    train, val, core = by_step(rows, 'train'), by_step(rows, 'validation'), by_step(rows, 'core')
    if status.get('phase') != 'training_complete' or status.get('step') != tc.steps or provenance.get('smoke'):
        raise ValueError('the run has not completed training')
    if provenance['config_sha256'] != config_hash(mc, tc):
        raise ValueError(f'the run was not trained with {config}')
    missing = [s for s in tc.validation_steps() if s not in val]+[s for s in tc.core_steps() if s not in core]
    if missing:
        raise ValueError(f'missing scheduled evaluations at steps {missing}')
    if any(not core[s].get('full_core') or len(core[s]['centered_results']) != CORE_TASKS for s in tc.core_steps()):
        raise ValueError(f'every CORE evaluation must cover the full {CORE_TASKS} tasks')
    final_bpb = json.loads((run/'bpb_final.json').read_text())
    final_core = json.loads((run/'core_final.json').read_text())
    if not math.isclose(final_core['core_metric'], core[tc.steps]['core_metric'], rel_tol=1e-12):
        raise ValueError('core_final.json differs from the event history')
    last, head = train[max(train)], mc.width*mc.vocab_size
    mfu = [r['mfu'] for r in train.values() if r.get('mfu')]
    return {
        'id': run_id, 'name': name, 'family': family, 'protocol': tc.protocol,
        'config': config.resolve().relative_to(REPO).as_posix(), 'configSha256': provenance['config_sha256'],
        'model': {'layers': mc.dense_layers, 'width': mc.width, 'heads': mc.heads, 'mlpWidth': mc.mlp_width,
                  'sequenceLength': mc.sequence_len, 'vocabSize': mc.vocab_size},
        'parameters': {'total': provenance['parameters'], 'blocks': provenance['scaling_parameters']-head,
                       'blocksAndHead': provenance['scaling_parameters']},
        'training': {'steps': tc.steps, 'inputTokens': last['input_tokens'], 'globalBatchTokens': tc.global_batch_tokens,
                     'worldSize': provenance['world_size'], 'microBatch': provenance['micro_batch'],
                     'gradientAccumulation': provenance['gradient_accumulation'],
                     'theoreticalGflopsPerToken': provenance['theoretical_GFLOPs_per_token'],
                     'trainingFlopsEstimate': last['training_flops_estimate'], 'gpuHours': last['gpu_hours'],
                     'medianMfu': statistics.median(mfu) if mfu else None},
        'source': {'sha256': provenance['source_sha256'], 'nativeManifestSha256': provenance['native_manifest_sha256'],
                   'dataSha256': provenance['data'], 'torch': provenance['torch'], 'cuda': provenance['cuda'],
                   'attention': provenance['attention']},
        'val': [{'step': s, 'value': val[s]['val_bpb']} for s in sorted(val)],
        'core': [{'step': s, 'value': core[s]['core_metric']} for s in sorted(core)],
        'finalBpb': {'val': final_bpb['val']['val_bpb'], 'train': final_bpb['train']['train_bpb']},
        'finalCore': {'value': final_core['core_metric'], 'tasks': final_core['centered_results']},
        'links': {'wandb': wandb, 'checkpoint': checkpoint},
    }


def check(results=None):
    problems = []
    for path in sorted(Path(results or RESULTS).glob('*.json')):
        run = json.loads(path.read_text())
        if tuple(run) != FIELDS or run['id'] != path.stem:
            problems.append(f'{path.name}: fields or id do not match the record schema')
            continue
        for key in ('val', 'core'):
            steps = [p['step'] for p in run[key]]
            if steps != sorted(set(steps)) or not all(math.isfinite(p['value']) for p in run[key]):
                problems.append(f'{path.name}: {key} must have increasing steps and finite values')
        if run['core'][-1]['step'] != run['training']['steps'] or len(run['finalCore']['tasks']) != CORE_TASKS:
            problems.append(f'{path.name}: the final CORE evaluation is missing or incomplete')
        if any(url and not url.startswith('https://') for url in run['links'].values()):
            problems.append(f'{path.name}: links must be https URLs')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run', nargs='?', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--id')
    parser.add_argument('--name')
    parser.add_argument('--family', default='dense')
    parser.add_argument('--wandb')
    parser.add_argument('--checkpoint')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.check:
        problems = check()
        print('\n'.join(problems) or 'scaling records: ok')
        return 1 if problems else 0
    if not (args.run and args.config and args.id and args.name):
        parser.error('export needs RUN_DIR, --config, --id and --name')
    run = record(args.run, args.config, args.id, args.name, args.family, args.wandb, args.checkpoint)
    path = RESULTS/f'{args.id}.json'
    text = json.dumps(run, indent=1, ensure_ascii=False)+'\n'
    if path.exists() and path.read_text() != text:
        raise SystemExit(f'{path} already holds a different record; scaling records are immutable')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f'wrote {path}: final val_bpb {run["finalBpb"]["val"]:.4f}, '
          f'CORE {run["finalCore"]["value"]:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
