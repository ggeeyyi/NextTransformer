"""Verified official CORE input cache and unchanged continuation scoring."""
from pathlib import Path
import hashlib, json, os, time
import torch
import torch.distributed as dist
from nanochat import core_eval
r = Path(__file__).resolve().parents[1]

def key(meta):
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()

def inputs(meta, data, rank):
    p = r / 'core-input-cache' / f'{key(meta)}-rank{rank}.pt'
    saved = torch.load(p, map_location='cpu', weights_only=False)
    assert saved['task_meta'] == meta and saved['rank'] == rank and (saved['world'] == 8) and (saved['count'] == len(data))
    raw = (r / 'resources/eval_bundle/eval_data' / meta['dataset_uri']).read_bytes()
    assert saved['data_sha'] == hashlib.sha256(raw).hexdigest()
    assert saved['official_source_sha'] == hashlib.sha256(Path(core_eval.__file__).read_bytes()).hexdigest()
    assert [e['idx'] for e in saved['entries']] == list(range(rank, len(data), 8))
    return saved

def score(e, losses, predictions, ids):
    if e['task_type'] == 'language_modeling':
        (s, t) = (e['start_idxs'][0], e['end_idxs'][0])
        return bool(torch.all(predictions[0, s - 1:t - 1] == ids[0, s:t]).item())
    means = [losses[i, s - 1:t - 1].mean().item() for (i, (s, t)) in enumerate(zip(e['start_idxs'], e['end_idxs']))]
    return means.index(min(means)) == e['gold']
