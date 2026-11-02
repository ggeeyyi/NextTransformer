"""Reject partial, nonfinite, smoke, or identity-mismatched scoring outputs."""
import argparse
from pathlib import Path
from environments.eval.common import finite, read, require, write


def validate(result, contract):
    require(result['step'] == contract['checkpoint']['step'], 'Checkpoint step mismatch')
    require(result['native_manifest_sha256'] == contract['native_manifest_sha256'], 'Resource identity mismatch')
    require(result['track'] == 'registered', 'This scoring validator supports registered methods only')
    require(set(result['bpb']) == {'train', 'val'}, 'Both BPB splits required')
    for split, item in result['bpb'].items():
        require(item['split'] == split and item['input_tokens'] == 20_971_520, 'Incomplete BPB budget')
        require(item['smoke'] is False and finite(item['bpb']), 'Invalid or smoke BPB')
        require(item['protocol_hash'] == contract['protocol_hash'], 'BPB protocol mismatch')
    core = result['core']
    require(core['full_core'] is True and core['expected_tasks'] == 22, 'Full CORE required')
    require(core['protocol_hash'] == contract['protocol_hash'], 'CORE protocol mismatch')
    tasks = set(contract['core_tasks'])
    require(len(tasks) == 22, 'Expected 22 task names')
    for key in ('results', 'centered_results'):
        require(set(core[key]) == tasks, 'CORE task coverage mismatch')
        require(all(finite(v) for v in core[key].values()), 'Nonfinite CORE task score')
    require(finite(core['core_metric']), 'Nonfinite CORE aggregate')
    mean = sum(core['centered_results'].values()) / 22
    require(abs(mean - core['core_metric']) <= 1e-12, 'CORE aggregate mismatch')
    return dict(valid=True, meaning='evaluation completeness, not model quality', step=result['step'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    write(args.output, validate(read(args.result), read(args.contract)))


if __name__ == '__main__':
    main()
