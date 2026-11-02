"""Compare complete results using tolerances registered before either evaluation."""
import argparse
from pathlib import Path
from environments.eval.common import digest, finite, read, require, write
from environments.eval.validate import validate


def compare(a, b, contract, tolerances):
    validate(a, contract)
    validate(b, contract)
    require(a['method'] == b['method'], 'Method mismatch')
    differences = {}
    pairs = {f'bpb.{s}': (a['bpb'][s]['bpb'], b['bpb'][s]['bpb']) for s in ('train', 'val')}
    pairs.update({f'core.{s}': (a['core']['results'][s], b['core']['results'][s]) for s in contract['core_tasks']})
    pairs['core_metric'] = (a['core']['core_metric'], b['core']['core_metric'])
    require(set(tolerances) == set(pairs), 'Provide a preregistered absolute tolerance for every metric')
    for name, (x, y) in pairs.items():
        tol = tolerances[name]
        require(finite(tol) and tol >= 0, 'Invalid tolerance')
        differences[name] = dict(absolute=abs(x-y), tolerance=tol, passed=abs(x-y) <= tol)
    return dict(passed=all(x['passed'] for x in differences.values()), differences=differences)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'contract', 'tolerances', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    report = compare(read(args.reference), read(args.candidate), read(args.contract), read(args.tolerances))
    report['inputs_sha256'] = {name: digest(getattr(args, name)) for name in
                              ('reference', 'candidate', 'contract', 'tolerances')}
    write(args.output, report)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
