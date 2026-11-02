"""Materialize a selected reference HF kernel repository without credentials or network."""
import argparse
import shutil
from pathlib import Path
from environments.eval.common import require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hub-cache', type=Path, required=True)
    p.add_argument('--repository', choices=('varunneal/flash-attention-3', 'kernels-community/flash-attn3'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = args.hub_cache / ('models--' + args.repository.replace('/', '--'))
    require(source.is_dir(), 'Reference kernel repository is not cached')
    require(not args.output.exists(), 'Use a fresh output directory')
    # Include refs and snapshots; dereference snapshot blob links into the archive.
    # Authentication files and unrelated model caches never enter the output.
    target = args.output / source.name
    target.mkdir(parents=True)
    for name in ('refs', 'snapshots'):
        require((source / name).is_dir(), f'Missing kernel {name}')
        shutil.copytree(source / name, target / name, symlinks=False)


if __name__ == '__main__':
    main()
