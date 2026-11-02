"""Build a reviewed method source layer on an existing fixed scoring runtime.

Only model/configuration changes are accepted. Changes to the evaluator, runtime,
backbone, tokenizer, or dependency locks require a separately reviewed runtime.
"""
import argparse
import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from environments.eval.common import digest, read, require, write
from environments.eval.launch import image_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-image', required=True)
    p.add_argument('--commit', required=True, help='Full reviewed source commit; not arbitrary unreviewed code')
    p.add_argument('--tag', required=True)
    p.add_argument('--builder')
    p.add_argument('--record', type=Path, required=True)
    args = p.parse_args()
    require(not args.record.exists(), 'Use a fresh provenance record')
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', args.commit + '^{commit}'], text=True).strip()
    require('@sha256:' in args.base_image, 'Source layers require a published base image digest')
    base = image_id(args.base_image)
    import json
    inventory = json.loads(subprocess.check_output(['docker', 'run', '--rm', '--network', 'none',
        '--entrypoint', 'cat', base, '/opt/nextt-manifest/build.json'], text=True))
    paths = ('nexttransformer', 'nanochat', 'configs', 'requirements', 'pyproject.toml', 'uv.lock')
    archive = subprocess.check_output(['git', 'archive', commit, *paths])
    with tempfile.TemporaryDirectory(prefix='nextt-source-') as temp:
        root = Path(temp)
        source = root / 'source'
        source.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                require(member.isfile() or member.isdir(), 'Source archive must not contain links or devices')
                require(not Path(member.name).is_absolute() and '..' not in Path(member.name).parts, 'Invalid source path')
                if member.isfile():
                    target = source / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
        def method_file(name):
            return name.startswith(('configs/', 'nexttransformer/models/')) and name not in (
                'nexttransformer/models/backbone.py', 'nexttransformer/models/__init__.py')
        expected = inventory['source_files']
        for path in source.rglob('*'):
            if path.is_file():
                name = str(path.relative_to(source))
                if not method_file(name):
                    require(expected.get(name) == digest(path), f'Runtime source differs: {name}; needs a new runtime release')
        for name in expected:
            if any(name == x or name.startswith(x + '/') for x in paths) and not method_file(name):
                require((source / name).is_file(), f'Missing protected source: {name}')
        # Use the full verified source subset, including removals within model/config directories.
        (root / 'Dockerfile').write_text('ARG BASE\nFROM ${BASE}\nUSER root\n'
            'RUN rm -rf /opt/nextt/nexttransformer/models /opt/nextt/configs\n'
            'COPY source/ /opt/nextt/\nCOPY provenance.json /opt/nextt-manifest/source-layer.json\n'
            f'LABEL org.opencontainers.image.revision={commit}\n'
            'RUN python -c "import json; from pathlib import Path; from environments.eval.inventory import inventory; '
            'p=Path(\'/opt/nextt-manifest/build.json\'); old=json.loads(p.read_text()); '
            'new=inventory(Path(\'/opt/nextt\'), \'' + commit + '\', old.get(\'download_sources\')); '
            'p.write_text(json.dumps(new, indent=2))"\n')
        provenance = dict(base_image_id=base, reviewed_source_commit=commit, status='candidate_not_parity_validated')
        write(root / 'provenance.json', provenance)
        cmd = ['docker', 'buildx', 'build', '--load', '--network=none', '--build-arg', 'BASE=' + args.base_image,
               '--tag', args.tag]
        if args.builder: cmd += ['--builder', args.builder]
        subprocess.run([*cmd, str(root)], check=True)
        provenance['image_id'] = image_id(args.tag)
        write(args.record, provenance)


if __name__ == '__main__':
    main()
