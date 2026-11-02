"""Launch one immutable, offline scoring attempt using an explicit GPU allocation."""
import argparse
import os
import re
import subprocess
import uuid
from pathlib import Path
from environments.eval.common import read, require, write


def configuration(path):
    cfg = read(path)
    required = {'image', 'checkpoint', 'resources', 'data', 'kernels', 'contract', 'gpu_devices', 'output', 'timeout_seconds'}
    require(set(cfg) == required, f'Configuration keys must be exactly {sorted(required)}')
    devices = cfg['gpu_devices']
    require(isinstance(devices, list) and len(devices) == len(set(devices)) == 8, 'Eight explicitly allocated unique devices required')
    require(all(isinstance(d, str) and re.fullmatch(r'(?:[0-9]+|GPU-[a-fA-F0-9-]+)', d) for d in devices), 'Invalid GPU IDs')
    require(isinstance(cfg['timeout_seconds'], int) and cfg['timeout_seconds'] > 0, 'Positive timeout required')
    for name in ('checkpoint', 'resources', 'data', 'kernels', 'contract', 'output'):
        path = Path(cfg[name]).expanduser()
        require(path.is_absolute() and not any(c in str(path) for c in (',', '$', '\n')), f'Absolute Docker-compatible {name} path required')
        cfg[name] = str(path.resolve())
        if name != 'output':
            require(path.is_file() if name == 'contract' else path.is_dir(), f'Missing {name}')
    require(not Path(cfg['output']).exists(), 'Output directory already exists; choose a fresh attempt')
    contract = read(cfg['contract'])
    require(contract['world_size'] == 8, 'Initial launcher requires eight GPUs')
    return cfg


def image_id(image):
    return subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()


def command(cfg, identity, name):
    cmd = ['docker', 'run', '--rm', '--pull=never', '--name', name, '--network', 'none',
           '--gpus', '"device=' + ','.join(cfg['gpu_devices']) + '"', '--read-only',
           '--cap-drop=ALL', '--security-opt=no-new-privileges', '--user', f'{os.getuid()}:{os.getgid()}',
           '--shm-size=16g', '--tmpfs', '/tmp:rw,nosuid,size=4g,mode=1777',
           '--tmpfs', '/work:rw,nosuid,size=16g,mode=1777', '--env', 'HOME=/work',
           '--env', 'PYTHONDONTWRITEBYTECODE=1']
    for key in ('checkpoint', 'resources', 'data', 'kernels', 'contract'):
        dest = '/inputs/' + ('contract.json' if key == 'contract' else key)
        cmd += ['--mount', f'type=bind,src={cfg[key]},dst={dest},readonly']
    cmd += ['--mount', f'type=bind,src={cfg["output"]},dst=/logs/verifier',
            '--entrypoint', '/opt/venv/bin/python', identity, '-m', 'environments.eval.run']
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true', help='Resolve local image and print command without allocating GPUs')
    args = p.parse_args()
    cfg = configuration(args.config)
    identity = image_id(cfg['image'])
    name = 'nextt-eval-' + uuid.uuid4().hex[:12]
    cmd = command(cfg, identity, name)
    if args.dry_run:
        import shlex
        print(shlex.join(cmd))
        return
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'launch.json', dict(image_id=identity, configuration=cfg, command=cmd))
    try:
        with (output / 'container.log').open('w') as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=cfg['timeout_seconds'])
        write(output / 'host-status.json', dict(returncode=result.returncode))
        raise SystemExit(result.returncode)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
        write(output / 'host-status.json', dict(state='interrupted', error=type(exc).__name__))
        raise


if __name__ == '__main__':
    main()
