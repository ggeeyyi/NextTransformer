"""Generate a local Docker/nop task for the pinned Harbor 0.9.0 controller."""
import argparse
import os
from pathlib import Path
from environments.eval.common import require, write
from environments.eval.launch import configuration, image_id


def generate(cfg, output, identity):
    require(not output.exists(), 'Refusing to overwrite a Harbor task')
    (output / 'environment').mkdir(parents=True)
    (output / 'tests').mkdir()
    mounts = []
    for key in ('checkpoint', 'resources', 'data', 'kernels', 'contract'):
        mounts.append(dict(type='bind', source=cfg[key], target='/inputs/' +
                           ('contract.json' if key == 'contract' else key), read_only=True))
    service = dict(image=identity, pull_policy='never', entrypoint=['/bin/sh', '-c'],
        command=['sleep infinity'], user=f'{os.getuid()}:{os.getgid()}', network_mode='none',
        read_only=False, cap_drop=['ALL'], security_opt=['no-new-privileges:true'],
        environment=dict(HOME='/work', PYTHONDONTWRITEBYTECODE='1'),
        shm_size='16gb', tmpfs=['/work:rw,nosuid,size=16g,mode=1777', '/tmp:rw,nosuid,size=4g,mode=1777'],
        volumes=mounts, deploy=dict(resources=dict(reservations=dict(devices=[dict(
            driver='nvidia', device_ids=cfg['gpu_devices'], capabilities=['gpu'])]))))
    # JSON is a YAML subset; configuration() rejects Compose dollar interpolation in paths.
    write(output / 'environment/docker-compose.yaml', {'services': {'main': service}})
    import json
    (output / 'task.toml').write_text('version = "1.0"\n[agent]\ntimeout_sec = 60\n'
        f'[verifier]\ntimeout_sec = {cfg["timeout_seconds"]}\n[environment]\n'
        f'docker_image = {json.dumps(identity)}\ncpus = 16\nmemory_mb = 131072\n'
        'gpus = 8\nallow_internet = false\n')
    (output / 'instruction.md').write_text('Run only with the nop agent. The verifier scores an existing '
        'reviewed checkpoint in the fixed runtime. Do not install packages or modify inputs.\n')
    (output / 'tests/test.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
        'rm -f /logs/verifier/reward.txt\n'
        '/opt/venv/bin/python -m environments.eval.run\n'
        'printf "1\\n" > /logs/verifier/reward.txt\n')
    (output / 'tests/test.sh').chmod(0o755)
    write(output / 'adapter.json', dict(harbor_version='0.9.0', image_id=identity,
          purpose='trusted checkpoint scoring', gpu_devices=cfg['gpu_devices']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(os.getuid() != 0, 'Run the Harbor adapter as a non-root host user')
    cfg = configuration(args.config)
    generate(cfg, args.output, image_id(cfg['image']))


if __name__ == '__main__':
    main()
