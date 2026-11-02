"""Machine configuration, resolved before importing torch, tiktoken or wandb.

Only the standard library is used so the same resolver can bootstrap a venv.
Precedence: CLI > explicit site file > existing environment > derived defaults.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys


# option: environment variable, default relative to runtime (None: no default)
PATHS = {
    'runtime_dir': ('NANOCHAT_RUNTIME', None),
    'venv_dir': ('NANOCHAT_VENV', 'venv'),
    'resources': ('NEXTT_RESOURCES', 'resources'),
    'data_dir': ('NEXTT_DATA_DIR', None),
    'data_manifest': ('NEXTT_DATA_MANIFEST', None),
    'source_manifest': ('NEXTT_SOURCE_MANIFEST', None),
    'core_archive': ('NEXTT_CORE_ARCHIVE', None),
    'upstream_base_dir': ('NANOCHAT_BASE_DIR', 'upstream'),
    'hf_home': ('HF_HOME', 'huggingface'),
    'hf_hub_cache': ('HF_HUB_CACHE', None),
    'tiktoken_cache_dir': ('TIKTOKEN_CACHE_DIR', 'cache/tiktoken'),
    'torch_home': ('TORCH_HOME', 'cache/torch'),
    'triton_cache_dir': ('TRITON_CACHE_DIR', 'cache/triton'),
    'torchinductor_cache_dir': ('TORCHINDUCTOR_CACHE_DIR', 'cache/inductor'),
    'cuda_cache_path': ('CUDA_CACHE_PATH', 'cache/cuda'),
    'xdg_cache_home': ('XDG_CACHE_HOME', 'cache/xdg'),
    'pythonpycacheprefix': ('PYTHONPYCACHEPREFIX', 'cache/pycache'),
    'uv_python_install_dir': ('UV_PYTHON_INSTALL_DIR', 'python'),
    'uv_cache_dir': ('UV_CACHE_DIR', 'cache/uv'),
    'pip_cache_dir': ('PIP_CACHE_DIR', 'cache/pip'),
    'tmpdir': ('TMPDIR', 'tmp'),
    'wandb_dir': ('WANDB_DIR', None),
    'wandb_cache_dir': ('WANDB_CACHE_DIR', 'cache/wandb'),
    'wandb_data_dir': ('WANDB_DATA_DIR', 'cache/wandb-data'),
    'wandb_artifact_dir': ('WANDB_ARTIFACT_DIR', 'artifacts'),
    'wandb_config_dir': ('WANDB_CONFIG_DIR', 'wandb-config'),
    'wandb_netrc': ('NETRC', None),
}
VALUES = {
    'run_name': ('NEXTT_RUN_NAME', None),
    'gpus': ('GPUS', '8'),
    'micro_batch': ('MICRO_BATCH', None),
    'python_version': ('NEXTT_PYTHON_VERSION', '3.10.19'),
    'index_url': ('NEXTT_INDEX_URL', None),
    'hf_endpoint': ('HF_ENDPOINT', None),
    'core_url': ('NEXTT_CORE_URL', None),
    'wandb_mode': ('WANDB_MODE', 'disabled'),
    'wandb_metrics': ('NEXTT_WANDB_METRICS', 'core'),
    'wandb_project': ('WANDB_PROJECT', None),
    'wandb_entity': ('WANDB_ENTITY', None),
    'wandb_group': ('WANDB_RUN_GROUP', None),
}


def add_runtime_arguments(parser, *, defaults=True):
    parser.add_argument('--site-config', type=Path, default=os.environ.get('NEXTT_SITE_CONFIG') if defaults else None,
                        help='Optional local JSON file; paths in it are relative to that file')
    for name, (env, _) in PATHS.items():
        parser.add_argument('--'+name.replace('_', '-'), type=lambda p: Path(os.path.expandvars(p)).expanduser().resolve(),
                            default=os.environ.get(env) if defaults else None)
    for name, (env, fallback) in VALUES.items():
        options = {'default': os.environ.get(env, fallback) if defaults else None}
        if name in ('gpus', 'micro_batch'):
            options['type'] = int
        if name == 'wandb_mode':
            options['choices'] = ('disabled', 'offline', 'online')
        if name == 'wandb_metrics':
            options['choices'] = ('core', 'full')
        parser.add_argument('--'+name.replace('_', '-'), **options)


def resolve(argv=None, environ=None):
    env = dict(os.environ if environ is None else environ)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_runtime_arguments(parser, defaults=False)
    args, _ = parser.parse_known_args(argv)
    local = args.site_config or env.get('NEXTT_SITE_CONFIG')
    config, config_dir = {}, Path.cwd()
    if local:
        local = Path(os.path.expandvars(str(local))).expanduser().resolve()
        config = json.loads(local.read_text())
        if not isinstance(config, dict):
            raise ValueError('site config must be a JSON object')
        unknown = set(config) - PATHS.keys() - VALUES.keys()
        if unknown:
            raise ValueError(f'Unknown site settings: {sorted(unknown)}; credentials belong in login/environment, not this file')
        config_dir = local.parent
    values = {}
    for name, (key, _) in {**PATHS, **VALUES}.items():
        value = getattr(args, name)
        origin = 'cli'
        if value is None:
            value, origin = config.get(name), 'file'
        if value is None:
            value, origin = env.get(key), 'environment'
        if value is not None:
            if not isinstance(value, (str, int, Path)) or isinstance(value, bool) or not str(value).strip():
                raise ValueError(f'{name} must be a nonempty string or integer')
            if name in PATHS:
                path = Path(os.path.expandvars(str(value))).expanduser()
                value = str(((config_dir if origin == 'file' else Path.cwd())/path).resolve())
            values[name] = str(value)
    root = values.get('runtime_dir')
    if not root:
        raise ValueError('Specify --runtime-dir, runtime_dir in --site-config, or NANOCHAT_RUNTIME (choose persistent storage)')
    for name, (_, relative) in PATHS.items():
        if name not in values and relative is not None:
            values[name] = str(Path(root)/relative)
    values.setdefault('hf_hub_cache', str(Path(values['hf_home'])/'hub'))
    for name, (_, fallback) in VALUES.items():
        if name not in values and fallback is not None:
            values[name] = fallback
    if values['wandb_mode'] not in ('disabled', 'offline', 'online'):
        raise ValueError('wandb_mode must be disabled, offline or online')
    if values['wandb_metrics'] not in ('core', 'full'):
        raise ValueError('wandb_metrics must be core or full')
    for name in ('gpus', 'micro_batch'):
        if name in values and int(values[name]) < 1:
            raise ValueError(f'{name} must be positive')
    exports = {({**PATHS, **VALUES})[name][0]: value for name, value in values.items()}
    if local:
        exports['NEXTT_SITE_CONFIG'] = str(local)
    exports['UV_PROJECT_ENVIRONMENT'] = values['venv_dir']
    # The caller chooses the persistent native/upstream resource root explicitly.
    exports['OMP_NUM_THREADS'] = env.get('OMP_NUM_THREADS', '4')
    exports['PYTHONUNBUFFERED'] = '1'
    exports['PATH'] = os.pathsep.join([str(Path(values['venv_dir'])/'bin'), str(Path(root)/'tools'), env.get('PATH', '')])
    return exports


def bootstrap_runtime(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--help' in argv or '-h' in argv:
        return {}
    exports = resolve(argv)
    os.environ.update(exports)
    # Inputs (data, resources, manifests) are never created or modified here.
    inputs = {'runtime_dir', 'venv_dir', 'resources', 'data_dir', 'data_manifest', 'source_manifest', 'core_archive', 'upstream_base_dir', 'wandb_netrc'}
    for name, (key, _) in PATHS.items():
        if name not in inputs and key in exports:
            Path(exports[key]).mkdir(parents=True, exist_ok=True)
    for folder in ('tools', 'runs'):
        (Path(exports['NANOCHAT_RUNTIME'])/folder).mkdir(parents=True, exist_ok=True)
    sys.pycache_prefix = exports['PYTHONPYCACHEPREFIX']
    return exports


if __name__ == '__main__':
    try:
        exports = bootstrap_runtime()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    if '--shell' in sys.argv:
        print('\n'.join(f'export {key}={shlex.quote(value)}' for key, value in exports.items()))
    else:
        print(json.dumps(exports, indent=2))
