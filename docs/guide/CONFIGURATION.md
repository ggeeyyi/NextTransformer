# Local configuration and launchers

Machine paths belong in CLI arguments or an ignored local JSON file, not scientific configurations. This guide owns machine paths, caches, tracking and launchers, including the explicitly marked historical native workflow. For the reference trainer and multi-node layouts, use the [training guide](METHODS_FRAMEWORK.md). The scientific contract is [PROTOCOL.md](../../PROTOCOL.md).

Docker is not required for training, evaluation, or pull requests that submit results or method code: use this guide and the [training guide](METHODS_FRAMEWORK.md). The fixed GPU runtime serves the arena gym's isolated training and scoring; use its [operational guide](../../environments/eval/README.md) and the [consolidated acceptance status](../../PLAN.md#fixed-runtime-subplan-for-g3).

## Setup

This prepares shared resources. For current registered methods, continue with the [reference trainer](METHODS_FRAMEWORK.md#train) or [reference ModelArts launcher](#reference-modelarts-launcher). The `native_a800.sh` checks below belong only to the historical native track.

```bash
cp site.native.example.json site.native.local.json
# Set persistent runtime_dir, venv_dir, resources, upstream_base_dir, and data_dir.
# Set upstream_base_dir to the same prepared resource directory as resources.
bash launch/setup_environment.sh --site-config site.native.local.json --tracking --upstream
source launch/env.sh --site-config site.native.local.json
python -m nexttransformer.prepare --site-config site.native.local.json
bash launch/native_a800.sh circle --site-config site.native.local.json --dry-run
bash launch/native_a800.sh circle --site-config site.native.local.json --preflight
```

Preparation reads original ClimbMix Parquet, trains the native 32K tokenizer, and prepares pinned CORE resources. Reuse a verified bundle with `--existing-core-resources PATH`. It does not pre-encode a 49B-token file or use the former GPT-2 resources. Original upstream byte hashes remain enforced after the [model relocation](REPOSITORY.md).

## Precedence and secrets

**CLI > explicit local JSON > exported environment > derived defaults.** JSON-relative paths resolve against that file; CLI-relative paths resolve against the working directory. Underscored JSON fields use hyphenated CLI names, such as `venv_dir` / `--venv-dir`. Resolving settings does not create input files.

Local JSON, `.env.*`, and W&B logs stay out of Git. Use standard authentication, secret environment variables, or a mounted netrc file; JSON does not accept API keys. Changing paths does not relax source, scientific configuration, tokenizer, data, or world-size resume checks.

## Paths

`R` is the explicitly configured persistent runtime root.

| CLI / JSON | Environment | Default / purpose |
|---|---|---|
| `--runtime-dir` / `runtime_dir` | `NANOCHAT_RUNTIME` | Required persistent root |
| `--venv-dir` / `venv_dir` | `NANOCHAT_VENV` | `R/venv` |
| `--resources` / `resources` | `NEXTT_RESOURCES` | `R/resources`; tokenizer, CORE, fingerprints |
| `--data-dir` / `data_dir` | `NEXTT_DATA_DIR` | Original ClimbMix directory; required for preparation |
| `--source-manifest` / `source_manifest` | `NEXTT_SOURCE_MANIFEST` | Optional official file manifest; pinned revision checked |
| `--core-archive` / `core_archive` | `NEXTT_CORE_ARCHIVE` | Optional local ZIP; pinned SHA256 checked |
| `--upstream-base-dir` / `upstream_base_dir` | `NANOCHAT_BASE_DIR` | Default `R/upstream`; explicitly set to `resources` for this workflow |
| `--output` | `OUTPUT_DIR` in launchers | Exact training/evaluation destination |
| `--run-name` / `run_name` | `NEXTT_RUN_NAME` | Optional run directory and W&B display name |
| `--wandb-dir` / `wandb_dir` | `WANDB_DIR` | `output/wandb/` |
| `--wandb-netrc` / `wandb_netrc` | `NETRC` | Mounted credential file; contents never enter provenance |

The legacy `data_manifest` field is accepted for compatibility; current preparation does not read GPT-2 token manifests. Without an output/run name, native A800 launchers use `R/runs/native_dense_a800_seed42`, `native_mor_core393_a800_seed42`, `native_loopformer_core393_a800_seed42`, and `native_circle_sparse_a800_seed42`. H200 uses its corresponding hardware suffix. Changed architectures require fresh runs.

## Persistent caches

Every JSON field below also has a hyphenated CLI override.

| JSON field | Environment | Default |
|---|---|---|
| `hf_home` | `HF_HOME` | `R/huggingface` |
| `hf_hub_cache` | `HF_HUB_CACHE` | `HF_HOME/hub` |
| `tiktoken_cache_dir` | `TIKTOKEN_CACHE_DIR` | `R/cache/tiktoken` |
| `torch_home` | `TORCH_HOME` | `R/cache/torch` |
| `triton_cache_dir` | `TRITON_CACHE_DIR` | `R/cache/triton` |
| `torchinductor_cache_dir` | `TORCHINDUCTOR_CACHE_DIR` | `R/cache/inductor` |
| `cuda_cache_path` | `CUDA_CACHE_PATH` | `R/cache/cuda` |
| `xdg_cache_home` | `XDG_CACHE_HOME` | `R/cache/xdg` |
| `pythonpycacheprefix` | `PYTHONPYCACHEPREFIX` | `R/cache/pycache` |
| `uv_python_install_dir` | `UV_PYTHON_INSTALL_DIR` | `R/python` |
| `uv_cache_dir` | `UV_CACHE_DIR` | `R/cache/uv` |
| `pip_cache_dir` | `PIP_CACHE_DIR` | `R/cache/pip` |
| `tmpdir` | `TMPDIR` | `R/tmp` |
| `wandb_cache_dir` | `WANDB_CACHE_DIR` | `R/cache/wandb` |
| `wandb_data_dir` | `WANDB_DATA_DIR` | `R/cache/wandb-data` |
| `wandb_artifact_dir` | `WANDB_ARTIFACT_DIR` | `R/artifacts` |
| `wandb_config_dir` | `WANDB_CONFIG_DIR` | `R/wandb-config` |

## Training and hardware

| Option / JSON field | Meaning |
|---|---|
| `--gpus` / `gpus` | Processes on one node; default 8, not total multi-node GPUs |
| `--micro-batch` / `micro_batch` | Sequences per rank; launcher defaults from `configs/native/hardware.json` |
| `--config` | Scientific JSON; native launchers select by model |
| `--resume` | Complete checkpoint directory; strict source/config/data/world checks |
| `--log-every` | Default 10 optimizer updates; also first and final |
| `--skip-final-core` | Skip only final CORE, retaining periodic CORE; not a complete formal evaluation |
| `--smoke-steps`, `--smoke-val-tokens` | Explicit bounded tests; smoke checkpoints cannot become formal training |
| `--circle-stage-step` | Smoke-only curriculum offset |
| `--world-size` | Dry-run estimate only; actual execution uses torchrun |
| `--python-version` / `python_version` | Setup default 3.10.19 |
| `--index-url` / `index_url` | Optional install mirror; otherwise existing uv/pip settings |
| `--hf-endpoint` / `hf_endpoint` | Optional Hub mirror |
| `--core-url` / `core_url` | Optional CORE mirror; fixed SHA256 still checked |

At eight ranks and context 2048, `micro_batch × accumulation = 64`. A800 defaults to 8×8; H200 16×4 is an unmeasured preflight starting point. Preflight reports capacity and halves the batch after OOM; it does not start formal training. See [hardware evidence](../history/NATIVE_VALIDATION.md).

The scientific evaluation schedule is defined in [PROTOCOL.md](../../PROTOCOL.md). Dry-run prints the selected schedule; `--skip-final-core` omits required formal evidence and cannot produce a complete result.

Use environment variables for `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS` (default 4), NCCL, proxy, and offline settings. Four independent eight-GPU native experiments are four separate DDP groups, not one 32-rank run.

### Historical four-node ModelArts launcher

```bash
python3 launch/modelarts_native.py --site-config site.native.local.json \
  --run-prefix native-study --hardware a800 --node-index 0 --dry-run
```

The entry point maps nodes 0–3 to the four native models. `--node-index` supports offline dry-runs; runtime node metadata selects the actual node. Hardware is `a800` or `h200`. The run prefix is the default W&B group unless overridden. Use a single-model launcher for `--output`, `--run-name`, `--config`, `--resume`, `--gpus`, or `--world-size` to avoid shared run/checkpoint destinations. Paths, tracking, and microbatch overrides remain available per node.

### Reference ModelArts launcher

```bash
python3 launch/modelarts.py --site-config site.native.local.json \
  --configs configs/residual/residual_standard.json configs/residual/hc.json \
            configs/residual/mhc.json configs/residual/attnres.json \
  --micro-batches 8 8 8 8 --run-prefix nextt-v2-residual --node-index 1 --dry-run
```

Pass one admitted configuration per node, in node order. Node i runs configuration i as a standalone eight-rank `nexttransformer.train --formal` group (a training option `--research` overrides the mode), so four configurations on four nodes are four independent eight-GPU experiments, not one 32-rank run. A single configuration is an ordinary one-node job. The launcher checks `MA_NUM_HOSTS`/`VC_WORKER_NUM` against the configuration count, requires eight GPUs per node, takes the node from `VC_TASK_INDEX` (or `--node-index`), and refuses per-node options such as `--output` or `--config`. Outputs are `RUNTIME/runs/<prefix>_<method>_<hardware>_seed42`; resubmitting the same prefix resumes from the newest complete checkpoint. `--micro-batches` sets each node's per-rank microbatch; accumulation follows from the 1,048,576-token global batch.

## Tracking

```bash
bash launch/native_a800.sh loopformer --site-config site.native.local.json \
  --wandb-mode offline --wandb-group native-study --dry-run
```

Modes are `disabled` (default), `offline`, and `online`. Project, entity, group, and metric profile accept JSON, CLI, and supported environment settings. `wandb_metrics=core` uploads the compact profile; `full` includes all numerical diagnostics. See [metric definitions](WANDB_METRICS.md).

Rank 0 logs numbers, scientific config, and provenance; it does not automatically upload models, data, source, or console text. The SDK may collect system metrics. Training throughput excludes logging, evaluation, and checkpoint time. Standalone native evaluation writes JSON only.

Forward online resume in the same output directory reuses the run ID and requires the same project/entity. Replaying an older checkpoint, offline resume, or mode changes create a new segment with parent provenance. Tracking choices never relax training-state checks.

## Standalone evaluation and resource migration

```bash
source launch/env.sh --site-config site.native.local.json
python -m torch.distributed.run --standalone --nproc_per_node=8 -m nexttransformer.evaluate \
  --checkpoint /persistent/run/checkpoint_00046875 --output /persistent/run/evaluation.json \
  --micro-batch 8 --eval bpb,core
```

`--split-tokens` defaults to 20,971,520 for each train/val split. `--max-per-task` leaves CORE unrestricted by default; reduced budgets are test-only. Moving resources requires unchanged Parquet contents, tokenizer, byte table, native manifest, CORE bundle and manifest, plus reconnected raw-data paths.

The old upstream speedrun, CPU, chat, SFT and RL entry points are not part of this maintained tree. Historical paths and removals are documented in [the repository guide](REPOSITORY.md); recover historical checkpoints using their original frozen checkout and [cleanup record](../history/CLEANUP.md).
