# Native training metrics

`nexttransformer.native.train` uses the `native-v3` schema for historical SSSL (`dense`), MoR, LoopFormer-Fixed2, and Circle Sparse. The default `--wandb-metrics core` is a compact **logging profile**, separate from the CORE benchmark. Schema/profile are stored in W&B config and local `tracking.json`.

## Default uploads

Counts exclude the `step` axis and SDK timestamps/system metrics. Training logs every 10 optimizer updates by default, plus first and final.

| Event / model | Count | Fields |
|---|---:|---|
| All training | 9 | `lm_loss`, `loss`, `input_tokens`, `training_flops_estimate`, `tokens_per_second`, `grad_norm`, `lr_multiplier`, `gpu_hours`, `peak_memory_bytes` |
| LoopFormer training | 11 total | Also `loop_condition_grad_norm`, `loop_modulation_saturated_fraction` |
| MoR training | 13 total | Also `router_aux_loss`, `mean_active_depth`, `mean_recursions`, `budget_max_abs_error` |
| Circle training | 16 total | Also `mean_ring_steps`, `cap_ratio`, `budget_target_steps`, `budget_error_ema`, `budget_weight_used`, `router_aux_loss`, `budget_shape_rmse` |
| SSSL / LoopFormer validation | 1 | `val_bpb` |
| MoR validation | 2 | `val_bpb`, `mean_recursions` |
| Circle validation | 3 | `val_bpb`, `mean_ring_steps`, `cap_ratio` |
| All CORE | 25 | `core_metric`, 22 `results/<task>` fields, `full_core`, `entries` |
| All final BPB | 2 | `train_bpb`, `val_bpb` from the independent final budget |

Keys carry event prefixes: `train/lm_loss`, `validation/val_bpb`, `core/core_metric`, `final_bpb/val_bpb`. Final BPB and periodic validation use different budgets and keys, so one cannot overwrite the other at the same step. Logging profiles do not change evaluation.

MoR `budget_max_abs_error=max(abs(actual_survival-target))` is the largest absolute survival deviation across its two optional recursions, aggregated globally per optimizer update. A value of 0.03 means three percentage points. Circle `mean_ring_steps` excludes prefix/suffix; `cap_ratio` measures tokens reaching ring block 27, not 27 complete loops. `mean_active_depth` includes prefix/suffix.

LoopFormer's condition-gradient norm is measured before its separate clip. Saturation means absolute modulation offset >0.475, against a 0.5 cap. `loop_modulation_absmax` remains local or in the full profile.

## Local diagnostics

`metrics.jsonl` retains depth survival/exit distributions, per-loop MoR errors/duals, Circle controller state, timing, and compute components. Updated `halter_delta_scale` and 18 `halter_depth_bias` values are logged locally at each training-log event.

`--wandb-metrics full` (or JSON `"wandb_metrics": "full"`) uploads all numerical event fields, including flattened arrays. Checkpoints, datasets, source, and console text are not uploaded. Detailed CORE output stays in result JSON; standalone evaluation writes local results only.

```bash
bash launch/native_a800.sh circle --site-config site.native.local.json \
  --wandb-mode online --wandb-metrics core --wandb-group native-study
```

The example site config defaults to `disabled`. Manage credentials separately as described in [configuration](CONFIGURATION.md). Offline SDK integration tests cover schema/filtering/final BPB, not online server writes. Historical runs keep their frozen logging code and cloud history.
