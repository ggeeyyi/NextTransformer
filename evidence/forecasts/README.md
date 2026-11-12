# Scaling forecasts

Predictions registered before the test runs of the scaling stage start. A forecast is merged into `main` before the
first C200 run of its family begins; the merge commit dates it. Forecasts are never edited: the outcome goes into a
separate `<forecast>.result.json`, written after the test run's record is published, whether the prediction held or not.

One file per forecast, `<family>-<budget>-<author>.json`:

| Field | Contents |
|---|---|
| `family`, `budget` | Ladder family id and tier id, for example `dense` and `C200` |
| `author` | GitHub handle of the person or team registering it |
| `protocol` | The ladder's protocol version, for example `nextt-scale-4k-v1` |
| `records` | Ids of the scaling records the fit uses, from `results/scaling/runs/` |
| `method` | How the prediction was made: estimator, fitted form and fitted parameters, or a link to the code |
| `optimum` | Predicted compute-optimal size at the budget: `layers`, `parameters`, `tokens` |
| `valBpb`, `core` | Predicted final validation BPB and CORE of that optimum, each as `{"value", "low", "high"}`, a 90% interval |

See the [scaling protocol](https://interactive-active-ai.github.io/NextTransformer/docs/scaling/protocol/) for how
forecasts are tested.
