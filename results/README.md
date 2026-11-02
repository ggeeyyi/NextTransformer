# Results

This is the field reference for published run records and model cards. The website bundles these files
at build time. [CONTRIBUTING.md](../CONTRIBUTING.md) owns the method/result submission workflow and required
pull-request contents; the [protocol](../PROTOCOL.md#result-admission-and-release) owns scientific acceptance.

```text
results/
  runs/<id>.json                One record per run, named after its model card; every run in one directory
  methods/<id>.json             One model card per method
  arena/<entry>.json            One disclosure card per auto-research arena entry (arena/README.md)
```

## Run records

`nexttransformer.export_run` produces records from a run directory; `--check` validates published records in CI.
The exporter enforces the protocol's evaluation coverage, per-task aggregate consistency, cumulative `matched`
audit, pinned tokenizer and frozen protocol. Follow [Submitting a run](../CONTRIBUTING.md#submitting-a-run)
for the command and review workflow.

The default file is `runs/<model card id>.json`; `--file NAME` distinguishes a second run of the same method.
`--accept-baseline` writes explicit acceptance for a completed, cumulatively audited `dense_full` run; a
nonempty metric array alone does not confer that status. `cohort` identifies the applicable cost ledger and
evaluation schedule for validation, rather than selecting a separate leaderboard.

| Field | Meaning |
|---|---|
| `id`, `methodId`, `cohort`, `name` | Run id, registered method (SSSL keeps `dense`), cost-ledger group for the checks, run name |
| `status`, `step`, `complete` | `finished` or `partial`; last update; whether the full protocol was met |
| `val`, `core` | `{step, value}` points: validation BPB at every scheduled step; CORE on its 0–1 scale |
| `taskEvaluations` | Per CORE event: `raw` accuracy and `centered` score for each of the 22 tasks |
| `finalBpb` | Independent final validation BPB pass, reported separately from the curve |
| `trainingCompute` | Cumulative training FLOPs, average GFLOPs per input token, and the accounting used |
| `proof` | Protocol, model, tokenizer, data, and source fingerprints of the run |
| `hardware`, `source`, `url` | Execution hardware, provenance of the record, and its evidence in this repository |
| `baselineAcceptance` | Only on the accepted `dense_full` run: the maintainer's acceptance, protocol hash, source commit, and audit report |
| `order` | Display order in the observatory (optional) |

## Model cards

`methods/<id>.json` holds the name, family, paper and code links, mechanism, parameter and block counts, configured
compute, owner, limitations, and the architecture diagram the observatory draws. Its `id` is the `methodId`
referenced by run records. The [new-method workflow](../CONTRIBUTING.md#adding-a-method) specifies when to add it.

Historical DeltaNet and GLA records retain their all-backends-replay accounting and original source
fingerprints. The [frozen-export workflow](../CONTRIBUTING.md#frozen-historical-exports) preserves that
compatibility path; its audit identity and historical limits are in [record 039](../docs/records/EXPERIMENTS.md#record-039); do not reinterpret them as a new-source run.
