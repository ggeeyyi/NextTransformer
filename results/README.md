# Results

The published run records, model cards and arena cards. The website bundles these files at build time. The
[submit-a-result guide](https://interactive-active-ai.github.io/NextTransformer/docs/guides/submit-a-result/) covers the
workflow; the [protocol](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/results/) covers scientific
acceptance.

```text
results/
  runs/<id>.json        one record per formal run, named after its model card
  methods/<id>.json     one model card per method
  arena/<entry>.json    one disclosure card per arena entry (arena/README.md)
```

## Run records

`python -m nexttransformer.export_run RUN_DIR --method <id> --name "<display name>" --audit evidence/audits/<id>.json`
writes a record from a finished run directory; `python -m nexttransformer.export_run --check` validates every
record, and CI runs it. The exporter refuses a run unless all 189 validation and six full CORE-22 events are present,
the per-task CORE scores reproduce the aggregate, the cumulative audit is `matched`, every segment was trained with
`--formal`, and the run used the pinned tokenizer and the frozen protocol.

The file is `runs/<model card id>.json`; `--file NAME` distinguishes a second run of the same method.
`--accept-baseline` records the maintainers' acceptance of a complete, audited `dense_full` run as the Dense baseline.
`cohort` names the cost ledger and evaluation schedule the checks apply (`reference-v1`, the historical `native-v3`
matrix-FLOPs estimate, or `frozen-replay-20261019` for two runs that keep their original replay accounting); it does
not select a separate leaderboard.

| Field | Meaning |
|---|---|
| `id`, `methodId`, `cohort`, `name` | Run id, registered method (SSSL keeps `dense`), accounting group for the checks, display name |
| `status`, `step`, `complete` | `finished` or `partial`; last update; whether the full protocol was met and accepted |
| `val`, `core` | `{step, value}` points: validation BPB at every scheduled step; CORE on its 0–1 scale |
| `coreContext` | Required for new records. Training length, truncation (`none` or `left`), configured cap (null or a token count), and longest evaluated input. The original records are identified by file hash in `core-context.json` and used a 2,048-token left crop. |
| `taskEvaluations` | Per CORE event: `raw` accuracy and `centered` score for each of the 22 tasks |
| `finalBpb` | Independent final validation BPB, reported separately from the curve |
| `trainingCompute` | Cumulative training FLOPs, average GFLOPs per input token, and the accounting used |
| `proof` | Protocol, model, tokenizer, data and source fingerprints of the run |
| `source`, `url` | Provenance of the record, and its evidence in this repository |
| `comparisonNote` | Provenance note shown with the run, when one is needed |
| `baselineAcceptance` | Only on the accepted `dense_full` run: acceptance, protocol hash, source revision and audit |
| `order` | Display order in the observatory (optional) |

## Model cards

`methods/<id>.json` holds the name, family, paper and code links, mechanism, parameter and block counts, configured
compute, limitations, and the architecture diagram the observatory draws. Its `id` is the `methodId` referenced by
run records. A card with `"status": "pending"` is a registered method whose formal result has not been published
yet; the website lists it without ranking it.
