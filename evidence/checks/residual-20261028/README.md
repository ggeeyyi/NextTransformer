# Residual run evidence — 2026-10-28

Read-only cumulative audits of the two finished Residual runs from the four-node A800 ModelArts job launched on
2026-10-23 ([record 050](../../../docs/records/EXPERIMENTS.md#record-050)). Each node trained one configuration as an
independent eight-GPU run from the frozen checkout at commit `4dd6ebb`.

| File | Evidence | Boundary |
|---|---|---|
| [a800-residual-standard-formal.json](a800-residual-standard-formal.json) | R00 `residual_standard`: periodic val BPB 0.673959, final BPB 0.671214, CORE 0.319513; cumulative +0.041% | One segment, 8×A800, microbatch 8 × accumulation 8, seed 42 |
| [a800-attnres-formal.json](a800-attnres-formal.json) | R03 `attnres`: periodic val BPB 0.662963, final BPB 0.660217, CORE 0.339971; cumulative +0.160% | One segment, 8×A800, microbatch 8 × accumulation 8, seed 42 |
| [audit_a800.py](audit_a800.py) | The auditor, the Dense Baseline audit's checks applied to these run directories | Reads metadata and metrics; does not load or rehash checkpoints |

Both audits report `ok` with no problems: 46,875 updates and 49,152,000,000 input tokens, all 189 validations and six
full CORE-22 events with centering and aggregates reproduced, one final BPB pass, `final_status=matched`, no budget
anomaly, and identical fingerprints across the single provenance segment. The tracked runs (`6462d316`, `e1d85791`)
are finished, and their summary final BPB, periodic val BPB, CORE and cumulative theory FLOPs equal the run directories.

The records were exported with the standard exporter, for example:

```bash
python -m nexttransformer.export_run RUN_DIR --method residual_standard --name "R00 Standard Residual" \
  --hardware "A800 ×8" --audit evidence/checks/residual-20261028/a800-residual-standard-formal.json \
  --commit 4dd6ebb9f4f1f8b4571963c5df2aaa384eea7825
```

HC and mHC from the same job were interrupted on 2026-10-27 and are not part of this evidence.
