# Others run evidence — 2026-10-28

The four formal runs were trained with the frozen Others implementation. This directory records their outcomes for the [progress report](../../../docs/records/PROGRESS-20261028.md); it does not admit an Others implementation into current main or publish new observatory entries.

| File | Evidence | Boundary |
|---|---|---|
| [h800-mqa-formal.json](h800-mqa-formal.json) | Complete MQA run; cumulative audit `matched` | Original audit copied unchanged |
| [h800-gqa-formal.json](h800-gqa-formal.json) | Complete GQA run; cumulative audit `matched` | Original audit copied unchanged |
| [h20-mla-qknorm-formal.json](h20-mla-qknorm-formal.json) | Complete MLA-QKNorm run; read-only cumulative audit passed | One segment, 32 H20 GPUs, seed 42 |
| [h20-mla-nope-qknorm-formal.json](h20-mla-nope-qknorm-formal.json) | Complete MLA-NoPE-QKNorm run; read-only cumulative audit passed | One segment, 32 H20 GPUs, seed 42 |
| [sources.json](sources.json) | Frozen source, original audit locations, launch bundle/commit and verification scope | Null Git commits inside original run provenance remain null |
| [audit_h20.py](audit_h20.py) | The read-only H20 auditor, derived from the frozen branch with supplemental checks | Reads metadata and metrics; does not load or rehash tensor payloads |

All four audits report 46,875 updates, 49.152B tokens, 189 validations and six complete CORE events. The H20 audit additionally checks the canonical 22 task names, score centering and aggregate at every CORE event, final BPB file/event agreement, pinned tokenizer/protocol, and recorded deterministic mode. Its checkpoint evidence is the completion marker plus file inventory, not a fresh tensor-load test. The two H800/H200 source directories were not re-audited for this report.

The `experiments_record` fields and record numbers in the audit text refer to [the frozen branch's records 038–041](https://github.com/Selen-Suyue/NextTransformer/blob/fe8089e3e615776cad622294d4e1a31beb8e6931/reference/EXPERIMENTS.md#record-038). They do not refer to main's similarly numbered records. Historical `Dense` compute labels in the two unchanged audits mean the SSSL cost anchor, **4.67239832544785 GFLOPs/input token**, not the new full-attention Dense quality baseline.

The frozen [31-configuration cost proof set](https://github.com/Selen-Suyue/NextTransformer/tree/fe8089e3e615776cad622294d4e1a31beb8e6931/reference/results/costs/mla-qknorm-20261025) supports those sources only. Main retains its own 26-configuration proof set. Calibration, oracle tests, pilots and the strict-determinism checks remain available in the [frozen evidence index](https://github.com/Selen-Suyue/NextTransformer/blob/fe8089e3e615776cad622294d4e1a31beb8e6931/reference/results/README.md).

To reproduce the H20 audits on a host with read access to the two completed run directories:

```bash
python3 evidence/checks/others-20261028/audit_h20.py --runs RUNS_DIRECTORY
```

The two directory names are `mla_qknorm_h20_seed42` and `mla_nope_qknorm_h20_seed42`. Output consists of two `NEXTT_AUDIT` JSON records; the command does not modify the inputs. O00 `mha_no_ve`, the port to main and new-source admission remain pending.

## Publishing the runs on the observatory

The model cards (`results/methods/{mqa,gqa,mla_qknorm,mla_nope_qknorm}.json`) and the site's Others family are on main, and all four records are exported: [MQA](../../../results/runs/mqa.json), [GQA](../../../results/runs/gqa.json), [MLA + QK-norm](../../../results/runs/mla_qknorm.json) and [MLA-NoPE + QK-norm](../../../results/runs/mla_nope_qknorm.json). Each contains all 189 validation points and the 22 task scores of every CORE event. The H20 trainer writes the same files as `nexttransformer.train`, so the standard exporter reads them; the H800/H200 runs come from the frozen eight-rank trainer and use its adapter. Both need only the standard library; from a checkout of main, on the host that holds the run directory:

```bash
NOTE="Trained from the frozen Others branch (gy/others-attention); the implementation is not yet in the current code."
# H20 runs (provenance has no Git commit; the launch record associates bundle c149799c… with a0c795c)
python -m nexttransformer.export_run RUNS/mla_qknorm_h20_seed42 --method mla_qknorm --name "MLA + QK-norm" \
    --hardware "H20 ×32 (4 nodes)" --audit evidence/checks/others-20261028/h20-mla-qknorm-formal.json \
    --commit a0c795c47eff3bf6fabb50034ddf84ab12028817 --note "$NOTE"
python -m nexttransformer.export_run RUNS/mla_nope_qknorm_h20_seed42 --method mla_nope_qknorm --name "MLA-NoPE + QK-norm" \
    --hardware "H20 ×32 (4 nodes)" --audit evidence/checks/others-20261028/h20-mla-nope-qknorm-formal.json \
    --commit a0c795c47eff3bf6fabb50034ddf84ab12028817 --note "$NOTE"
# H800/H200 runs: the frozen eight-rank trainer (scripts/reference_train.py) writes a flatter provenance and no
# core_final.json, so these use its adapter; the commit is that of the frozen tree wt_a96982a_827b56d023
python -m nexttransformer.export_frozen_8rank RUNS/f3_mqa --method mqa --name "MQA" --hardware "H800/H200 ×8" \
    --audit evidence/checks/others-20261028/h800-mqa-formal.json --commit 8cba0a3b268263ec553862025db65a1159521158 --note "$NOTE"
python -m nexttransformer.export_frozen_8rank RUNS/f3_gqa --method gqa --name "GQA" --hardware "H800/H200 ×8" \
    --audit evidence/checks/others-20261028/h800-gqa-formal.json --commit 8cba0a3b268263ec553862025db65a1159521158 --note "$NOTE"
python -m nexttransformer.export_run --check
```

The exporter refuses a run unless every scheduled validation and full CORE event is present, the CORE aggregates reproduce, the cumulative audit is `matched`, and the tokenizer and protocol match. The commands above reproduce all four records ([records 043 and 049](../../../docs/records/EXPERIMENTS.md#record-049)).
