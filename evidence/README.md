# Evidence

Reproducibility artifacts that admission and review depend on. Run results shown on the
website live in [`results/`](../results); this directory holds what makes them admissible.

| Path | Contents |
|---|---|
| [`costs/plugin-interface-20261031/`](costs/plugin-interface-20261031/summary.json) | Current proofs for all 33 registered configurations, after the method-plugin interface ([record 055](../docs/records/EXPERIMENTS.md#record-055)); its counts and file sets equal the previous set. |
| [`costs/method-fingerprints-20261031/`](costs/method-fingerprints-20261031/summary.json) | Preserved proof set that introduced per-method source fingerprints ([record 054](../docs/records/EXPERIMENTS.md#record-054)). |
| [`costs/mhc-logsinkhorn-20261028/`](costs/mhc-logsinkhorn-20261028/summary.json) | Preserved proof set after mHC's log-space Sinkhorn-Knopp ([record 051](../docs/records/EXPERIMENTS.md#record-051)); its counts equal the current set. |
| [`costs/hybrid-main-20261027/`](costs/hybrid-main-20261027/summary.json) | Preserved rebased-source proof set; its counts equal the current set except mHC. |
| [`costs/gsa-removal-20261027/`](costs/gsa-removal-20261027/summary.json) | Preserved main-branch proof set after GSA removal. |
| [`costs/hybrid-kernels-20261026-release/`](costs/hybrid-kernels-20261026-release/summary.json) | Preserved Hybrid kernel-optimization proof set. |
| [`costs/h05-h06-20261026/`](costs/h05-h06-20261026/summary.json), [`costs/h04-shared-20261026/`](costs/h04-shared-20261026/summary.json), [`costs/h01-shared-20261025/`](costs/h01-shared-20261025/summary.json) | Preserved Hybrid implementation and control proof sets. |
| [`costs/layout-20261024/`](costs/layout-20261024/summary.json) | Previous main proof set; GSA was removed by main. |
| [`checks/others-20261028/`](checks/others-20261028/README.md) | Four frozen Others run audits: the original MQA and GQA audits preserved unchanged and two H20 completions audited on 2026-10-28. Historical run evidence; the model port and main admission remain pending. |
| [`checks/residual-20261028/`](checks/residual-20261028/README.md) | Read-only cumulative audits of the finished Residual R00 (`residual_standard`) and R03 (`attnres`) A800 runs, with their auditor; tracked summaries equal the run directories. |
| [`checks/scoring-dense-full-a800-20261028/`](checks/scoring-dense-full-a800-20261028/README.md) | First GPU parity check of the scoring image: the final `dense_full` checkpoint scored bit-identically to the host reference and the training-final record on 8×A800, with the captured reference, preregistered tolerances and comparison. |
| [checks/hybrid-pr-20261028/validation.json](checks/hybrid-pr-20261028/validation.json) | Latest-main scoring/Hybrid compatibility: 588 CPU, 80 Gym and 26 website passes; 33 configs admitted with unchanged scientific source/proofs; no new GPU acceptance |
| [`checks/arena-v1-stage0-20261031/`](checks/arena-v1-stage0-20261031/README.md) | Arena v1 Stage 0: four S-prefix Gym runs (`dense_full` twice, `gdn`, `kda`) on 8×H20 through the MCP tools, with trainer agreement, resume at segment ends and after an abrupt job stop, ledger and sealing checks and their auditor. Tooling evidence, not formal results. |
| [`checks/arena-v1-stage1-20261101/`](checks/arena-v1-stage1-20261101/README.md) | Arena v1 Stage 1: thirteen scale-C (compressed 512-update schedule) Gym runs on 8×H20, eleven methods with published finals plus two `dense_full` seeds, with run checks, the preregistered rank analysis, the screen comparison and the prefix backtest. Tooling evidence, not formal results. |
| [`checks/arena-v1-stage2-20261102/`](checks/arena-v1-stage2-20261102/README.md) | Arena v1 Stage 2: seventeen scale-M (first 2,000 updates of the formal schedule) Gym runs on 8×H20 in three search rounds, with the anchors, a noise repeat, run checks, the preregistered screen and decision, Koala job times and every candidate's submission. Research evidence, not formal results. |
| [`checks/`](checks) | Bounded checks and cumulative audits of completed runs; each record retains its own validation scope. |

## Current proof set

[plugin-interface-20261031](costs/plugin-interface-20261031/summary.json) re-audits every structure after the method-plugin
interface (record 055): counts and audited files equal [method-fingerprints-20261031](costs/method-fingerprints-20261031/summary.json),
which introduced per-method source fingerprints (record 054): each proof covers only the source files its method runs, so a change elsewhere keeps it
valid. All 33 keep their model/protocol hashes, parameter counts, stage weights and every operator count from the
previous set. SSSL's configuration moved to `configs/others/` with its reclassification as Others.

mHC's Sinkhorn-Knopp iterates in log space (record 051). Re-audited under that source, 32 structures keep
their model/protocol hashes, parameter counts, stage weights and every operator count; mHC's normalizations add
6,072 FLOPs per input token (+0.042% versus SSSL). The rebased source preserves H01–H06, their controls and kernel
optimizations while adopting main's GSA removal.
The SSSL anchor remains 4.67239832544785 GFLOPs/input token. Integration evidence is in
[hybrid-main-20261027](checks/hybrid-main-20261027/validation.json). New interface checks do not constitute formal training.

## Archive

Superseded proof sets and historical diagnostics are preserved at the tag
[`evidence-archive-20261024`](https://github.com/Selen-Suyue/NextTransformer/tree/evidence-archive-20261024/reference/results),
where they sit under their original `reference/results/` paths:

- Cost proof sets `elu-precision-20261020`, `elu-qknorm-20261020`, `gdn-20261022`,
  `gdn-fused-20261022`, `merged-20261022`, `layout-20261023`, `v2-selen-20261023`,
  `hybrids-20261024`, `sdpa-window-20261024`, and the 19 earliest root-level proofs.
- CPU regression logs, ELU numerics/precision diagnostics, the KDA speed A/B, and the validation
  directories of the 2026-10-22 merge and the 2026-10-23 cleanup.

Formal runs record the proof and source fingerprints they were admitted with; resume a historical
run from its frozen source snapshot, not from a newer tree.
