# H03 B300 run: read-only metadata audit

Run `h03-b300-fix-20261028T1845`, submitted for issue #13. The owner supplied
`h03-b300-fix-20261028T1845-rundir-metadata.zip` (202 metadata files). The original archive
remains outside Git. Every run-metadata file's SHA256 is recorded in the
[audit](h03-b300-fix-20261028T1845-formal.json).

The audit passes the supplied metadata's budget, schedule, score, configuration/cost-identity and
checkpoint-marker checks. The trainer reports `matched`; **prerequisite evidence and maintainer
acceptance remain pending**. Neither metadata consistency nor formal completion proves a pilot or a
fresh-process resume check. Historical H20 checks are in
[record 034](../../../docs/records/EXPERIMENTS.md#record-034) and H100 numerics and timing are in
[record 046](../../../docs/records/EXPERIMENTS.md#record-046); the timing benchmark's schedule
offset 512 is not 512 completed pilot updates. A separate H03 pilot and target-B300
distributed/resume report have not been located. Earlier failed attempts are not established by
this single-run archive beyond the `fix` label in the run id.

The model configuration (`model_hash` `f9fd81f8…`, 22 layers, FFN 6528, `sliding_window=2048`,
1,329,681,662 parameters), protocol (`protocol_hash` `f8cd1b56…`) and every stage/operator cost
match frozen commit `fab8c032e9ebc91cc14933d8e7e0275f6c60224f` (preserved by tag
[`archive/hybrid-source-20261026`](https://github.com/Selen-Suyue/NextTransformer/tree/archive/hybrid-source-20261026)).
The original [cost proof](../../costs/hybrid-kernels-20261026-release/gdn_hybrid_swa.json) is
byte-identical to that commit. The deployed B300 package recorded cluster-patched software source
hashes (`science_hash` `e1b5a94c…`, `reference_source_hash` `9a3aecef…`, `native_source_sha256`
`2822cb47…`); the source diff between `fab8c03` and the cluster fix package was not included in the
metadata-only archive and is disclosed as an open item for maintainer review. No new-source audit,
model change, training or evaluation was performed here.

The serialized model's shared unused fields do not imply looping or a burn-in stage: this is 22
independent layers, `[GDN, SWA] × 11`, trained in one `joint` stage. At the 2048-token training
context, the registered 2048-key window (left window 2047 plus the current token) covers every
causal key, so this run evaluates a 1:1 GDN/full-attention alternation at 2K context.

## Reproduce

Run from a repository checkout containing the frozen commit and the extracted original metadata:

```bash
python evidence/checks/h03-b300-20261031/audit.py RUN_DIR \
  --output /tmp/h03-formal-audit.json
python -m nexttransformer.export_run RUN_DIR --method gdn_hybrid_swa \
  --name "H03 Hybrid GDN + SWA" \
  --hardware "8 x B300 (1 node; device reports NVIDIA B30Z)" \
  --audit evidence/checks/h03-b300-20261031/h03-b300-fix-20261028T1845-formal.json \
  --commit fab8c032e9ebc91cc14933d8e7e0275f6c60224f \
  --note "Frozen base source fab8c03 (tag archive/hybrid-source-20261026) with B300 cluster fix package; model/cost proofs match fab8c03 and metadata audit passes; pilot, target-B300 resume prerequisite evidence and cluster patch diff remain pending maintainer review. Window 2048 spans the full 2K context. Single seed 42; FLA GDN and SDPA without flash. See experiment record 053."
python -m nexttransformer.export_run --check
```

The shared exporter reads this frozen run schema without changing its recorded identities. The
tracked record preserves every supplied metric, fingerprint and cumulative counter; `lastRecordedAt`
is a file timestamp, not a claimed training finish time. When extracting ZIPs, file timestamps can
change; they do not affect audited JSON contents or scores.

## Limits and resource accounting

Only checkpoint completion metadata was supplied; tensor payloads were not loaded or rehashed.
Resource identity hashes were retained, but dataset/tokenizer/evaluation bytes were not available
for independent content rehashing. Peak memory is the largest logged rank-0 allocated value, not
an all-rank reserved-memory measurement. No MFU was reported.

The owner supplied start `2026-10-30 01:58:03` and end `2026-10-31 06:40:30`, interpreted as UTC+8.
Do not infer the start from the run id's `20261028T1845` label. Elapsed wall time is 28h 42m 27s
(28.7075 h); 24.6133 h training plus 2.3737 h evaluation leaves 1.7205 h of unclassified overhead.
Training GPU-hours are 196.9066; elapsed wall time × 8 is an estimate of 229.6600 allocated
GPU-hours, assuming all eight devices were allocated for that entire interval. These are separate
measures.

See [record 053](../../../docs/records/EXPERIMENTS.md#record-053) for same-metric comparisons and
the [run record](../../../results/runs/gdn_hybrid_swa.json) for the complete curves and task scores.
