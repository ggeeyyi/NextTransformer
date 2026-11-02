# H02 B300 run: read-only metadata audit

Run `h02-b300-formal-20261027T0035`, submitted for issue #13. The owner supplied `H02.zip`
(model card and exported results) and `h02-b300-formal-20261027T0035-rundir-metadata.zip`
(202 metadata files). The original archives remain outside Git. Every run-metadata file's
SHA256 is recorded in the [audit](h02-b300-formal-20261027T0035-formal.json).

The audit passes the supplied metadata's budget, schedule, score, source and checkpoint-marker
checks. The trainer reports `matched`; **prerequisite evidence and maintainer acceptance remain
pending**. Neither metadata consistency nor formal completion proves a pilot or a fresh-process
resume check. Historical H100 numerics and timing are in [record 046](../../../docs/records/EXPERIMENTS.md#record-046);
the timing benchmark's schedule offset 512 is not 512 completed pilot updates. The owner recalls
prechecks before transferring the code, but a separate H02 pilot and target-B300 distributed/resume
report have not been located. Earlier failed attempts are not established by this single-run archive.

All three recorded source hashes reproduce from `fab8c032e9ebc91cc14933d8e7e0275f6c60224f`.
That commit is the original hybrid series before its rebase into PR #11 and is on no branch; tag
[`archive/hybrid-source-20261026`](https://github.com/Selen-Suyue/NextTransformer/tree/archive/hybrid-source-20261026)
keeps it. The original [cost proof](../../costs/hybrid-kernels-20261026-release/gdn_hybrid.json) is byte-identical
to that commit. No new-source audit, model change, training or evaluation was performed.
The serialized model's shared unused fields do not imply looping or a burn-in stage: this is
24 independent layers, `[GDN, GDN, GDN, MHA] × 6`, trained in one `joint` stage.

## Reproduce

Run from a repository checkout containing the frozen commit and the extracted original metadata:

```bash
python evidence/checks/h02-b300-20261030/audit.py RUN_DIR \
  --output /tmp/h02-formal-audit.json
python -m nexttransformer.export_run RUN_DIR --method gdn_hybrid \
  --name "H02 Hybrid GDN + MHA" \
  --hardware "8 x B300 (1 node; device reports NVIDIA B30Z)" \
  --audit evidence/checks/h02-b300-20261030/h02-b300-formal-20261027T0035-formal.json \
  --commit fab8c032e9ebc91cc14933d8e7e0275f6c60224f \
  --note "Frozen source; metadata audit passes; pilot and target-cluster prerequisite evidence pending review."
python -m nexttransformer.export_run --check
```

The shared exporter reads this frozen run schema without changing its recorded identities. The
tracked record preserves every supplied metric, fingerprint, cumulative counter and the original
export's `lastRecordedAt` value; that value is a file timestamp, not a claimed training finish time.
Only publication URLs, hardware-label clarification and the disclosure note change. The old export
pointed at a nonexistent `results/runs/reference-v1/` subdirectory; the new record is flat under
`results/runs/` and links to the real audit. When extracting ZIPs, file timestamps can change; they
do not affect audited JSON contents or scores.

## Limits and resource accounting

Only checkpoint completion metadata was supplied; tensor payloads were not loaded or rehashed.
Resource identity hashes were retained, but dataset/tokenizer/evaluation bytes were not available
for independent content rehashing. Peak memory is the largest logged rank-0 allocated value, not
an all-rank reserved-memory measurement. No MFU was reported.

The owner supplied start `2026-10-28 18:19:27` and end `2026-10-30 01:57:20`, interpreted as UTC+8.
Do not infer the start from the run id's `20261027T0035` label. Elapsed wall time is 31h 37m 53s;
27.4034 h training plus 2.7900 h evaluation leaves 1.4379 h of unclassified overhead. Training
GPU-hours are 219.2274; elapsed wall time × 8 is an estimate of 253.0511 allocated GPU-hours,
assuming all eight devices were allocated for that entire interval. These are separate measures.

After the merge, the maintainer removed the internal row code from the model card's display name
(now "Hybrid GDN + MHA") and limitations, and from the run record's comparison note, which now points
at the archive tag instead of `wxn/hybrid`. The run label `H02 Hybrid GDN + MHA` is unchanged. The
SHA256 values in `validation.json` refer to the files as submitted; no score, identity or counter changed.

See [record 052](../../../docs/records/EXPERIMENTS.md#record-052) for same-metric comparisons and
the [run record](../../../results/runs/gdn_hybrid.json) for the complete curves and task scores.
