# Evidence

What makes the published results admissible. The records shown on the website live in [`results/`](../results);
this directory holds the cost proofs that admit each configuration, the cumulative audits of finished runs, and the
evidence of research checks.

| Path | Contents |
|---|---|
| [`costs/plugin-interface-20261031/`](costs/plugin-interface-20261031/summary.json) | **Current proof set**: full-size cost audits of all 33 registered configurations, with per-method source fingerprints. Every `configs/<family>/<method>.json` points to its proof here. |
| [`costs/mhc-logsinkhorn-20261028/`](costs/mhc-logsinkhorn-20261028/summary.json) | Earlier proof set, after mHC's Sinkhorn-Knopp moved to log space; its counts equal the current set. Kept for runs admitted with it. |
| [`costs/hybrid-kernels-20261026-release/`](costs/hybrid-kernels-20261026-release/summary.json) | Earlier proof set of the frozen source the published hybrid runs trained on. |
| [`costs/layout-20261024/`](costs/layout-20261024/summary.json) | Earlier proof set from before the package reorganization; model hashes, parameter counts and operator counts equal the later sets. Kept for runs admitted with it. |
| [`audits/`](audits) | One cumulative audit per published run that has one, `audits/<method>.json`, written by `python -m nexttransformer.audit_run` (or, for earlier runs, by the same checks) and referenced by `results/runs/<method>.json`. |
| [`checks/arena-v1/`](checks/arena-v1/README.md) | Arena v1: the research gym's acceptance on real data and the proxy and search stages on the dev split, with their run files, the scripts that recompute every check, and the code bundle of every run. Research evidence, not formal results. |

## Cost proofs

A proof records, for one configuration, every counted operator of the forward, backward, auxiliary, recomputation
and optimizer work at full size, weighted by the stages of the training schedule, and the fingerprint of the source
files the method runs. The SSSL anchor is 4.67239832544785 GFLOPs per input token; a configuration is admitted when
it is within 1% of it. `python -m nexttransformer.protocol.fingerprint` checks that every published method still
matches its proof. See [compute accounting](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/compute/)
and [calibration and admission](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/admission/).

## Run audits

An audit re-reads a finished run directory and checks it against the protocol: the ledger, every training,
validation and CORE event, the independent final BPB, the final checkpoint, and the provenance of every segment.
Audits record scientific quantities, fingerprints and the data layout, never host names, paths or devices. Two
historical audits (DeltaNet and GLA) keep their original all-backends-replay accounting, with the source manifest
of their frozen source in [`audits/deltanet-gla-source-manifest.json`](audits/deltanet-gla-source-manifest.json).

Published proofs and audits are never edited. A changed method gets a new proof; formal runs keep the proof and
source fingerprints they were admitted with.
