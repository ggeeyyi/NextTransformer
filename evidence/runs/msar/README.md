# MSAR result and source correspondence

This is the formal benchmark result for
[Multi-Stream Attention Residuals](https://arxiv.org/abs/2610.13630), by Xiaohan
Qin et al.

## Artifacts

- `metrics.jsonl`: the original retained training/evaluation log, byte-identical
  to the completed run. Its SHA256 is in `submission.json`.
- `checkpoint-manifest.json`: the original final checkpoint completion manifest.
  Checkpoint payloads are stored separately.
- `release-check.json`: bounded comparisons against the frozen training plugin
  and verification that the final model payload loads strictly into the release.
- `submission.json`: historical identity, publication transformations and review
  items. The public result replaces a private snapshot location with the SHA256
  of that exact snapshot manifest; it does not rewrite the training provenance.
- `../../audits/msar.json`: the cumulative audit of the original run.
- `../../../results/runs/msar.json`: full periodic validation and CORE results.
- `../../costs/msar/`: a fresh cost proof for the released code and configuration,
  plus the shared compute anchor.

## Training and release identities

The run used a frozen working-tree snapshot based on commit
`538653bbc67c63396ac27a87b023769be54d2d90`, with the method implementation added.
Model and source hashes are recorded in the run, audit and correspondence record.

The release uses newer upstream code and a simplified configuration schema, with
separate model/source fingerprints. The historical run used SDPA attention.

`release-check.json` records FP32 comparisons at depths 1, 3, 4, 7 and 23, with
width 48, batch two and sequence length seven. Initialization, logits and
gradients match exactly, and optimizer groups are identical. The final model
payload passes its original checksum and loads strictly into the release model.
These checks cover the recorded test cases, not a full training replay.

The released configuration has 1,358,888,937 parameters and a training charge of
4.716253378 GFLOPs per input token, matching the historical run. The training
snapshot, checkpoint payloads and original run directory are retained separately;
this directory contains the exported records. Re-running the cumulative audit
requires the original run directory. Artifact access and launch-history review
are tracked in `submission.json`.

## Regenerate the public views

```bash
python -m nexttransformer.export_run --check
python -m nexttransformer.protocol.fingerprint --methods msar --all
python docs/scripts/generate.py
python docs/scripts/generate.py --check
```

The README chart and website use the final periodic validation BPB.
