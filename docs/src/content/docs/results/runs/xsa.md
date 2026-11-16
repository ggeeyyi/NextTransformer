---
title: XSA · Exclusive Self Attention
description: Full 49.152B-token run with original protocol scores and post-run BOS diagnostics.
---

At 46,875 updates, `xsa` reached **0.666837 periodic validation BPB**, **0.664079 final validation BPB**, **0.666130 final training BPB**, and **31.7346 CORE ×100**. Final SQuAD is **18.0416%** under the unchanged evaluator.

The calibrated model has **23 layers**, width **1536**, **12 heads**, FFN width **6144**, and **1,355,810,568 parameters**. Training charge is **4.675544617 GFLOPs per input token**, **+0.06734%** versus the SSSL anchor. The cumulative charge matched the proof with no budget anomalies.

## Protocol and execution

Training uses the frozen data, tokenizer, eight-stream data order, seed 42, 2048-token context, Muon + AdamW schedule, 49.152B input tokens and 1,048,576 tokens per update. All 189 validation events and six full CORE-22 events completed. The run used 64 ranks with microbatch eight and accumulation one, following bounded full-update and save/resume preflights on training data. Formal training completed in one segment without a training retry. Calibration and preflight checks did not use held-out scores for candidate selection.

**Execution layout:** this run used 64 ranks, which [#57](https://github.com/Interactive-Active-AI/NextTransformer/pull/57) registered in the run audit, so the unchanged audit command accepts it and the checked-in audit reports no problems. `data_layout` drives the same eight `BestFitStream`s from any multiple of eight ranks, and 64 x 8 x 1 x 2048 = 1,048,576 matches the protocol's global batch, so the data each update sees is identical to an 8-, 16- or 32-rank run. Machine-specific launch scripts are not included in the method implementation.

The frozen training source commit is `9dd94c5540fdfb31190feb466f2669bb581a50c6`. Plugin code and its scientific fingerprint match the submitted implementation; private source-directory paths are omitted from the exported record. No training provenance or original score file was edited.

An independent eight-rank scorer loaded the final checkpoint and reproduced all 22 CORE task scores and final validation BPB exactly. Training BPB agrees within 1.2e-7. The final independent evaluation and cumulative local audit completed successfully.

## BOS changes result interpretation

Training best-fit rows begin with BOS, while CORE left truncation removes it from 6,895 of 10,570 SQuAD examples. Post-run paired diagnostics reproduce the original correct-answer count exactly. Replacing only the first retained token with BOS on cropped inputs raises full diagnostic SQuAD to **50.4352%** (**+32.3936 percentage points**), retaining sequence length and all other tokens. These modified-input scores are diagnostic evidence and do not replace the original published SQuAD or CORE values.

The equally restored Dense diagnostic scores 53.6329%, versus its same-run original 50.0378% (four examples above the published 50.0000%). This evaluation-format sensitivity means the raw CORE ranking should not alone be read as an architecture-quality verdict. It does not explain every task: REL's bounded cropped BoolQ controls decline after restoration, Language ID evidence is inconclusive, and LSAT-AR does not truncate. No claim is made that an attention-sink circuit or kernel defect has been identified.

[Full counts and controls](https://github.com/Interactive-Active-AI/NextTransformer/tree/main/evidence/diagnostics/xsa-rel-bos-20261116) · [Run record](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/xsa.json) · [Cumulative audit](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/xsa.json)
