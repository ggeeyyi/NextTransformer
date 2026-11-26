---
title: Data and evaluation
description: The fixed inputs every method trains on, and the fixed points at which it is evaluated.
---

## Fixed inputs

| Control | Required value and behaviour |
|---|---|
| Upstream source | nanochat `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; the exact project revision and source fingerprints are recorded per run |
| Data | ClimbMix revision `915333b4f8b8684f39aeaafea600fea6f43fb703`; identical files and content |
| Split | Training excludes `shard_06542.parquet`; validation reads only that shard |
| Data order | The same sorted Parquet files, row groups, document stream, global row IDs and eight logical streams |
| Tokenizer | The same prepared 32,768-token vocabulary and byte table, checked by a joint SHA-256 |
| Context | 2,048 tokens; identical shifted targets, label mask, padding and BOS handling |
| Packing | Native BOS best-fit packing; a 1,000-document buffer; 128 documents encoded at a time |
| Boundaries | Attention may cross a BOS inside a packed row; linear-attention state resets at the row boundary, not at each BOS |
| Input budget | **49,152,000,000 tokens**, including warmup, burn-in, gate training and all stages |
| Global batch | **1,048,576 tokens = 512 × 2,048-token rows per optimizer update** |
| Updates | **46,875**; gradient-accumulation microsteps are not optimizer steps |
| Randomness | **Single seed 42**; method-specific routing randomness is documented, independent and restorable |
| Objective | The same next-token objective; auxiliary losses are named and costed separately |
| Precision | BF16 training with the common master-parameter and reduction policy; method-required accumulation exceptions are documented; no per-method FP8 switch |

Data manifests, tokenizer and byte-table hashes, the validation-shard content, evaluation resources, source and
configuration are verified at start-up. A directory name alone is not evidence of identical data.

A repeated internal loop or teacher pass over the same update consumes no additional *input* tokens, but every
computation is charged to the compute budget. Reading input again for another optimizer update consumes tokens again.
A pilot can never supply pretrained weights to a run labelled "from scratch".

## Evaluation

| Event | Schedule |
|---|---|
| Periodic validation | Step 0, every 250 updates and the final update 46,875: **189 events** of **41,943,040 tokens** each |
| CORE | Updates **8,000 / 16,000 / 24,000 / 32,000 / 40,000 / 46,875**: **6 events**, all **22 tasks**, every question; no CORE at step 0 |
| Checkpoints | Every 5,000 updates and the final update; results use the **46,875** checkpoint, never a retrospectively selected best checkpoint |
| Independent final BPB | Training and validation splits of **20,971,520 tokens** each, reported separately from periodic validation |
| Logging axis | The optimizer step, in local logs and in optional tracking |

**Validation BPB** is bits per byte on the held-out shard: lower is better. **CORE** is the mean centered accuracy
over 22 tasks (scored on a 0–1 scale and shown ×100): higher is better. Leaderboard comparisons use the values
measured at update 46,875; nothing is interpolated.

CORE uses the full prompt by default. Pass `--core-max-tokens N` to left-truncate each overlong candidate to its
last N tokens, including the scored answer. Full CORE requires N ≥ 225 to keep every answer and its preceding token.
The original leaderboard results used a 2,048-token left crop. Full prompts
retain their initial BOS and all few-shot examples. Training context, attention windows and BPB retain their usual
settings. Models with a learned distance limit, such as the 2,048-token Inkling REL checkpoint, require an explicit
compatible cap. A training run keeps the same CORE policy across checkpoints and resumed segments.

With the pinned 2,048-token training context, only **3 of 22** tasks contain longer inputs. These are SQuAD
(6,895 of 10,570 questions), BoolQ (185 of 3,270), and Language Identification (1,820 of 10,000). The longest
candidate uses 5,835 tokens. A question counts once if any candidate exceeds the training length.

## Reserved signals

The validation shard and the CORE-22 tasks belong to the formal run. Do not use them to choose among candidates,
settings or seeds. For selection during research, use training loss or the **dev split**, `shard_06541.parquet`, the
last training shard in sorted order: a formal run reads shards from the first one and never reaches it.
