---
title: Versions and identifiers
description: The protocol and arena identifiers stored in records, and how they change.
---

| Name | Identifier stored in data | Meaning |
|---|---|---|
| Reference Protocol v1 · 50B | `nextt-reference-50b-v1` | The training contract on these pages |
| Cost ledger v2 | `reference-aten-math-v2` | The operator-level accounting on [compute accounting](../compute/) |
| Arena Protocol v1 | `arena-v1` | The rules for auto-research entries, until `arena-v2` |
| Arena Protocol v2 | `arena-v2` | Arena Protocol v1 plus the `arena-only` status: an entry shown in the arena but not ranked on the leaderboard |

The protocol identifier is part of the protocol's SHA-256 fingerprint, which every configuration and run record
carries; it therefore never changes for an existing protocol. The website and documentation show the display name.

A change to any controlled value defines a new protocol with a new identifier and fingerprint. Runs, proofs and
arena cards keep the version they were made under and are never rewritten. Documentation can be reorganized without
changing a protocol.
