# H03-SWA512 evidence

This directory publishes the evidence for `gdn_hybrid_swa512`, the 512-key configuration of the existing `gdn_hybrid_swa` implementation at revision `4926c5de832968957cd84411b1fa24ca0ada81c4`.

- `metrics.jsonl` is the original retained training/evaluation log, byte-for-byte equal to the archived final snapshot.
- `audit-inputs.json` bundles the trainer's JSON audit inputs. Private site fields are removed as listed in `submission.json`. Scientific values and fingerprints are retained.
- `configs/hybrid/gdn_hybrid_swa.json` and `proofs/` are the original configuration and operator-level cost proofs. Their relative paths are preserved. These proofs cover the frozen training revision.
- `execution/` contains the exact external wrapper sources collected from the continuation host. They record the execution path used for this run and depend on its site layout.
- `core-graph-check.json`, `core-input-cache-manifest.json` and `resume-replay-check.json` retain the bounded execution checks. The checkpoint-7500 CORE check is separate from the scheduled evaluations.
- `checkpoint-manifest.json` records the final model and eight rank payload hashes, sizes and verified archive generations. The contributor archive retains the payloads.
- `submission.json` records source identity, original provenance hashes, wrapper hashes, public artifact hashes and redactions.

The published result and cumulative audit are `results/runs/gdn_hybrid_swa512.json` and `evidence/audits/gdn_hybrid_swa512.json`. The public ID distinguishes the SWA-512 card and leaderboard row. The original training method name remains in the configuration and provenance.

## Reproduce the cumulative audit and result

Run from the repository root after installing its pinned CPU dependencies. This checks the public files against the submission manifest, reconstructs the audit inputs in a temporary directory, reproduces the cumulative audit, and compares the exported scientific result. The audit date and original final-evaluation file timestamp are metadata supplied by the publication.

```bash
uv run python - <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from nexttransformer.audit_run import audit
from nexttransformer.export_run import read_run, record

root = Path('evidence/runs/gdn_hybrid_swa512')
manifest = json.loads((root / 'submission.json').read_text())
for name, expected in manifest['artifacts'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
for name, expected in manifest['runtime_sources'].items():
    assert hashlib.sha256((root / 'execution' / name).read_bytes()).hexdigest() == expected, name

expected_audit = json.loads(Path('evidence/audits/gdn_hybrid_swa512.json').read_text())
published = json.loads(Path('results/runs/gdn_hybrid_swa512.json').read_text())
with tempfile.TemporaryDirectory() as temporary:
    run = Path(temporary)
    for name, value in json.loads((root / 'audit-inputs.json').read_text()).items():
        (run / name).write_text(json.dumps(value))
    shutil.copyfile(root / 'metrics.jsonl', run / 'metrics.jsonl')
    result, problems = audit(run, 'gdn_hybrid_swa512')
    assert not problems, problems
    result = json.loads(json.dumps(result))
    result['date'] = expected_audit['date']
    assert result == expected_audit
    exported, _ = record(read_run(run), 'gdn_hybrid_swa512', published['name'],
                         audit='evidence/audits/gdn_hybrid_swa512.json',
                         note=published['comparisonNote'])
    exported['lastRecordedAt'] = published['lastRecordedAt']
    assert exported == published
print('Artifact hashes, cumulative audit and exported result match.')
PY
```

The final checkpoint payload verification is recorded separately in `checkpoint-manifest.json`. It rehashed the actual model and rank files on the training host and compared them with the complete manifest and archive readback receipts. The public audit reproduction uses the recorded training evidence.
