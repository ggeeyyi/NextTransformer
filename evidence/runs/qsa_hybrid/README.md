# H05 evidence

This directory records the completed `qsa_hybrid` formal run at frozen revision `4926c5de832968957cd84411b1fa24ca0ada81c4`.

- `metrics.jsonl` is the original retained training/evaluation log, byte-identical to its archived final snapshot.
- `audit-inputs.json` bundles the original trainer JSON inputs with private site fields removed as listed in `submission.json`. Scientific values and fingerprints are preserved.
- `configs/` and `proofs/` retain the original admitted configuration and cost proofs, including separate dense burn-in and joint sparse stages.
- `execution/` contains the exact site wrapper sources collected from the continuation host. They depend on the original site layout and are archived for review.
- `core-graph-check.json` is the full checkpoint-7500 numerical execution check. `core-tail-check.json` checks only the last 64 shape groups per rank after the resource handoff failure. Neither is a scheduled benchmark evaluation.
- `resume-replay-check.json` compares the two pre-restart histories with the canonical log at overlapping recorded steps.
- `rejected-core-prototypes.json` records excluded execution experiments and a correction to their debug reporting.
- `checkpoint-manifest.json` records the final model and eight rank payload hashes, sizes and archive generations.
- `submission.json` records provenance, redactions and hashes of the public artifacts and external wrapper sources.

The published result and cumulative audit are `results/runs/qsa_hybrid.json` and `evidence/audits/qsa_hybrid.json`. The existing method identity is preserved.

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

root = Path('evidence/runs/qsa_hybrid')
manifest = json.loads((root / 'submission.json').read_text())
for name, expected in manifest['artifacts'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
for name, expected in manifest['runtime_sources'].items():
    assert hashlib.sha256((root / 'execution' / name).read_bytes()).hexdigest() == expected, name

expected_audit = json.loads(Path('evidence/audits/qsa_hybrid.json').read_text())
published = json.loads(Path('results/runs/qsa_hybrid.json').read_text())
with tempfile.TemporaryDirectory() as temporary:
    run = Path(temporary)
    for name, value in json.loads((root / 'audit-inputs.json').read_text()).items():
        (run / name).write_text(json.dumps(value))
    shutil.copyfile(root / 'metrics.jsonl', run / 'metrics.jsonl')
    result, problems = audit(run, 'qsa_hybrid')
    assert not problems, problems
    result = json.loads(json.dumps(result))
    result['date'] = expected_audit['date']
    assert result == expected_audit
    exported, _ = record(read_run(run), 'qsa_hybrid', published['name'],
                         audit='evidence/audits/qsa_hybrid.json',
                         note=published['comparisonNote'])
    exported['lastRecordedAt'] = published['lastRecordedAt']
    assert exported == published
print('Artifact hashes, cumulative audit and exported result match.')
PY
```

The final checkpoint payload verification is recorded separately in `checkpoint-manifest.json`. It rehashed the actual model and rank files on the training host and compared them with the complete manifest and archive readback receipts. The public audit reproduction uses the recorded training evidence.
