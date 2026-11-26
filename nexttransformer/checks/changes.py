"""What a pull request changes, and which checks it needs.

    python -m nexttransformer.checks.changes --base origin/main
    python -m nexttransformer.checks.changes --base origin/main --github-output "$GITHUB_OUTPUT"

CI and contributors use the same rules:

- `tests`: the fast test suite runs unless only results, run audits, documentation or the website changed.
- `site`: the website and documentation are built when they, or the published results, changed.
- `contract`: files of the training contract changed (protocol, cost ledger, evaluation, trainer, data
  pipeline, pinned upstream files, protocol pages). Pull requests from outside contributors that touch
  them need a maintainer's approval first.
- `affected`: published methods whose fingerprinted source files changed. `protocol.fingerprint` decides
  whether their proofs are stale; a stale proof needs the maintainer's full audit before merging.

Reads git only; does not import torch.
"""
import argparse
import json
import subprocess
from pathlib import Path

from nexttransformer.models.registry import FAMILY
from nexttransformer.protocol.fingerprint import CORE, DISPLAY
from nexttransformer.protocol.spec import config_path

REPO = Path(__file__).resolve().parents[2]
DOCS = "docs/src/content/docs/"
NO_TESTS = ("results/", "evidence/audits/", "docs/", "website/", ".github/ISSUE_TEMPLATE/",
            ".github/PULL_REQUEST_TEMPLATE.md", "README.md", "README.zh-CN.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md",
            "SECURITY.md", "CITATION.cff", "AGENTS.md", "THIRD_PARTY_NOTICES.md", "LICENSE", ".github/assets/")
SITE = ("website/", "results/", "docs/", "moe/results/", "moe/configs/", "moe/evidence/", "moe/methods/", ".github/workflows/pages.yml")
CONTRACT = ("nexttransformer/protocol/", "nexttransformer/evaluation/", "nexttransformer/training/",
            "nexttransformer/runtime/", "nexttransformer/train.py", "nexttransformer/evaluate.py",
            "nexttransformer/prepare.py", "nexttransformer/models/backbone.py", "nanochat/",
            DOCS+"protocol/", DOCS+"zh-cn/protocol/")
CARD_ALIASES = {"sssl": "dense"}


def changed(base):
    """`(status, path)` rows changed between the merge base with `base` and HEAD."""
    git = lambda *args: subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout
    merge_base = git("merge-base", base, "HEAD").strip()
    rows = git("diff", "--name-status", "--no-renames", merge_base, "HEAD").splitlines()
    return [tuple(row.split("\t", 1)) for row in rows if row]


def published():
    """Method ids with a published run."""
    ids = {json.loads(p.read_text())["methodId"] for p in (REPO/"results/runs").glob("*.json")}
    return sorted(CARD_ALIASES.get(i, i) for i in ids)


def fingerprinted(method):
    """Repository files whose code the method's cost proof covers (none for the historical native track)."""
    if method not in FAMILY:
        return set()
    path = config_path(method)
    if not path.exists():
        return set()
    proof = json.loads((path.parent/json.loads(path.read_text())["calibration"]["cost_record"]).read_text())
    files = set((proof.get("method_source") or {}).get("files", ())) | set(CORE)
    return files - set(DISPLAY)


def classify(paths, methods=None):
    paths = list(paths)
    methods = published() if methods is None else methods
    tests = any(not p.startswith(NO_TESTS) for p in paths)
    site = any(p.startswith(SITE) for p in paths) or any(p.startswith(".github/workflows/") for p in paths)
    contract = sorted(p for p in paths if p.startswith(CONTRACT))
    touched = set(paths)
    affected = sorted(m for m in methods if fingerprinted(m) & touched)
    return dict(tests=tests or any(p.startswith(".github/workflows/") for p in paths), site=site, contract=contract,
                affected=affected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--github-output", type=Path, help="append tests/site/contract/affected for GitHub Actions")
    args = parser.parse_args(argv)
    paths = [path for _, path in changed(args.base)]
    result = classify(paths)
    print(f"{len(paths)} changed file(s)")
    print(f"fast tests:      {'run' if result['tests'] else 'skipped (results, audits, documentation or website only)'}")
    print(f"site build:      {'run' if result['site'] else 'skipped'}")
    if result["contract"]:
        print("training contract changed: " + ", ".join(result["contract"]))
    if result["affected"]:
        print("published methods whose fingerprinted files changed: " + ", ".join(result["affected"]))
        print("  python -m nexttransformer.protocol.fingerprint decides whether their proofs are stale.")
    if args.github_output:
        with open(args.github_output, "a") as out:
            out.write(f"tests={str(result['tests']).lower()}\nsite={str(result['site']).lower()}\n")
            out.write(f"contract={str(bool(result['contract'])).lower()}\naffected={','.join(result['affected'])}\n")


if __name__ == "__main__":
    main()
