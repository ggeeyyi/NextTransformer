"""Scan tracked files for content that does not belong in a public repository.

    python -m nexttransformer.checks.public            # every tracked file
    python -m nexttransformer.checks.public PATH ...   # only these files

Flags credentials and tokens, private keys, absolute paths into a home or data directory, personal
e-mail addresses, private-network IP addresses, and GPU model names in results, evidence,
documentation, launch scripts, configurations and the website (run records describe the protocol, not the machines; arena cards are
exempt because the arena rules ask entrants to disclose their compute). Exits non-zero on a finding.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SECRETS = re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,}|hf_[A-Za-z0-9]{34,}|sk-[A-Za-z0-9]{32,}"
                     r"|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,})\b|-----BEGIN [A-Z ]*PRIVATE KEY-----")
PATHS = re.compile(r"(/home/(?!runner/)[a-z_][a-z0-9_.-]*/|/Users/[A-Za-z][^/\s]*/|/data/home/|[A-Z]:\\Users\\)")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9-]{2,}(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,6}\b")
EMAIL_OK = re.compile(r"(noreply|no-reply)@|@users\.noreply\.github\.com|@example\.(com|org|net)$|^git@github\.com$")
PRIVATE_IP = re.compile(r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b")
GPU = re.compile(r"(?<![A-Za-z0-9])(?:A800|A100|H800|H100|H200|H20|B200|B300|B30Z|GB200|GB300|L40S|RTX[ _-]?\d{4})(?![A-Za-z0-9])", re.IGNORECASE)
GPU_SCOPE = ("results/", "evidence/", "configs/", "docs/", "launch/", "README", "website/", ".github/assets/")
GPU_EXEMPT = ("results/arena/",)
# These two product names identify the public T3 budget scenarios, never run provenance.
PLANNING_HARDWARE_FILES = {"configs/scaling/t3.json", "website/app/lib/scaling.ts",
                           "website/app/scaling.tsx", "website/tests/scaling.test.mjs"}
SELF = "nexttransformer/checks/public.py"
SKIP = ("uv.lock", "package-lock.json", ".lock.txt", ".woff2", ".png", ".jpg", ".jpeg", ".ico", ".pdf", "-OFL.txt")


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def scan(paths):
    findings = []
    for path in paths:
        if path.endswith(SKIP) or path == SELF or not (REPO/path).is_file():
            continue
        try:
            text = (REPO/path).read_text()
        except UnicodeDecodeError:
            continue
        gpu = path.startswith(GPU_SCOPE) and not path.startswith(GPU_EXEMPT)
        for number, line in enumerate(text.splitlines(), 1):
            hits = ["possible credential (redacted)"] if SECRETS.search(line) else []
            hits += [m.group(0) for m in PATHS.finditer(line)]
            hits += [m.group(0) for m in EMAIL.finditer(line) if not EMAIL_OK.search(m.group(0))]
            hits += [m.group(0) for m in PRIVATE_IP.finditer(line)]
            if gpu:
                hits += [m.group(0) for m in GPU.finditer(line)
                         if path not in PLANNING_HARDWARE_FILES or m.group(0) not in {"A100", "H200"}]
            findings += [f"{path}:{number}: {hit}" for hit in hits]
    return findings


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    findings = scan(argv or tracked())
    print("\n".join(findings) if findings else "no findings")
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
