"""Run bundles: the committed repository plus one submission, as one hashed archive.

Every cluster runs byte-identical code. The repository part comes from
`git archive HEAD`, so `nextt_gym/` must be committed; the submission files are
added under `submission/`.
"""
import gzip
import hashlib
import io
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from .contract import OPTIONAL_FILES, REQUIRED_FILES, files_sha256


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def make_bundle(repo_root, submission_dir, dest_dir):
    """Write `bundle-<commit>-<submission>.tar.gz` into dest_dir; returns (path, sha256)."""
    repo, submission = Path(repo_root), Path(submission_dir)
    if _git(repo, "status", "--porcelain", "--", "nextt_gym").strip():
        raise RuntimeError("nextt_gym/ has uncommitted changes; commit them so the bundled code is reproducible")
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    files = [n for n in REQUIRED_FILES + OPTIONAL_FILES if (submission / n).exists()]
    missing = [n for n in REQUIRED_FILES if n not in files]
    if missing: raise FileNotFoundError(f"submission is missing {missing}")
    sub_hash = files_sha256(submission)
    dest = Path(dest_dir); dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"bundle-{commit[:12]}-{sub_hash[:12]}.tar.gz"
    source = io.BytesIO(_git(repo, "archive", "--format=tar", commit))
    tmp = path.with_suffix(".tmp")
    # mtime=0 keeps the gzip header, hence the hash, stable for identical content.
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as out, tarfile.open(fileobj=source, mode="r") as repo_tar:
        for member in repo_tar.getmembers():
            out.addfile(member, repo_tar.extractfile(member) if member.isfile() else None)
        for name in files:
            data = (submission / name).read_bytes()
            info = tarfile.TarInfo(f"submission/{name}"); info.size = len(data); info.mode = 0o644
            out.addfile(info, io.BytesIO(data))
        meta = f'{{"commit": "{commit}", "submission_sha256": "{sub_hash}"}}\n'.encode()
        info = tarfile.TarInfo("BUNDLE.json"); info.size = len(meta); info.mode = 0o644
        out.addfile(info, io.BytesIO(meta))
    tmp.replace(path)
    return path, _sha256(path)


def unpack_bundle(path, sha256, dest):
    """Verify the hash and extract into dest (refusing absolute or parent-relative members)."""
    if _sha256(path) != sha256: raise ValueError("bundle checksum mismatch")
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk() and \
                    PurePosixPath(member.linkname).is_absolute():
                raise ValueError(f"unsafe bundle member {member.name}")
        tar.extractall(dest)
    (dest / ".bundle_sha256").write_text(sha256 + "\n")
    return dest
