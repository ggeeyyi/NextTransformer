"""Read result files of a run directory on another machine through a shell command.

Backends whose runs write to storage that the gym host does not mount (Koala's CPFS, a cluster's
scratch disk) fetch results with a `reader` command template. `{script}` is replaced by a quoted
POSIX shell script that prints the requested files as one base64 tar between two marker lines, so
any transport that runs a shell next to the files works:

  reader = "bash -c {script}"                                        # the directory is mounted here
  reader = "ssh -o BatchMode=yes HOST sh -c {script}"                # a bare-metal node
  reader = "koala --cluster h20 exec --cpu -q -- bash -c {script}"   # a Koala CPU debug pod

Other literal braces in a template must be doubled (`str.format`). Only regular files are
extracted, flattened to their base names as `LocalBackend.fetch` does, and replaced atomically.
"""
import base64
import io
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile

BEGIN, END = "NEXTT_FILES_BEGIN", "NEXTT_FILES_END"
_PATTERN = re.compile(r"[A-Za-z0-9_.*/-]+")


def read_script(directory, patterns):
    """Shell script printing the files that match `patterns` (globs relative to `directory`) as a base64 tar."""
    for pattern in patterns:
        if not _PATTERN.fullmatch(pattern) or ".." in pattern.split("/"):
            raise ValueError(f"unsafe file pattern {pattern!r}")
    # Patterns stay unquoted so that the remote shell expands the globs; unmatched globs fail `-f`.
    return (f"cd {shlex.quote(str(directory))} 2>/dev/null || {{ echo {BEGIN}; echo {END}; exit 0; }}\n"
            "set --\n"
            f"for f in {' '.join(patterns)}; do [ -f \"$f\" ] && set -- \"$@\" \"$f\"; done\n"
            f"echo {BEGIN}\n"
            "if [ $# -gt 0 ]; then tar czf - -- \"$@\" | base64 | tr -d '\\n'; echo; fi\n"
            f"echo {END}\n")


def fetch(template, directory, patterns, dest, *, timeout=300):
    """Copy the matching files of `directory` into `dest`; returns their names. Raises if the reader fails."""
    command = template.format(script=shlex.quote(read_script(directory, patterns)))
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
    match = re.search(rf"^{BEGIN}[ \t\r]*$(.*?)^{END}[ \t\r]*$", result.stdout, re.S | re.M)
    if result.returncode or not match:
        raise RuntimeError(f"reader exited {result.returncode}: {(result.stderr or result.stdout)[-500:]}")
    payload = "".join(match.group(1).split())
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    names = []
    if payload:
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile(): continue
                name = PurePosixPath(member.name).name
                tmp = dest / f".{name}.tmp"
                tmp.write_bytes(tar.extractfile(member).read()); tmp.replace(dest / name)
                names.append(name)
    return names
