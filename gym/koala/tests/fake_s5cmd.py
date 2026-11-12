#!/usr/bin/env python3
"""A local stand-in for the s5cmd calls koala_gym.py makes (tests only): s3://BUCKET/KEY is ROOT/BUCKET/KEY.

fake_s5cmd.py ROOT cp SRC DST | ls URI | ls PREFIX/* | rm URI[/*] | cat URI | run FILE
"""

import shlex
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])


def local(uri):
    return ROOT / uri[len("s3://") :] if uri.startswith("s3://") else Path(uri)


def line(path, name):
    print(f"{time.strftime('%Y/%m/%d %H:%M:%S')} {path.stat().st_size:>12} {name}")


def main(args):
    if args[0] == "cp":
        source, target = local(args[1]), local(args[2])
        if not source.is_file():
            print(f"ERROR no object found: {args[1]}", file=sys.stderr)
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return 0
    if args[0] == "ls":
        uri = args[1]
        if uri.endswith("/*"):
            base = local(uri[:-2])
            found = (
                sorted(p for p in base.rglob("*") if p.is_file())
                if base.is_dir()
                else []
            )
            for path in found:
                line(path, str(path.relative_to(base)))
        else:
            found = [local(uri)] if local(uri).is_file() else []
            for path in found:
                line(path, path.name)
        if not found:
            print("ERROR no object found", file=sys.stderr)
            return 1
        return 0
    if args[0] == "rm":
        uri = args[1]
        base = local(uri[:-2]) if uri.endswith("/*") else local(uri)
        if base.is_dir():
            shutil.rmtree(base)
        elif base.is_file():
            base.unlink()
        return 0
    if args[0] == "cat":
        sys.stdout.write(local(args[1]).read_text())
        return 0
    if args[0] == "run":
        for command in Path(args[1]).read_text().splitlines():
            if command.strip() and main(shlex.split(command)):
                return 1
        return 0
    print(f"unsupported: {args}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[2:]))
