"""Download and verify the pinned ClimbMix parquet shards; resumable and idempotent.

Every shard is checked against the SHA-256 in the frozen source manifest (the
same file whose hash is `source_manifest_sha256` in the A800 native manifest),
not only by size. The manifest is copied, never regenerated, so the rebuilt
native manifest stays byte-identical. Run from the repository root:

    uv run --no-project --with huggingface_hub==1.26.0 \
        python launch/koala/fetch_data.py --output DIR --manifest FILE \
        --expect-manifest-sha256 HEX
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nexttransformer.runtime.files import DATA_REPO, DATA_REVISION, atomic_json, sha256


def fetch(name, size, digest, output, attempts):
    from huggingface_hub import hf_hub_download
    target = output/name
    for attempt in range(1, attempts+1):
        try:
            if not (target.exists() and target.stat().st_size == size):
                hf_hub_download(DATA_REPO, name, repo_type='dataset', revision=DATA_REVISION,
                                local_dir=output)
            if target.stat().st_size != size:
                raise ValueError(f'size {target.stat().st_size} != {size}')
            actual = sha256(target)
            if actual != digest:
                # A corrupt file must not survive as a "complete" shard.
                target.unlink()
                raise ValueError(f'sha256 {actual} != {digest}')
            return name, digest
        except Exception as exc:
            if attempt == attempts:
                raise RuntimeError(f'{name}: {exc}') from exc
            # Anonymous Hub requests are rate limited per 5-minute window.
            limited = '429' in str(exc) or 'Rate limit' in str(exc)
            time.sleep(min(300, 60*attempt) if limited else min(60, 5*attempt))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True, help='frozen A800 source manifest JSON')
    parser.add_argument('--expect-manifest-sha256', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--attempts', type=int, default=12)
    parser.add_argument('--limit', type=int, help='only the first N shards (connectivity test)')
    args = parser.parse_args()

    if sha256(args.manifest) != args.expect_manifest_sha256:
        raise SystemExit('source manifest hash differs from the pinned A800 manifest')
    record = json.loads(args.manifest.read_text())
    if record['repo'] != DATA_REPO or record['revision'] != DATA_REVISION:
        raise SystemExit('source manifest repository/revision mismatch')
    shards = sorted((f['rfilename'], f['size'], f['lfs']['sha256'])
                    for f in record['files'] if f['rfilename'].endswith('.parquet'))
    if args.limit: shards = shards[:args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output/'.downloads'/DATA_REVISION[:12]/'manifest.json'
    if frozen.exists():
        if sha256(frozen) != args.expect_manifest_sha256:
            raise SystemExit(f'existing {frozen} differs from the pinned manifest')
    else:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(args.manifest.read_bytes())
    ledger_path = frozen.parent/'verified.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    pending = [s for s in shards
               if ledger.get(s[0]) != s[2] or not (args.output/s[0]).exists()
               or (args.output/s[0]).stat().st_size != s[1]]
    print(f'{len(shards)} shards, {len(shards)-len(pending)} already verified, {len(pending)} to fetch', flush=True)

    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    failures, started, done = [], time.monotonic(), 0
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, *s, args.output, args.attempts) for s in pending]
        for future in concurrent.futures.as_completed(futures):
            try:
                name, digest = future.result()
                ledger[name] = digest; done += 1
            except Exception as exc:
                failures.append(str(exc)); print(f'FAILED {exc}', flush=True)
            if done % 50 == 0 or done == len(pending):
                atomic_json(ledger_path, ledger)
                print(f'{done}/{len(pending)} verified, {time.monotonic()-started:.0f}s', flush=True)
    atomic_json(ledger_path, ledger)

    inventory = sorted((p.name, p.stat().st_size) for p in args.output.glob('*.parquet'))
    expected = [(n, s) for n, s, _ in shards]
    summary = {'repo': DATA_REPO, 'revision': DATA_REVISION, 'manifest_sha256': args.expect_manifest_sha256,
               'expected_shards': len(shards), 'verified_shards': sum(ledger.get(n) == d for n, _, d in shards),
               'inventory_matches': inventory == expected, 'failures': failures}
    print(json.dumps(summary, indent=2), flush=True)
    if failures or not summary['inventory_matches'] or summary['verified_shards'] != len(shards):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
