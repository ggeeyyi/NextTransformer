"""H200 checks of the object-store mode: real ClimbMix data staged from S3, continuation from an S3 checkpoint.

  gym/.venv/bin/python gym/koala/tests/h200_data_e2e.py --koala .agent-work/koala.h200-ll-s3.json

A research run on ClimbMix (shard_00000, dev shard_06541) to 10 updates, its continuation to 20 from the checkpoint
in S3, and an uninterrupted run to 20: the continuation must equal the uninterrupted run.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_e2e import Session, climbmix_profile, example_candidate, fixture_profiles, log  # noqa: E402

DATA = "/local-ssd/nextt-data/climbmix-400b-shuffle"
RESOURCES = "/local-ssd/nextt-resources/native-d24"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--koala", type=Path, required=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="only run the uninterrupted 0 -> 20 this many times (run-to-run spread on the GPU)",
    )
    args = parser.parse_args()
    base = (
        Path.home()
        / ".local/share/koala-gym/e2e"
        / f"h200-data-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    state = base / "state"
    state.mkdir(parents=True)
    log(f"h200 data e2e; state {state}")
    profiles = fixture_profiles(6 * 3600)
    profile = climbmix_profile(6 * 3600)
    profile["data"] = {
        "kind": "climbmix",
        "tokenizer": f"{RESOURCES}/tokenizer",
        "train_paths": [f"{DATA}/shard_00000.parquet"],
        "dev_path": f"{DATA}/shard_06541.parquet",
        "token_bytes": f"{RESOURCES}/tokenizer/token_bytes.pt",
    }
    profiles["research-climbmix"] = profile
    session = Session(args.koala, profiles, state)
    candidate = session.call(
        "submit_candidate",
        package_base64=base64.b64encode(example_candidate()).decode(),
        idempotency_key="example",
    )
    if args.repeat:
        runs = [
            session.train(
                candidate,
                "research-climbmix",
                f"climbmix 0 -> 20 #{i + 1}",
                target_update=20,
            )
            for i in range(args.repeat)
        ]
        bpb = [r.get("result", {}).get("dev_bpb") for r in runs]
        log("repeat", dev_bpb=bpb, identical=len(set(bpb)) == 1)
        (base / "summary.json").write_text(
            json.dumps({"repeat": runs, "dev_bpb": bpb}, indent=1, default=str)
        )
        return
    steps = [
        session.train(
            candidate, "research-climbmix", "climbmix 0 -> 10", target_update=10
        )
    ]
    if steps[-1]["state"] == "succeeded":
        checkpoint = steps[-1]["result"]["checkpoint_id"]
        steps.append(
            session.train(
                candidate,
                "research-climbmix",
                "climbmix 10 -> 20 (from S3)",
                target_update=20,
                checkpoint_id=checkpoint,
            )
        )
    steps.append(
        session.train(
            candidate, "research-climbmix", "climbmix 0 -> 20", target_update=20
        )
    )
    bpb = [s.get("result", {}).get("dev_bpb") for s in steps]
    pointer = (
        json.loads(
            session.backend.checkpoint_path(
                steps[0]["result"]["checkpoint_id"]
            ).read_text()
        )
        if steps[0]["state"] == "succeeded"
        else None
    )
    summary = {
        "steps": steps,
        "dev_bpb": bpb,
        "pointer": pointer,
        "continuation_matches": len(bpb) == 3
        and bpb[1] is not None
        and bpb[1] == bpb[2],
    }
    log(
        "result",
        dev_bpb=bpb,
        continuation_matches=summary["continuation_matches"],
        checkpoint=pointer and pointer["remote_path"],
    )
    (base / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"summary written to {base / 'summary.json'}")


if __name__ == "__main__":
    main()
