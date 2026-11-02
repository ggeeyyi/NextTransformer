"""Admin CLI (not exposed to the agent).

    python -m nextt_gym.cli import                     # upstream methods and formal results into the archive
    python -m nextt_gym.cli budget [--set-cap FLOPS]   # show or set the total FLOPs cap
    python -m nextt_gym.cli backends                   # configured clusters (backends.local.toml)
    python -m nextt_gym.cli anchor --scale S --backend ID   # run the Dense anchor at a scale
    python -m nextt_gym.cli monitor [--every 60]       # poll running jobs, charge the ledger, settle
    python -m nextt_gym.cli reconcile                  # compare the ledger with each run's budget.json
    python -m nextt_gym.cli leaderboard --scale L [--split official]   # unsealed: the operator sees all results
    python -m nextt_gym.cli reveal ATTEMPT_ID --reason TEXT  # show an attempt's sealed official BPB and CORE to agents
Set NEXTT_GYM_HOME for the state directory (default ~/.nextt-gym).
"""
import argparse
import json
from pathlib import Path
import time

from .service import REPO_ROOT, GymService


def _print(value):
    print(json.dumps(value, indent=1, default=str))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--home"); parser.add_argument("--backends")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("import")
    budget = sub.add_parser("budget"); budget.add_argument("--set-cap", type=float)
    sub.add_parser("backends")
    anchor = sub.add_parser("anchor"); anchor.add_argument("--scale", required=True)
    anchor.add_argument("--backend", required=True); anchor.add_argument("--world-size", type=int)
    monitor = sub.add_parser("monitor"); monitor.add_argument("--every", type=int, default=60)
    monitor.add_argument("--once", action="store_true")
    sub.add_parser("reconcile")
    board = sub.add_parser("leaderboard"); board.add_argument("--scale", required=True)
    board.add_argument("--split", default="dev")
    reveal = sub.add_parser("reveal"); reveal.add_argument("attempt_id"); reveal.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    service = GymService(home=args.home, backends_path=args.backends)

    if args.command == "import":
        from .importer import import_upstream
        _print(import_upstream(service.archive, REPO_ROOT))
    elif args.command == "budget":
        if args.set_cap: service.ledger.set_cap(int(args.set_cap))
        _print(service.get_budget())
    elif args.command == "backends":
        _print(service.list_backends())
    elif args.command == "anchor":
        # The upstream SSSL Dense, tagged as this scale's anchor; it is charged like any run (kind anchor).
        dense = next(a for a in service.archive.list_attempts(origin="imported", limit=100)["items"]
                     if a["title"] == "dense")
        quote = service.quote_run(dense["attempt_id"], args.scale, args.backend, args.world_size)
        if not quote["feasible"]: _print(quote); raise SystemExit(1)
        run = service.launch_run(quote["quote_id"], f"anchor-{args.scale}-{args.backend}-{time.time():.0f}")
        service.archive.add_note(dense["attempt_id"], f"anchor run {run['run_id']} at scale {args.scale}", ["anchor"])
        _print(run)
    elif args.command == "monitor":
        while True:
            service.poll(force=True)
            active = [r for r in service.archive.list_runs() if r["status"] not in
                      ("succeeded", "failed", "cancelled", "budget_exceeded") and r["origin"] == "agent"]
            print(time.strftime("%H:%M:%S"), f"{len(active)} active run(s)",
                  [(r["run_id"], r["status"], r["progress"].get("step")) for r in active], flush=True)
            if args.once: break
            time.sleep(args.every)
    elif args.command == "reconcile":
        rows = []
        for run in service.archive.list_runs(origin="agent"):
            path = Path(service.home) / "runs" / run["run_id"] / "fetched" / "budget.json"
            file_flops = int(json.loads(path.read_text()).get("consumed_flops") or 0) if path.exists() else None
            rows.append(dict(run_id=run["run_id"], status=run["status"], ledger=run["consumed"]["flops_exact"],
                             budget_json=str(file_flops), match=file_flops is None or str(file_flops) == run["consumed"]["flops_exact"]))
        _print(dict(runs=rows, all_match=all(r["match"] for r in rows)))
    elif args.command == "leaderboard":
        _print(service.archive.leaderboard(args.scale, split=args.split, limit=100))
    elif args.command == "reveal":
        service.archive.reveal(args.attempt_id, args.reason)
        _print(service.archive.get_attempt(args.attempt_id, include=("runs",)))


if __name__ == "__main__":
    main()
