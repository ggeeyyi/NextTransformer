"""Admin CLI (not exposed to the agent).

    python -m nextt_gym.cli import                     # upstream methods and formal results into the archive
    python -m nextt_gym.cli budget [--set-cap FLOPS]   # show or set the total FLOPs cap
    python -m nextt_gym.cli backends                   # configured clusters (backends.local.toml)
    python -m nextt_gym.cli anchor --scale S --backend ID   # run the Dense anchor at a scale
    python -m nextt_gym.cli monitor [--every 60]       # poll running jobs, charge the ledger, settle
    python -m nextt_gym.cli reconcile                  # compare the ledger with each run's budget.json
    python -m nextt_gym.cli leaderboard --scale L [--split official]   # unsealed: the operator sees all results
    python -m nextt_gym.cli reveal ATTEMPT_ID --reason TEXT  # show an attempt's sealed official BPB and CORE to agents
    python -m nextt_gym.cli meter [RUN_ID ...] [--record FILE]   # platform GPU-hours of Koala runs (cluster records)
    python -m nextt_gym.cli llm-usage --tokens N --cost-usd X [--note TEXT]   # record the research's LLM use
    python -m nextt_gym.cli disclose ATTEMPT_ID --method ID --entry ID --operator FILE --out CARD.json \
        [--metering METERING.json] [--since DATE] [--until DATE]   # draft the arena disclosure card
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
    metered = sub.add_parser("meter"); metered.add_argument("run_ids", nargs="*")
    metered.add_argument("--record", help="JSON list of {run, gpu_hours} the operator kept, for runs the cluster "
                                          "no longer records")
    llm = sub.add_parser("llm-usage"); llm.add_argument("--tokens", type=int); llm.add_argument("--cost-usd", type=float)
    llm.add_argument("--note")
    disclose = sub.add_parser("disclose"); disclose.add_argument("attempt_id")
    for name in ("--method", "--entry", "--operator", "--out"): disclose.add_argument(name, required=True)
    for name in ("--metering", "--since", "--until"): disclose.add_argument(name)
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
    elif args.command == "meter":
        _print(meter(service, args.run_ids, args.record))
    elif args.command == "llm-usage":
        service.store.set_setting("llm_usage", json.dumps(dict(tokens=args.tokens, cost_usd=args.cost_usd, note=args.note)))
        _print(service.ledger.llm_usage())
    elif args.command == "disclose":
        from .disclosure import draft
        card, metering, problems = draft(service, args.attempt_id, method=args.method, entry=args.entry,
                                         operator=json.loads(Path(args.operator).read_text()),
                                         since=args.since, until=args.until)
        Path(args.out).write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n")
        if args.metering: Path(args.metering).write_text(json.dumps(metering, indent=1) + "\n")
        _print(dict(card=args.out, gpu_hours=card["research"]["gpuHours"], designs=metering["designs_trained"],
                    problems=problems))
        if problems: raise SystemExit(1)


def meter(service, run_ids=(), record=None):
    """Platform GPU-hours of Koala runs, from each segment's job record (`koala get`), into the run and the ledger.

    Segments the chain driver already metered keep their record; older ones are read from the cluster now (see
    chain.metering for jobs it no longer records). A run with any ended segment of unknown hours stays unknown, unless
    `record` (a JSON list of {run, gpu_hours} the operator kept) has it; the row names the source of every figure.
    """
    kept = {r["run"]: float(r["gpu_hours"]) for r in json.loads(Path(record).read_text())} if record else {}
    from .backends.koala import KoalaBackend
    from .backends.koala.chain import FINAL, metering, read_chain
    rows, total = [], 0.
    for run in service.archive.list_runs():
        if run_ids and run["run_id"] not in run_ids: continue
        backend = service.backends.get(run["backend_id"])
        if not isinstance(backend, KoalaBackend) or not run.get("handle"): continue
        plan_path = Path(run["handle"]) / "koala_plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
        chain = read_chain(run["handle"])
        segments = chain.get("segments", [])
        if chain.get("state") not in FINAL:      # a run in progress: the service meters it as it goes
            rows.append(dict(run_id=run["run_id"], status=run["status"], segments=len(segments), platform_gpu_hours=None,
                             sources=["in progress, not metered here"]))
            continue
        saved = lambda job: (Path(run["handle"]) / f"times-{job}.json")
        meters = [s.get("metering") or metering(backend.koala, s["job"], s, plan.get("gpus", 8) * plan.get("nodes", 1),
                                                json.loads(saved(s["job"]).read_text()) if saved(s["job"]).exists()
                                                else None)
                  for s in segments if s.get("job") and s.get("finished")]
        known = [m["platform_gpu_hours"] for m in meters if m["platform_gpu_hours"] is not None]
        sources = sorted({m["source"] for m in meters})
        hours = round(sum(known), 4) if meters and len(known) == len(meters) else None
        if hours is None and run["run_id"] in kept: hours, sources = kept[run["run_id"]], [f"record {Path(record).name}"]
        if hours is not None:
            service.archive.update_run(run["run_id"], platform_gpu_hours=hours)
            service.ledger.meter(run["run_id"], hours)
            total += hours
        rows.append(dict(run_id=run["run_id"], status=run["status"], segments=len(segments), platform_gpu_hours=hours,
                         known_segment_hours=round(sum(known), 4), sources=sources,
                         queue_hours=round(sum(m["queue_seconds"] or 0 for m in meters) / 3600, 3)))
    return dict(runs=rows, platform_gpu_hours=round(total, 4))


if __name__ == "__main__":
    main()
