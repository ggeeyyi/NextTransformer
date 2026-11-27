"""Draft an arena disclosure card (results/arena/<entry>.json) and its metering record from the gym's archive.

The gym fills what it measured over the research window: the period, the distinct designs it trained, the selection
signal its policy allows (training loss and the dev split; official results stay sealed), the GPU-hours, the hardware
and the commit the first run's code bundle was cut from. The operator supplies what the gym cannot know (people, LLMs
and their roles, framework, involvement, internet use, novelty, artifacts) in a JSON file that uses the card's field
names; its values are copied unchanged and override the gym's. The draft is checked with
`nexttransformer.arena.check_card`, the check CI runs on every card.
"""
from datetime import date, datetime
import json
from pathlib import Path
import tarfile

from .errors import GymError
from .task import SCALES

SIGNALS = ["dev-bpb"]     # the only score the gym shows an agent; training loss is visible too but ranked nothing


def _day(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).date()


def research_runs(service, since=None, until=None):
    """Runs the gym launched (not imported upstream records) and created within [since, until], dates inclusive."""
    since, until = (date.fromisoformat(d) if isinstance(d, str) else d for d in (since, until))
    runs = []
    for run in service.archive.list_runs():
        if run["origin"] == "imported": continue
        day = _day(run["created_at"])
        if (since and day < since) or (until and day > until): continue
        runs.append(run)
    return sorted(runs, key=lambda r: r["created_at"])


def bundle_commit(service, run):
    """The commit a run's code bundle was cut from (BUNDLE.json inside the bundle), or None."""
    for path in sorted((Path(service.home) / "runs" / run["run_id"]).glob("bundle-*.tar.gz")):
        with tarfile.open(path) as tar:
            return json.loads(tar.extractfile("BUNDLE.json").read())["commit"]
    return None


def metering_rows(service, runs):
    """One row per run: what it trained and the compute it was charged."""
    rows = []
    for run in runs:
        attempt = service.archive.get_attempt(run["attempt_id"], include=())
        spec = Path(service.home) / "runs" / run["run_id"] / "run_spec.json"
        rows.append(dict(run_id=run["run_id"], attempt_id=run["attempt_id"], title=attempt["title"],
                         synthetic=json.loads(spec.read_text()).get("synthetic") if spec.exists() else None,
                         origin=attempt["origin"], scale=run["scale"], status=run["status"],
                         steps=(run["progress"] or {}).get("step") or 0, backend_id=run["backend_id"],
                         gpu_type=run["gpu_type"], world_size=run["world_size"],
                         flops=run["consumed"]["flops_exact"], trainer_gpu_hours=run["consumed"]["gpu_hours"],
                         platform_gpu_hours=run["consumed"]["platform_gpu_hours"],
                         wall_hours=run["consumed"]["wall_hours"], created_at=run["created_at"],
                         ended_at=run["ended_at"], commit=bundle_commit(service, run)))
    return rows


def draft(service, attempt_id, *, method, entry, operator, since=None, until=None):
    """(card, metering) for the entry `entry` of `method`, found as gym attempt `attempt_id`."""
    from nexttransformer.arena import VERSION, check_card
    service.archive.get_attempt(attempt_id, include=())         # NOT_FOUND for an unknown attempt
    runs = research_runs(service, since, until)
    if not any(r["attempt_id"] == attempt_id for r in runs):
        raise GymError("INVALID_ARGUMENT", f"no run of {attempt_id} in the research window")
    rows = metering_rows(service, runs)
    trained = [r for r in rows if r["steps"] > 0]
    # Candidates are designs trained beyond smoke tests: agent attempts with a run on real data (not synthetic rows).
    designs = sorted({r["attempt_id"] for r in trained if r["origin"] == "agent" and r["synthetic"] is False})
    platform = [r for r in rows if r["platform_gpu_hours"] is not None]
    unmetered = [r for r in rows if r["platform_gpu_hours"] is None]
    measured = sum(r["platform_gpu_hours"] for r in platform)
    hours = None if unmetered else measured          # a total with unknown parts would understate the research
    shapes = sorted({(r["world_size"], r["gpu_type"]) for r in trained if r["world_size"]}, key=str)
    llm = service.ledger.llm_usage()
    note = (f"Metered by NextT-Gym over {len(rows)} runs ({len(trained)} trained at least one update"
            + (f", {sum(r['synthetic'] is True for r in rows)} of them on synthetic rows" if any(r["synthetic"] for r in rows)
               else "") + "): platform GPU-hours, pod time x GPUs as the cluster recorded it (queueing excluded)"
            + (f"; {round(measured, 2)} GPU-hours for the {len(platform)} runs with a platform record, and no total "
               f"because {len(unmetered)} runs have none (their trainers recorded "
               f"{round(sum(r['trainer_gpu_hours'] or 0 for r in unmetered), 2)} GPU-hours of training and evaluation)"
               if unmetered else "")
            + ". Scales: " + ", ".join(f"{s} x{sum(r['scale'] == s for r in trained)}" for s in SCALES
                                       if any(r["scale"] == s for r in trained))
            + ". LLM tokens and API cost: " + (f"{llm['tokens']} tokens, ${llm['cost_usd']}"
                                                + (f" ({llm['note']})" if llm.get("note") else "")
                                                if llm.get("tokens") is not None or llm.get("cost_usd") is not None
                                                else "not metered") + ".")
    card = dict(
        id=entry, name=method, methodId=method, runIds=[], arenaProtocol=VERSION, status="admitted", statusNote="",
        submitted=date.today().isoformat(), verification="self-reported",
        researcher=dict(people=[], llms=[], framework=dict(name="NextT-Gym", version=None, url=None, note=None),
                        involvement=None, involvementNote=""),
        research=dict(start=_day(rows[0]["created_at"]).isoformat(),
                      end=_day(max(r["ended_at"] or r["created_at"] for r in rows)).isoformat(),
                      baseCommit=rows[0]["commit"], candidatesTrained=len(designs), formalAttempts=0,
                      formalNote="No formal run yet.", gpuHours=None if hours is None else round(hours, 2),
                      hardware="; ".join(f"{w} x {g} per run" for w, g in shapes) or None, computeNote=note,
                      internet=False, selectionSignal=list(SIGNALS)),
        novelty=dict(closestPrior=[], statement=""),
        artifacts=dict(checkpointUrl=None, logsUrl=None, traceUrl=None, historyUrl=None, codeUrl=None))
    for key, value in operator.items():
        if isinstance(value, dict) and isinstance(card.get(key), dict): card[key] = {**card[key], **value}
        else: card[key] = value
    metering = dict(entry=entry, method=method, attempt_id=attempt_id, window=dict(since=str(since), until=str(until)),
                    designs_trained=designs, gpu_hours=None if hours is None else round(hours, 4),
                    measured_gpu_hours=round(measured, 4), unmetered_runs=[r["run_id"] for r in unmetered], llm=llm,
                    runs=rows)
    return card, metering, check_card(card, f"results/arena/{entry}.json")
