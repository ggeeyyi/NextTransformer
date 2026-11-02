"""Read-only H03 metadata audit; run from the repository root with Python 3.10+.

python evidence/checks/h03-b300-20261031/audit.py RUN_DIR --output REPORT.json
Checks the frozen training commit and cost proof; never loads tensors or changes the run directory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[3]
COMMIT = "fab8c032e9ebc91cc14933d8e7e0275f6c60224f"
PROOF = "evidence/costs/hybrid-kernels-20261026-release/gdn_hybrid_swa.json"
PACKAGE_FINGERPRINTS = dict(
    science_hash="e1b5a94ce3233d9cc34a36b281c90d4099687f8816c9fc3ed157f32813656f78",
    reference_source_hash="9a3aecef522791bacfc1317bf447fed7bf109e1b9748f52d8cd3e7618b0cadf1",
    native_source_sha256="2822cb470984fe3a4874191837d37e182535a9ac3236b25fb6860d9c5974cec1",
)
STEPS, BATCH, VAL_TOKENS, FINAL_TOKENS = 46875, 1048576, 41943040, 20971520
TOKENS = STEPS * BATCH
VAL_PLAN = list(range(0, STEPS, 250)) + [STEPS]
CORE_PLAN = [8000, 16000, 24000, 32000, 40000, STEPS]
TOKENIZER = "06258dc0044b6198c6fa8ba75c5b3bbaeaeab44f4804b464d9ce5f5e3797ec54"


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def source_fingerprints():
    paths = git("ls-tree", "-r", "--name-only", COMMIT).decode().splitlines()
    blobs = {p: git("show", f"{COMMIT}:{p}") for p in paths
             if p.startswith(("nexttransformer/", "nanochat/", "configs/", "launch/", "requirements/"))
             or p in ("pyproject.toml", "uv.lock")}
    science = ["nexttransformer/" + p for p in
               ("protocol/spec.py", "training/optimizer.py", "protocol/compute.py", "protocol/calibrate.py")]
    science += [p for p in blobs if p.startswith("nexttransformer/models/") and p.endswith(".py")]
    science += ["nanochat/" + p for p in ("common.py", "flash_attention.py", "optim.py")]
    reference = [p for p in blobs if p.startswith(("nexttransformer/", "nanochat/")) and p.endswith(".py")
                 and not {"results", "tests", "__pycache__", ".agent-work"}.intersection(Path(p).parts)]
    native = {p: digest(b) for p, b in blobs.items()
              if (Path(p).suffix in (".py", ".sh", ".json", ".txt")
                  and "__pycache__" not in Path(p).parts and Path(p).name != "environment-daxiao.lock.txt"
                  and not p.endswith((".local.json", ".local.toml"))) or p in ("pyproject.toml", "uv.lock")}
    def combined(names):
        h = hashlib.sha256()
        for p in sorted(names):
            h.update(p.encode())
            h.update(blobs[p])
        return h.hexdigest()
    return dict(science_hash=combined(science), reference_source_hash=combined(reference),
                native_source_sha256=digest(json.dumps(native, sort_keys=True).encode()))


def audit(run):
    p, status, budget = [read(run / n) for n in ("provenance.json", "run_status.json", "budget.json")]
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    train, vals, cores, finals = [[e for e in events if e["event"] == kind]
                                 for kind in ("train", "validation", "core", "final_bpb")]
    cfg = json.loads(git("show", f"{COMMIT}:configs/hybrid/gdn_hybrid_swa.json"))
    proof_bytes = git("show", f"{COMMIT}:{PROOF}")
    proof = json.loads(proof_bytes)
    tasks = {t["id"]: t["randomBaseline"] for t in read(ROOT / "website/data/core-tasks.json")["tasks"]}
    fingerprints = source_fingerprints()
    checks = {}
    def need(name, value):
        checks[name] = bool(value)
    finite = lambda x: isinstance(x, (int, float)) and math.isfinite(x)
    need("recorded_source_fingerprints", all(p["software"][k] == v for k, v in PACKAGE_FINGERPRINTS.items())
         and all(len(v) == 64 for v in fingerprints.values()))
    need("original_cost_proof_preserved", (ROOT / PROOF).read_bytes() == proof_bytes)
    need("protocol_and_tokenizer", p["protocol"] == cfg["protocol"]
         and p["protocol_hash"] == cfg["protocol_hash"] and p["tokenizer_sha256"] == TOKENIZER)
    need("model_and_cost_identity", p["model_hash"] == proof["model_hash"] == cfg["model_hash"]
         and p["parameters"] == proof["parameters"] and p["cost"]["total_training_flops"] == proof["total_training_flops"]
         and p["cost"]["flops_per_token"] == proof["flops_per_token"]
         and p["software"]["torch"] == proof["torch"] and not proof["unknown"])
    need("registered_layout", cfg["model"]["layer_mixers"] == ["gdn", "swa"] * 11
         and cfg["model"]["sliding_window"] == 2048
         and all(p["model"][k] == cfg["model"][k] for k in cfg["model"] if k != "layer_mixers"))
    need("eight_rank_global_batch", p["world_size"] == 8 and p["micro_batch"] == 8 and p["accumulation"] == 8
         and p["world_size"] * p["micro_batch"] * p["accumulation"] * p["model"]["sequence_len"] == BATCH
         and p["compute_dtype"] == "torch.bfloat16" and not p["smoke"])
    need("training_settings", all(p["train"][k] == p["protocol"][k] for k in p["train"] if k in p["protocol"])
         and p["train"]["core_max_per_task"] == -1 and p["resumed_from"] is None
         and not list(run.glob("resume_*.json")))
    need("completed_status", status["phase"] == "training_complete" and status["final_status"] == "matched"
         and status["step"] == STEPS and status["latest"] == read(run / "latest.json"))
    ledger = status["budget"]
    need("budget_agreement", all(budget[k] == v for k, v in ledger.items())
         and ledger["optimizer_step"] == STEPS and ledger["input_tokens"] == TOKENS
         and ledger["val_steps"] == VAL_PLAN and ledger["core_steps"] == CORE_PLAN
         and ledger["theory_flops"] == budget["charged_theory_flops"] == proof["total_training_flops"])
    per_update = proof["stages"][0]["cost"]["total"]
    need("single_joint_stage", len(proof["stages"]) == 1 and proof["stages"][0]["stage"] == "joint"
         and proof["stages"][0]["optimizer_steps"] == STEPS and per_update * STEPS == ledger["theory_flops"])
    relative = ledger["theory_flops"] / (TOKENS * p["cost"]["dense_flops_per_token"]) - 1
    need("cost_tolerances", abs(p["cost"]["relative_error"]) <= .01 and abs(relative) <= .02)
    need("logged_updates", [e["step"] for e in train] == [1] + list(range(10, STEPS, 10)) + [STEPS]
         and all(e["input_tokens"] == e["step"] * BATCH and e["theory_flops"] == e["step"] * per_update
                 and finite(e["loss"]) and finite(e["grad_norm"]) and e["aux_loss"] == 0 for e in train))
    need("189_validation_events", [e["step"] for e in vals] == VAL_PLAN
         and len(list(run.glob("val_*.json"))) == len(VAL_PLAN)
         and all(e == read(run / f"val_{e['step']:08d}.json") and finite(e["val_bpb"])
                 and e["input_tokens"] == VAL_TOKENS and e["split"] == "val" and not e["smoke"]
                 and e["protocol_hash"] == p["protocol_hash"] for e in vals))
    need("six_full_core_events", [e["step"] for e in cores] == CORE_PLAN
         and all(e["full_core"] and e["expected_tasks"] == 22 and set(e["results"]) == set(tasks)
                 and set(e["centered_results"]) == set(tasks) for e in cores))
    core_errors = []
    for e in cores:
        stored = read(run / ("core_final.json" if e["step"] == STEPS else f"core_{e['step']:08d}.json"))
        need(f"core_file_{e['step']}", all(e[k] == v for k, v in stored.items()))
        core_errors += [abs((e["results"][t] - b) / (1 - b) - e["centered_results"][t]) for t, b in tasks.items()]
        core_errors.append(abs(sum(e["centered_results"].values()) / len(tasks) - e["core_metric"]))
    need("core_scores_and_aggregates", all(finite(e) and e <= 1e-7 for e in core_errors))
    final = read(run / "bpb_final.json")
    need("independent_final_bpb", len(finals) == 1 and finals[0]["step"] == STEPS
         and all(final[s]["input_tokens"] == FINAL_TOKENS and not final[s]["smoke"]
                 and finite(final[s]["bpb"]) and final[s]["bpb"] == finals[0][f"{s}_bpb"] for s in ("train", "val")))
    complete = read(run / "checkpoint_00046875/complete.json")
    need("checkpoint_completion_marker", complete["step"] == STEPS and complete["world_size"] == 8
         and set(complete["files"]) == {"model.pt", *(f"rank_{rank:04d}.pt" for rank in range(8))}
         and all(f["bytes"] > 0 and len(f["sha256"]) == 64 for f in complete["files"].values()))
    steady = [e for e in train if e["step"] > 10]
    comparisons = {}
    for name in ("dense_full", "gdn", "gdn_hybrid"):
        base = read(ROOT / f"results/runs/{name}.json")
        comparisons[name] = dict(independent_final_val_bpb=base["finalBpb"], final_core=base["core"][-1]["value"],
                                 delta_final_val_bpb=final["val"]["bpb"] - base["finalBpb"],
                                 delta_core_points=100 * (cores[-1]["core_metric"] - base["core"][-1]["value"]))
    return dict(kind="h03_frozen_run_metadata_audit", date="2026-10-31", run_id=run.name, method="gdn_hybrid_swa",
                scope="Read-only audit of supplied run metadata against the frozen training configuration and cost proof; not hardware replay or prerequisite acceptance.",
                ok=all(checks.values()), checks=checks, problems=[k for k, v in checks.items() if not v],
                final_status=status["final_status"], acceptance_status="pending_prerequisite_evidence_and_maintainer_review",
                optimizer_steps=STEPS, input_tokens=TOKENS, theory_flops=ledger["theory_flops"],
                average_gflops_per_token=ledger["theory_flops"] / TOKENS / 1e9,
                cumulative_relative_error=relative, deviation_from_registered_proof=ledger["theory_flops"] / proof["total_training_flops"] - 1,
                budget_anomalies=sum(e["event"] == "budget_anomaly" for e in events),
                validation_events=len(vals), core_events=len(cores), train_events=len(train), max_core_arithmetic_error=max(core_errors),
                final_bpb={s: final[s]["bpb"] for s in ("train", "val")}, final_bpb_tokens_per_split=FINAL_TOKENS,
                periodic_val_bpb=vals[-1]["val_bpb"], core_final=cores[-1]["core_metric"], comparisons=comparisons,
                parameters=p["parameters"], hardware=dict(owner_label="8 x B300 (1 node)", reported_device=p["device"],
                world_size=p["world_size"], micro_batch=p["micro_batch"], accumulation=p["accumulation"],
                native_attention=p["software"]["native_attention"], fla_core=p["software"]["fla_core"]),
                resources=dict(training_seconds=status["training_seconds"], evaluation_seconds=status["evaluation_seconds"],
                training_gpu_hours=train[-1]["gpu_hours"], median_step_seconds=statistics.median(e["step_seconds"] for e in steady),
                median_tokens_per_second=statistics.median(e["tokens_per_second"] for e in steady),
                peak_logged_rank0_allocated_bytes=max(e["peak_memory_bytes"] for e in train),
                evaluation_forward_flops=status["evaluation_forward_flops"]),
                owner_reported_timing=dict(start="2026-10-30T01:58:03+08:00", end="2026-10-31T06:40:30+08:00",
                source="Owner supplied; interpreted in the chat's UTC+8 timezone, not inferred from the run id or copied file timestamps."),
                provenance=dict(source_commit=COMMIT, original_git_commit=p["software"]["git_commit"],
                source_verification="Model hash, protocol hash, parameter count and stage/operator cost proof match source_commit (fab8c032e9ebc91cc14933d8e7e0275f6c60224f); recorded software source hashes reflect the deployed B300 fix package (h03-b300-fix-20261028T1845) whose source diff against source_commit was not archived in the metadata-only bundle.",
                cost_proof=PROOF, cost_proof_sha256=digest(proof_bytes),
                frozen_commit_fingerprints=fingerprints, **PACKAGE_FINGERPRINTS,
                model_hash=p["model_hash"], protocol_hash=p["protocol_hash"], data_manifest_sha256=p["data_manifest_sha256"],
                tokenizer_sha256=p["tokenizer_sha256"], software={k: p["software"][k] for k in ("python", "torch", "cuda", "nccl", "fla_core", "fla_record_sha256")},
                segments=1, resume_records=[]), checkpoint_marker=complete,
                limits=["Checkpoint tensor files were not supplied, loaded or rehashed; only their completion marker was checked.",
                        "Data/tokenizer/evaluation identities are recorded; original resource bytes were not supplied for independent rehashing.",
                        "Recorded software source hashes (science_hash e1b5a94ce..., reference_source_hash 9a3aecef..., native_source_sha256 2822cb47...) come from the deployed B300 fix package on top of fab8c032e9ebc91cc14933d8e7e0275f6c60224f; model_hash, protocol_hash, parameters and every cost figure match fab8c03's proof, while the cluster source patch itself was not included in the metadata-only archive.",
                        "At the 2048-token context, the registered sliding_window=2048 (left window 2047 plus current token) spans the full sequence, so SWA layers compute full causal attention and this run evaluates a 1:1 GDN/global-attention alternation at 2K context.",
                        "No H03 512-update pilot or target-B300 distributed/fresh-process resume evidence was supplied or located. Earlier H100 timing at schedule offset 512 is not a pilot.",
                        "No earlier failed-attempt inventory was supplied beyond the 'fix' label in the run id; a single uninterrupted submitted run does not establish that no other attempts existed.",
                        "Logged memory is sampled rank-0 allocated memory, not an all-rank reserved-memory maximum; MFU was unreported.",
                        "Formal acceptance remains a maintainer decision; trainer-reported matched and metadata audit success do not fill missing prerequisites."],
                input_sha256={f.relative_to(run).as_posix(): digest(f.read_bytes()) for f in sorted(run.rglob("*")) if f.is_file()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = audit(args.run_dir)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("ok", "problems", "final_bpb", "core_final", "resources", "comparisons")}, indent=2))
    raise SystemExit(0 if report["ok"] else 1)
