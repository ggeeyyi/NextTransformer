"""A4 sweep: call every agent-visible tool, resource and prompt over one MCP stdio session and search the outputs
for published official values (final BPB, CORE events, last validation points of results/runs/*.json).
Usage (with NEXTT_GYM_HOME set): python a4_sweep.py OUT.json"""
import asyncio, glob, json, os, re, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
values = set()
for path in glob.glob(f"{REPO}/results/runs/*.json"):
    record = json.load(open(path))
    for v in [record.get("finalBpb")] + [p["value"] for p in record.get("core") or []] + \
            [p["value"] for p in record.get("val") or []][-3:]:
        if isinstance(v, float): values |= {f"{v:.4f}", f"{int(v * 1e4) / 1e4:.4f}"}
PATTERN = re.compile(r"(?<!\d)(" + "|".join(re.escape(v) for v in sorted(values)) + ")")


async def main(out):
    params = StdioServerParameters(command=sys.executable, args=["-m", "nextt_gym.mcp_server"], cwd=REPO,
                                   env=dict(os.environ))
    outputs = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def tool(tool_name, **args):
                result = await session.call_tool(tool_name, args)
                text = "".join(getattr(c, "text", "") for c in result.content)
                outputs[f"{tool_name} {json.dumps(args, sort_keys=True)}"] = text
                try: return json.loads(text)
                except json.JSONDecodeError: return text

            async def resource(uri):
                result = await session.read_resource(uri)
                outputs[f"resource {uri}"] = "".join(getattr(c, "text", "") for c in result.contents)

            await tool("get_task"); await tool("get_budget"); await tool("get_instructions"); await tool("get_spec")
            for scale in ("S", "L"):
                for split in ("dev", "official"): await tool("get_leaderboard", scale=scale, split=split, limit=100)
            attempts = (await tool("list_attempts", limit=100))["items"]
            await tool("list_attempts", limit=100, sort="best_L"); await tool("list_attempts", limit=100, sort="best_S")
            runs = []
            for a in attempts:
                detail = await tool("get_attempt", attempt_id=a["attempt_id"], include=["runs", "lineage", "audit"])
                runs += [r["run_id"] for r in detail.get("run_details", [])]
            for run_id in runs:
                run = await tool("get_run", run_id=run_id)
                for metric in ("val_bpb", "train_loss"):
                    for split in ("dev", "official"):
                        await tool("get_run_curve", run_id=run_id, metric=metric, split=split, max_points=500)
                await tool("get_run_logs", run_id=run_id, tail=2000)
            imported = [a["attempt_id"] for a in attempts if a["origin"] == "imported"]
            for i in range(0, len(imported), 8): await tool("compare_attempts", attempt_ids=imported[i:i + 8] or imported[:2], scale="L")
            await tool("wait_events", cursor=0, timeout_s=0)
            for name in ("spec", "failures", "protocol", "costs", "configurations", "sources", "experiments", "plan"):
                offset = 0
                while offset is not None:
                    page = await tool("read_doc", name=name, offset=offset, max_chars=100000)
                    offset = page.get("next_offset") if isinstance(page, dict) else None
            for a in attempts:
                if a["origin"] == "imported": await tool("read_doc", name="method", method=a["title"], max_chars=100000)
            for uri in ["nextt://task", "nextt://spec", "nextt://budget", "nextt://leaderboard/S", "nextt://leaderboard/L",
                        "nextt://knowledge/failures"] + [f"nextt://upstream/{d}" for d in
                        ("protocol", "costs", "configurations", "sources", "experiments", "plan")] + \
                       [f"nextt://attempts/{a['attempt_id']}" for a in attempts]:
                await resource(uri)
            prompt = await session.get_prompt("research_iteration", {})
            outputs["prompt research_iteration"] = "".join(getattr(m.content, "text", "") for m in prompt.messages)
    leaks = {k: sorted(set(PATTERN.findall(v))) for k, v in outputs.items()}
    leaks = {k: v for k, v in leaks.items() if v}
    official_keys = {k: [m for m in re.findall(r'"(val_bpb_official|core)"', v)] for k, v in outputs.items()}
    official_keys = {k: v for k, v in official_keys.items() if v}
    summary = dict(calls=len(outputs), characters=sum(map(len, outputs.values())), published_values=len(values),
                   attempts=len(attempts), runs=len(runs), leaks=leaks, official_fields=official_keys)
    json.dump(dict(summary=summary, outputs=outputs), open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])


asyncio.run(main(sys.argv[1]))
