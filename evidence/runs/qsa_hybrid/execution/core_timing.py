"""CORE-only latency control. Batch construction, tokenizer, forward and scoring stay upstream."""
import json,os,sys,time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
active=False
checked=-1.
sample=75
next_sync=0.
rank=int(os.environ.get('LOCAL_RANK',0))

def pacing():
    """None lets genuinely CPU-limited forwards submit asynchronously."""
    global checked,sample,next_sync
    now=time.monotonic()
    if now-checked>.25:
        row=json.loads((root/'gpu-latest.json').read_text())
        sample=row['util'][rank]
        checked=now
    if sample<70:return None
    if now<next_sync:return None
    next_sync=now+(.001 if sample>88 else .004)
    return (.70 if sample>88 else .85)/int(os.environ.get("NEXTT_CORE_WORKERS","1"))

def install():
    import nanochat.base_eval as base
    tool=next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
    sys.monitoring.use_tool_id(tool,'NextTransformer CORE timing')
    def enter(code,offset):
        global active
        active=True
    def leave(code,offset,result):
        global active
        active=False
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_START,enter)
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_RETURN,leave)
    sys.monitoring.set_local_events(tool,base.evaluate_core.__code__,sys.monitoring.events.PY_START|sys.monitoring.events.PY_RETURN)
