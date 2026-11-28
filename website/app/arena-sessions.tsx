'use client';
import {useEffect,useRef,useState} from 'react';
import {ArrowRight,ArrowUpRight} from 'lucide-react';
import Chart,{seriesColor} from './learning-chart';
import {bpbDifference} from './lib/chart-series';
import {FINAL_STEP} from './lib/arena';
import {arenaEntries,loadArenaSessions} from './lib/results';
import {budgetShare,changeList,changeText,devPath,formalResult,hoursBetween,mainStep,roleOf,screeningSteps,spendCurve,standings,trainedCandidates} from './lib/arena-sessions';
import type {ArenaSession,SessionCandidate} from './lib/arena-sessions';
import {methodLabel} from './lib/labels';
import type {Method,Run} from './lib/types';
import './arena-sessions.css';

// The published method every session could build on, shown as the reference in the formal comparison.
const REFERENCE='attnres_full_canon_abcd';
const CONTROL='#5E6166',OTHER='#8FA3B8';
const SUP:Record<string,string>={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};
const flops=(v:number)=>{const e=Math.floor(Math.log10(v));return (v/10**e).toFixed(2)+' × 10'+[...String(e)].map(c=>SUP[c]).join('');};
const tokens=(n:number)=>n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(0)+'K':String(n);
const duration=(h:number)=>h<48?h.toFixed(1)+' h':(h/24).toFixed(1)+' days';
const share=(x:number)=>(x*100>=99.95?(x*100).toFixed(3):(x*100).toFixed(1))+'%';
const steps=(n:number)=>n>=1000?Number((n/1000).toFixed(n%1000?1:0))+'k':String(n);
const signed=(v:number,n:number)=>(v<0?'−':v>0?'+':'±')+Math.abs(v).toFixed(n);
const day=(iso:string)=>new Date(iso).toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'});
const final=(run:Run|undefined,kind:'val'|'core')=>run?.[kind].find(p=>p.step===FINAL_STEP)?.value;

function useWidth(initial:number){
 const ref=useRef<HTMLDivElement>(null),[width,setWidth]=useState(initial);
 useEffect(()=>{if(!ref.current)return;const o=new ResizeObserver(([e])=>setWidth(Math.max(280,e.contentRect.width)));o.observe(ref.current);return()=>o.disconnect();},[]);
 return [ref,width] as const;
}

type Props={runs:Run[];methods:Method[];baseline?:Run;coreRanks:Map<string,number>;onModel:(method:Method)=>void;onCompare:(run:Run)=>void};

export default function ArenaSessions({runs,methods,baseline,coreRanks,onModel,onCompare}:Props){
 const [sessions,setSessions]=useState<ArenaSession[]|null>(null),[active,setActive]=useState(0);
 // #arena?section=research opens the Arena tab at this section once the sessions have loaded.
 useEffect(()=>{if(sessions?.length&&new URLSearchParams(window.location.hash.split('?')[1]).get('section')==='research')document.getElementById('arena-research')?.scrollIntoView({block:'start'});},[sessions]);
 useEffect(()=>{let live=true;loadArenaSessions().then(list=>{if(live)setSessions(list);},()=>{if(live)setSessions([]);});return()=>{live=false;};},[]);
 if(sessions===null)return <section className="sessions" id="arena-research" aria-busy="true"><p className="sessions-loading">Loading research sessions…</p></section>;
 if(!sessions.length)return null;
 const name=(id:string)=>{const m=methods.find(x=>x.id===id);return m?methodLabel(m):id;};
 const chartMethods=[{id:REFERENCE,name:name(REFERENCE)},...sessions.map(s=>({id:s.submission.method,name:name(s.submission.method)}))];
 const color=(s:ArenaSession)=>seriesColor(s.submission.method,chartMethods);
 const session=sessions[Math.min(active,sessions.length-1)];
 const latest=sessions.map(s=>s.gym.instruction).sort((a,b)=>sessions.filter(s=>s.gym.instruction===b).length-sessions.filter(s=>s.gym.instruction===a).length)[0];
 const earlier=sessions.filter(s=>s.gym.instruction!==latest);
 const open=(id:string)=>{const m=methods.find(x=>x.id===id);return m?()=>onModel(m):undefined;};
 return <section className="sessions" id="arena-research" aria-labelledby="sessions-title">
  <div className="arena-surface">
   <div className="arena-surface-heading"><div><h3 id="sessions-title">Research sessions</h3><span>{sessions.length} autonomous sessions in the research gym · the same budget of {flops(sessions[0].budget.limitFlops)} training FLOPs each · no human input between the task and the submission</span></div><span className="arena-reference">Research is selected on the dev split; validation and CORE stay reserved for the formal run</span></div>
   <div className="sessions-columns" aria-hidden="true"><span>RESEARCHER</span><span>TIME</span><span>RESEARCH COMPUTE</span><span>DESIGNS</span><span>LLM TOKENS</span><span>SUBMISSION</span><span>FORMAL RESULT</span></div>
   <div className="sessions-rows" role="list" aria-label="Research sessions">{sessions.map((s,i)=>{const {run}=formalResult(s,arenaEntries,runs),core=final(run,'core'),val=final(run,'val'),rank=run&&coreRanks.get(run.id),go=open(s.submission.method);
    return <article role="listitem" key={s.id} className={'sessions-row'+(i===active?' active':'')} style={{['--session' as string]:color(s)}}>
     <button className="sessions-agent" onClick={()=>setActive(i)} aria-pressed={i===active} aria-label={'Show the research of '+s.agent.name}><i/><span><b>{s.agent.name}</b><small>{s.agent.harness} {s.agent.harnessVersion}{s.agent.effort&&' · effort '+s.agent.effort}</small><small>{s.gym.instruction===latest?'Arena task text':'Earlier task text'} · <code>{s.gym.instruction.slice(0,7)}</code></small></span></button>
     <div><span className="sessions-mobile-label">Time</span><b>{duration(hoursBetween(s.start,s.end))}</b><small>{day(s.start)} – {day(s.end)}</small></div>
     <div><span className="sessions-mobile-label">Research compute</span><b>{share(budgetShare(s))}</b><span className="sessions-meter" aria-hidden="true"><i style={{width:Math.min(100,budgetShare(s)*100)+'%'}}/></span><small>{flops(s.budget.spentFlops)} FLOPs · {s.hardware}</small></div>
     <div><span className="sessions-mobile-label">Designs</span><b>{s.candidatesTrained} trained</b><small>{s.candidatesRegistered} registered · {s.runs.length} training jobs, {s.runs.filter(r=>r.state!=='succeeded').length} cancelled</small></div>
     <div><span className="sessions-mobile-label">LLM tokens</span><b>{tokens(s.tokens.input)} in</b><small>{tokens(s.tokens.output)} out · {Math.round(s.tokens.cachedInput/s.tokens.input*100)}% of input cached</small></div>
     <div className="sessions-submission"><span className="sessions-mobile-label">Submission</span>{go?<button onClick={go}>{name(s.submission.method)} <ArrowRight size={14}/></button>:<b><code>{s.submission.method}</code></b>}<small>submitted after {duration(hoursBetween(s.start,s.submittedAt))}</small></div>
     <div><span className="sessions-mobile-label">Formal result</span>{run&&core!==undefined&&val!==undefined?<><b>CORE {(core*100).toFixed(2)}</b><small>val BPB {val.toFixed(4)}{rank?` · #${rank} by CORE`:''}</small></>:<small>{s.submission.arenaEntry?'Formal record pending':'Not trained formally'}</small>}</div>
    </article>;})}</div>
   <div className="arena-caption"><span>Every session had the same data, protocol and budget; the agent chose the controls, the candidates, how long to train each and when to stop. Compute is the gym's metered training FLOPs; tokens are the agent's own model calls. {earlier.length>0&&<> {earlier.map(s=>s.agent.name).join(' and ')} ran on an earlier task text, which also stated that unused research budget earns no reward.</>}</span><span>One session per model · not a controlled comparison of models</span></div>
  </div>
  <SessionDetail key={session.id} session={session} color={color(session)} name={name}/>
  <FormalRuns sessions={sessions} runs={runs} baseline={baseline} coreRanks={coreRanks} chartMethods={chartMethods} color={color} name={name} onCompare={onCompare}/>
 </section>;
}

function SessionDetail({session:s,color,name}:{session:ArenaSession;color:string;name:(id:string)=>string}){
 const selected=s.candidates.find(c=>c.selected)??s.candidates.find(c=>c.submitted);
 const [focus,setFocusState]=useState(selected?.id??''),[picked,setPicked]=useState(false),[mode,setMode]=useState<'raw'|'delta'>('raw');
 const setFocus=(id:string)=>{setFocusState(id);setPicked(true);};
 const hasControl=s.candidates.some(c=>c.control&&c.trained);
 const candidate=s.candidates.find(c=>c.id===focus);
 return <div className="sessions-detail" style={{['--session' as string]:color}}>
  <div className="sessions-detail-heading"><div><p className="eyebrow">SEARCH TRAJECTORY</p><h3>{s.agent.name} <span>· {name(s.submission.method)}</span></h3></div>
   <div className="sessions-key" aria-label="Legend"><span><i className="selected"/>Submitted design</span><span><i className="control"/>Agent's control</span><span><i className="other"/>Other candidates</span><span><i className="cancelled"/>Cancelled job</span></div></div>
  <div className="sessions-card"><h4>Jobs over time</h4><p>Each row is a candidate the agent queued for training, in the order it was first queued; bars are training jobs from their first metered compute to their result. The band on top is the share of the research budget spent.</p><Timeline session={s} color={color} focus={focus} dim={picked} onFocus={setFocus}/></div>
  <div className="sessions-pair">
   <div className="sessions-card"><div className="sessions-card-heading"><h4>Dev BPB by update</h4>{hasControl&&<div className="metric-switch" role="group" aria-label="Dev BPB view"><button aria-pressed={mode==='raw'} onClick={()=>setMode('raw')}>Dev BPB <span>↓</span></button><button aria-pressed={mode==='delta'} onClick={()=>setMode('delta')}>Δ vs control</button></div>}</div><p>{mode==='raw'?'One point per finished training job; a line joins the jobs that continued one candidate. Updates are on a log scale.':"Difference from the agent's control at the same update; only updates where the control was measured."}</p><DevChart session={s} color={color} mode={mode} focus={focus} dim={picked} onFocus={setFocus}/></div>
   <Standings session={s} color={color} focus={focus} onFocus={setFocus}/>
  </div>
  {candidate&&<CandidateCard session={s} candidate={candidate} color={color}/>}
 </div>;
}

const roleColor=(c:SessionCandidate,color:string)=>({selected:color,control:CONTROL,candidate:OTHER})[roleOf(c)];

function Timeline({session:s,color,focus,dim:dimming,onFocus}:{session:ArenaSession;color:string;focus:string;dim:boolean;onFocus:(id:string)=>void}){
 const [ref,W]=useWidth(900);
 const first=new Map<string,string>();for(const r of s.runs)if(!first.has(r.candidate)||r.queuedAt<first.get(r.candidate)!)first.set(r.candidate,r.queuedAt);
 const rows=s.candidates.filter(c=>first.has(c.id)).sort((a,b)=>first.get(a.id)!.localeCompare(first.get(b.id)!));
 const total=hoursBetween(s.start,s.end),narrow=W<600,L=narrow?74:118,R=14,band=54,top=band+22,rowH=narrow?19:22,H=top+rows.length*rowH+30;
 const x=(iso:string)=>L+Math.max(0,hoursBetween(s.start,iso))/total*(W-L-R);
 const unit=total>72?24:1,span=total/unit,tick=[.25,.5,1,2,3,6,12,24].find(t=>span/t<=(narrow?4:8))??48;
 const ticks=Array.from({length:Math.floor(span/tick)+1},(_,i)=>i*tick);
 const curve=spendCurve(s),by=(v:number)=>band-v*(band-8);
 const spendPath=curve.map((p,i)=>`${i?'L':'M'}${L+p.hours/total*(W-L-R)},${by(p.share)}`).join(' ');
 return <div className="sessions-plot" ref={ref}>
  <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Training jobs of ${s.agent.name} over ${duration(total)}, with the research budget spent`}>
   <line x1={L} x2={W-R} y1={by(0)} y2={by(0)} className="grid-line"/><line x1={L} x2={W-R} y1={by(1)} y2={by(1)} className="grid-line"/>
   <text x={L-8} y={by(1)+4} textAnchor="end" className="axis-text">100%</text><text x={L-8} y={by(0)+4} textAnchor="end" className="axis-text">0%</text>
   <path d={spendPath+` L${W-R},${by(curve.at(-1)!.share)} L${W-R},${by(0)} L${L},${by(0)} Z`} fill={color} opacity={.12}/>
   <path d={spendPath+` L${W-R},${by(curve.at(-1)!.share)}`} fill="none" stroke={color} strokeWidth={2}/>
   <text x={W-R} y={by(curve.at(-1)!.share)-7} textAnchor="end" className="sessions-plot-value" fill={color}>{share(budgetShare(s))} of budget</text>
   {rows.map((c,i)=>{const y=top+i*rowH,mid=y+rowH/2,hue=roleColor(c,color),dim=dimming&&focus!==c.id;
    return <g key={c.id} className="sessions-row-plot" opacity={dim?.45:1} onClick={()=>onFocus(c.id)} role="button" tabIndex={0} aria-label={`Candidate ${c.id}${c.selected?', submitted design':c.control?', control':''}`} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onFocus(c.id);}}}>
     {focus===c.id&&<rect x={0} y={y} width={W} height={rowH} rx={4} className="sessions-focus-row"/>}
     <rect x={0} y={y} width={W} height={rowH} fill="transparent"/>
     <text x={L-8} y={mid+4} textAnchor="end" className="sessions-id">{c.id}</text>
     {s.runs.filter(r=>r.candidate===c.id).map((r,j)=>{const end=x(r.endedAt),start=r.startedAt?x(r.startedAt):end,cancelled=r.state!=='succeeded';
      return <g key={j}><title>{`${c.id} · updates ${r.from.toLocaleString('en-US')}–${r.to.toLocaleString('en-US')} · ${r.state}${r.devBpb!==null?' · dev BPB '+r.devBpb.toFixed(4):''}`}</title>
       <line x1={x(r.queuedAt)} x2={start} y1={mid} y2={mid} className="sessions-queue"/>
       {r.startedAt&&<rect x={start} y={y+5} width={Math.max(2.5,end-start)} height={rowH-10} rx={2} fill={cancelled?'white':hue} stroke={hue} strokeWidth={cancelled?1.2:0} strokeDasharray={cancelled?'3 2':undefined}/>}
       {!r.startedAt&&<line x1={end} x2={end} y1={y+6} y2={y+rowH-6} stroke={hue} strokeWidth={1.5}/>}
      </g>;})}
    </g>;})}
   {(()=>{const sx=x(s.submittedAt);return <g><line x1={sx} x2={sx} y1={band+4} y2={H-26} className="sessions-submit-line" stroke={color}/><text x={sx-5} y={band+16} textAnchor="end" className="axis-text">submitted</text></g>;})()}
   {ticks.map(t=>{const tx=L+t*unit/total*(W-L-R);return <text key={t} x={tx} y={H-12} textAnchor={t===0?'start':'middle'} className="axis-text">{unit===24?'day '+Number(t.toFixed(2)):Number(t.toFixed(2))+' h'}</text>;})}
  </svg>
 </div>;
}

function DevChart({session:s,color,mode,focus,dim:dimming,onFocus}:{session:ArenaSession;color:string;mode:'raw'|'delta';focus:string;dim:boolean;onFocus:(id:string)=>void}){
 const [ref,W]=useWidth(520),[hover,setHover]=useState<{id:string;update:number;value:number}|null>(null);
 const control=s.candidates.find(c=>c.control),reference=new Map(control?devPath(s,control.id).map(p=>[p.update,p.bpb]):[]);
 // Checks shorter than the update most candidates were compared at would stretch the scale; they are left out.
 const from=mainStep(s)??0,hidden=s.runs.filter(r=>r.devBpb!==null&&r.to<from).length;
 const series=trainedCandidates(s).map(c=>({c,points:devPath(s,c.id).flatMap(p=>mode==='raw'?(p.update>=from?[{update:p.update,value:p.bpb}]:[]):reference.has(p.update)?[{update:p.update,value:p.bpb-reference.get(p.update)!}]:[])})).filter(x=>x.points.length);
 const all=series.flatMap(x=>x.points);
 if(!all.length)return <p className="sessions-empty">No measured results.</p>;
 const H=W<560?250:290,L=mode==='delta'?70:52,R=16,T=14,B=34;
 const u0=Math.min(...all.map(p=>p.update))/1.25,u1=Math.max(...all.map(p=>p.update))*1.25;
 let lo=Math.min(...all.map(p=>p.value)),hi=Math.max(...all.map(p=>p.value));if(mode==='delta'){lo=Math.min(lo,0);hi=Math.max(hi,0);}const pad=(hi-lo)*.12||.005;lo-=pad;hi+=pad;
 const x=(u:number)=>L+Math.log(u/u0)/Math.log(u1/u0)*(W-L-R),y=(v:number)=>H-B-(v-lo)/(hi-lo)*(H-T-B);
 const ticks=[16,32,64,128,256,512,1024,2048,4096,8192,16384,32768].filter(u=>u>=u0&&u<=u1),shown=ticks.filter((_,i)=>ticks.length<=(W<560?4:7)||i%2===0);
 const order=[...series].sort((a,b)=>Number(a.c.id===focus)-Number(b.c.id===focus)||Number(a.c.selected)-Number(b.c.selected));
 const digits=mode==='delta'?4:3;
 return <div className="sessions-plot" ref={ref}>
  <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={mode==='raw'?'Dev BPB of each candidate by training update; lower is better':"Dev BPB difference from the agent's control at the same update; below zero is better"} onMouseLeave={()=>setHover(null)}>
   {[0,1,2,3,4].map(i=>{const v=lo+(hi-lo)*i/4;return <g key={i}><line x1={L} x2={W-R} y1={y(v)} y2={y(v)} className="grid-line"/><text x={L-8} y={y(v)+4} textAnchor="end" className="axis-text">{mode==='delta'?signed(v,4):v.toFixed(digits)}</text></g>;})}
   {shown.map(u=><text key={u} x={x(u)} y={H-12} textAnchor="middle" className="axis-text">{steps(u)}</text>)}
   {mode==='delta'&&<line x1={L} x2={W-R} y1={y(0)} y2={y(0)} stroke={CONTROL} strokeDasharray="5 5"/>}
   {order.map(({c,points})=>{const hue=roleColor(c,color),dim=dimming&&focus!==c.id&&!c.selected&&!c.control;
    return <g key={c.id} opacity={dim?.35:1} className="sessions-series" onClick={()=>onFocus(c.id)}>
     {points.length>1&&<path d={points.map((p,i)=>`${i?'L':'M'}${x(p.update)},${y(p.value)}`).join(' ')} fill="none" stroke={hue} strokeWidth={c.selected||c.id===focus?2.6:1.6} strokeDasharray={c.control?'6 4':undefined}/>}
     {points.map(p=><circle key={p.update} cx={x(p.update)} cy={y(p.value)} r={c.selected||c.id===focus?4.5:3.4} fill={c.control?'white':hue} stroke={hue} strokeWidth={1.6} onMouseEnter={()=>setHover({id:c.id,update:p.update,value:p.value})}/>)}
    </g>;})}
  </svg>
  {hover&&<div className="chart-tooltip sessions-tooltip"><small>Candidate {hover.id}</small><div><b>{mode==='delta'?signed(hover.value,4):hover.value.toFixed(4)}</b><em>update {hover.update.toLocaleString('en-US')}</em></div></div>}
  <div className="plot-caption"><span>Training update (log scale){mode==='raw'&&hidden>0?` · ${hidden} shorter ${hidden===1?'check':'checks'} before ${from.toLocaleString('en-US')} not shown`:''}</span><span>{mode==='raw'?'Bits / byte on the dev split · lower is better':'Δ dev BPB · below zero is better'}</span></div>
 </div>;
}

function Standings({session:s,color,focus,onFocus}:{session:ArenaSession;color:string;focus:string;onFocus:(id:string)=>void}){
 const options=screeningSteps(s),[step,setStep]=useState(mainStep(s));
 const rows=step===undefined?[]:standings(s,step);
 return <div className="sessions-card sessions-standings"><h4>Standing at one update</h4><p>Candidates measured at exactly the same update, best first.</p>
  {options.length>1&&<div className="chart-mode sessions-steps" role="group" aria-label="Update">{options.map(u=><button key={u} className={u===step?'active':''} aria-pressed={u===step} onClick={()=>setStep(u)}>{u.toLocaleString('en-US')}</button>)}</div>}
  <div className="sessions-table" role="table" aria-label={`Candidates at update ${step}`}>
   <div role="row" className="sessions-table-head"><span role="columnheader">#</span><span role="columnheader">CANDIDATE</span><span role="columnheader">DEV BPB ↓</span><span role="columnheader">Δ CONTROL</span></div>
   {rows.map(r=><button role="row" key={r.candidate.id} className={'sessions-table-row'+(focus===r.candidate.id?' focus':'')} onClick={()=>onFocus(r.candidate.id)}>
    <span role="cell">{r.rank}</span>
    <span role="cell" className="sessions-candidate"><i style={{background:roleColor(r.candidate,color)}}/><span><code>{r.candidate.id}</code>{r.candidate.selected&&<em>submitted</em>}{r.candidate.control&&<em className="control">control</em>}<small>{changeText(r.candidate)}</small></span></span>
    <span role="cell">{r.bpb.toFixed(4)}</span>
    <span role="cell">{r.candidate.control?'—':r.delta===undefined?'n/a':signed(r.delta,4)}</span>
   </button>)}
  </div>
 </div>;
}

function CandidateCard({session:s,candidate:c,color}:{session:ArenaSession;candidate:SessionCandidate;color:string}){
 const jobs=s.runs.filter(r=>r.candidate===c.id);
 return <div className="sessions-card sessions-candidate-card" aria-live="polite">
  <div className="sessions-candidate-head"><span className="sessions-role" style={{background:roleColor(c,color)}}>{c.selected?'Submitted design':c.control?"Agent's control":c.trained?'Candidate':'Queued, never measured'}</span><h4><code>{c.id}</code> · {c.title}</h4></div>
  <div className="sessions-facts">
   <div><b>Shape</b><span>{c.layers??'—'} layers · FFN {c.mlpWidth?.toLocaleString('en-US')??'—'}</span></div>
   <div><b>Parameters</b><span>{c.parameters?(c.parameters/1e9).toFixed(3)+'B':'—'}</span></div>
   <div><b>Training cost</b><span>{c.anchorRatio?((c.anchorRatio-1)*100>=0?'+':'−')+Math.abs((c.anchorRatio-1)*100).toFixed(2)+'% FLOPs per token vs the anchor':'—'}</span></div>
   <div className="wide"><b>Jobs</b><span>{jobs.map(r=>`${r.from.toLocaleString('en-US')}→${r.to.toLocaleString('en-US')} ${r.state==='succeeded'&&r.devBpb!==null?r.devBpb.toFixed(4):r.state}`).join(' · ')||'None'}</span></div>
  </div>
  {c.basis&&<p className="sessions-changes"><b>{changeList(c).length?`Differs from ${c.basis.candidate}, the closest candidate queued before it:`:`Same design as ${c.basis.candidate}, registered again.`}</b>{changeList(c).map(x=><code key={x}>{x}</code>)}</p>}
  <p className="sessions-hypothesis"><b>Hypothesis, in the agent's words.</b> {c.hypothesis}</p>
 </div>;
}

function FormalRuns({sessions,runs,baseline,coreRanks,chartMethods,color,name,onCompare}:{sessions:ArenaSession[];runs:Run[];baseline?:Run;coreRanks:Map<string,number>;chartMethods:{id:string;name:string}[];color:(s:ArenaSession)=>string;name:(id:string)=>string;onCompare:(run:Run)=>void}){
 const [metric,setMetric]=useState<'val'|'core'>('val');
 const items=sessions.map(s=>({s,...formalResult(s,arenaEntries,runs)}));
 const referenceEntry=arenaEntries.find(e=>e.methodId===REFERENCE);
 const reference=runs.filter(r=>r.methodId===REFERENCE&&(!referenceEntry||referenceEntry.runIds.includes(r.id))).sort((a,b)=>(b.lastRecordedAt??0)-(a.lastRecordedAt??0))[0];
 const formal=items.flatMap(i=>i.run?[i.run]:[]);
 if(!formal.length)return <div className="sessions-formal"><div className="sessions-card"><h4>Formal runs</h4><p>The formal results of the submitted designs appear here once their records are published.</p></div></div>;
 const shown=[...formal,...(reference?[reference]:[]),...(baseline?[baseline]:[])];
 const delta=metric==='val'&&!!baseline;
 const plotted=delta?bpbDifference(shown,baseline):shown;
 const methods=[...chartMethods,...(baseline?[{id:baseline.methodId,name:'Dense Baseline'}]:[])];
 const refVal=final(reference,'val'),refCore=final(reference,'core');
 return <div className="sessions-formal">
  <div className="sessions-detail-heading"><div><p className="eyebrow">FORMAL RUNS</p><h3>The submitted designs at the full protocol</h3></div></div>
  <div className="sessions-pair">
   <div className="sessions-card"><div className="sessions-card-heading"><h4>Learning curves</h4><div className="metric-switch" role="group" aria-label="Formal metric"><button aria-pressed={metric==='val'} onClick={()=>setMetric('val')}>{baseline?'Δ val BPB vs Dense':'Validation BPB'} <span>↓</span></button><button aria-pressed={metric==='core'} onClick={()=>setMetric('core')}>CORE <span>↑</span></button></div></div>
    <div className="plot-key">{items.filter(i=>i.run).map(({s})=><span key={s.id}><i style={{background:color(s)}}/>{s.agent.name}</span>)}{reference&&<span><i style={{background:seriesColor(REFERENCE,chartMethods)}}/>{name(REFERENCE)}</span>}{baseline&&<span><i className="dense-key"/>Dense Baseline</span>}</div>
    <Chart runs={plotted} methods={methods} metric={metric} cursor={FINAL_STEP} wide compact delta={delta} ariaLabel={delta?'Validation BPB difference from Dense of the formal runs':metric==='val'?'Validation BPB of the formal runs':'CORE of the formal runs'}/>
    <div className="plot-caption"><span>Training step</span><span>{metric==='core'?'CORE ×100 · markers are evaluations':delta?'Exact shared checkpoints · below zero is better':'Bits / byte · lower is better'}</span></div>
   </div>
   <div className="sessions-card"><h4>Final scores</h4><p>Update {FINAL_STEP.toLocaleString('en-US')}; differences are from {name(REFERENCE)}, the Arena v1 entry every session could build on.</p>
    <div className="sessions-table sessions-final" role="table" aria-label="Final scores of the formal runs">
     <div role="row" className="sessions-table-head"><span role="columnheader">RESEARCHER</span><span role="columnheader">CORE ↑</span><span role="columnheader">VAL BPB ↓</span><span role="columnheader"/></div>
     {items.map(({s,entry,run})=>{const core=final(run,'core'),val=final(run,'val'),rank=run&&coreRanks.get(run.id);
      return <div role="row" key={s.id} className="sessions-table-row static">
       <span role="cell" className="sessions-candidate"><i style={{background:color(s)}}/><span><b>{s.agent.name}</b><small>{name(s.submission.method)}{rank?` · #${rank} by CORE`:''}</small></span></span>
       {run&&core!==undefined&&val!==undefined?<><span role="cell">{(core*100).toFixed(2)}<small>{refCore!==undefined&&signed((core-refCore)*100,2)}</small></span><span role="cell">{val.toFixed(4)}<small>{refVal!==undefined&&signed(val-refVal,4)}</small></span><span role="cell">{entry?.status!=='arena-only'&&<button className="sessions-link" onClick={()=>onCompare(run)} aria-label={'Compare '+s.agent.name+' with Dense'}><ArrowUpRight size={15}/></button>}</span></>:<span role="cell" className="sessions-wide">{s.submission.arenaEntry?'Formal record pending':'Not trained formally'}</span>}
      </div>;})}
     {reference&&refCore!==undefined&&refVal!==undefined&&<div role="row" className="sessions-table-row static reference"><span role="cell" className="sessions-candidate"><i style={{background:seriesColor(REFERENCE,chartMethods)}}/><span><b>{name(REFERENCE)}</b><small>Arena v1 entry · reference</small></span></span><span role="cell">{(refCore*100).toFixed(2)}</span><span role="cell">{refVal.toFixed(4)}</span><span role="cell"/></div>}
    </div>
   </div>
  </div>
 </div>;
}
