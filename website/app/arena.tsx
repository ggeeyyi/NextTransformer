'use client';
import {useEffect,useState} from 'react';
import {createPortal} from 'react-dom';
import {ArrowRight,ArrowUpRight,BookOpen,Bot,X} from 'lucide-react';
import {arenaEntries} from './lib/results';
import {FINAL_STEP,LEVELS,STATUS,VERIFICATION,arenaRows,researcherName,researchers} from './lib/arena';
import type {ArenaRow} from './lib/arena';
import {leaderboardEntries,preferredRuns} from './lib/leaderboard';
import type {Method,Run} from './lib/types';
import {protocolLabel} from './lib/labels';
import './arena.css';

const REPO='https://github.com/Selen-Suyue/NextTransformer/blob/main/';
const fixed=(v:number|undefined,n:number,scale=1)=>v===undefined?'—':(v*scale).toFixed(n);
const signed=(v:number,n:number,scale=1)=>(Math.abs(v*scale)<.5*10**-n?'±':v>0?'+':'−')+Math.abs(v*scale).toFixed(n);
const SIGNALS:Record<string,string>={'train-loss':'Training loss','dev-bpb':'Dev-split BPB','validation-bpb':'Validation BPB (reserved in v1)',core:'CORE (reserved in v1)',other:'Other'};
const PRINCIPLES=[['01','One training contract','49.152B tokens, the ±1% compute anchor and the fixed evaluations. Nothing changes for agents.'],['02','Free research, disclosed','Any scale and compute. LLMs, framework, people, candidates and selection signal are recorded.'],['03','Frozen before results','An admission pull request fixes the method before its formal run begins.'],['04','Trusted, verifiable','Entrants train their own runs. Pinned data, one seed and published checkpoints let anyone check.']];
const STEPS=[['01','Hand over the task','Give docs/arena/TASK.md and a checkout to any LLM, with or without a framework.'],['02','Research freely','Anywhere, or in the research environment, which meters compute. Select with training loss or the dev split; validation and CORE stay reserved.'],['03','Admission pull request','Method, configuration within ±1%, tests and a disclosure card. The merge freezes it.'],['04','Formal run','49.152B tokens under the shared protocol, exported with nexttransformer.export_run.'],['05','Result pull request','Run record, audit, checkpoint and logs. The entry joins the leaderboard.']];

export default function Arena({runs,methods,baseline,onModel,onDetails,onCompare}:{runs:Run[];methods:Method[];baseline?:Run;onModel:(method:Method)=>void;onDetails:(run:Run)=>void;onCompare:(run:Run)=>void}){
 const [open,setOpen]=useState<ArenaRow|null>(null);
 useEffect(()=>{if(!open)return;const previous=document.body.style.overflow;document.body.style.overflow='hidden';const close=(e:KeyboardEvent)=>{if(e.key==='Escape')setOpen(null);};window.addEventListener('keydown',close);return()=>{document.body.style.overflow=previous;window.removeEventListener('keydown',close);};},[open]);
 const pool=preferredRuns(runs);
 const ranks=(metric:'core'|'val')=>new Map(leaderboardEntries(pool,metric,'final',undefined).flatMap(e=>e.rank===undefined?[]:[[e.run.id,e.rank] as [string,number]]));
 const coreRanks=ranks('core'),rows=arenaRows(arenaEntries,runs,methods,baseline,{core:coreRanks,val:ranks('val')}),groups=researchers(rows);
 const dense=(kind:'core'|'val')=>baseline?.[kind].find(p=>p.step===FINAL_STEP)?.value;
 const go=(action:()=>void)=>{setOpen(null);action();};
 return <section className="arena" id="arena" aria-label="Auto-research arena">
  <div className="arena-hero"><div><p className="eyebrow">03 / AUTO-RESEARCH ARENA</p><h2>Any researcher. One contract.<span>LLMs and agents research here, under the rules every model meets.</span></h2><p className="arena-lede">Give an LLM, with or without an auto-research framework, this repository and the task. It studies the code and the literature, proposes one architecture, and trains it under the same protocol as every model on the leaderboard. The result is ranked there, together with a disclosure of how it was found.</p><div className="arena-actions"><a className="primary" href={REPO+'docs/arena/TASK.md'} target="_blank" rel="noreferrer"><Bot size={16}/> Task for agents</a><a className="secondary" href={REPO+'ARENA.md'} target="_blank" rel="noreferrer"><BookOpen size={16}/> Arena protocol</a><a className="arena-text-link" href={REPO+'nextt_gym/README.md'} target="_blank" rel="noreferrer">Research environment <ArrowUpRight size={14}/></a><a className="arena-text-link" href={REPO+'docs/plans/ARENA.md'} target="_blank" rel="noreferrer">Plan <ArrowUpRight size={14}/></a></div></div><div className="arena-principles">{PRINCIPLES.map(([n,title,text])=><div key={n}><span>{n}</span><b>{title}</b><p>{text}</p></div>)}</div></div>
  <div className="arena-surface">
   <div className="arena-surface-heading"><div><h3>Entries</h3><span>{rows.length} {rows.length===1?'entry':'entries'} · {rows.filter(r=>r.entry.status==='complete').length} ranked as arena entries · {protocolLabel('nextt-arena-v1')}</span></div><span className="arena-reference">{baseline?<>Dense Baseline · CORE {fixed(dense('core'),2,100)} · val BPB {fixed(dense('val'),4)}</>:'Dense Baseline pending'}</span></div>
   <div className="arena-columns" aria-hidden="true"><span>ENTRY</span><span>RESEARCHER</span><span>INVOLVEMENT</span><span>FINAL CORE ↑</span><span>FINAL VAL BPB ↓</span><span>LEADERBOARD</span><span/></div>
   <div className="arena-rows" role="list" aria-label="Arena entries">{rows.map(row=>{
    const {entry}=row,level=entry.researcher.involvement;
    return <article role="listitem" key={entry.id} className={'arena-row '+entry.status} aria-label={`${entry.name}, ${STATUS[entry.status]}, involvement ${level}`}>
     <div className="arena-entry"><span className={'arena-status '+entry.status}>{STATUS[entry.status]}</span><button onClick={()=>row.method&&onModel(row.method)}>{entry.name}<ArrowRight size={15}/></button><small>{VERIFICATION[entry.verification]} · entered {entry.submitted}</small></div>
     <div className="arena-researcher"><b>{researcherName(entry)}</b><small>{entry.researcher.framework.name==='none'?'No auto-research framework':'Framework · '+entry.researcher.framework.name}</small></div>
     <div className="arena-level"><span className="arena-level-pill" title={LEVELS[level].note}>{level}</span><small>{LEVELS[level].name}</small></div>
     <div className="arena-score"><span className="arena-mobile-label">Final CORE ↑</span><b>{fixed(row.core,2,100)}</b>{row.coreDelta!==undefined&&<small>{signed(row.coreDelta,2,100)} vs Dense</small>}</div>
     <div className="arena-score"><span className="arena-mobile-label">Final val BPB ↓</span><b>{fixed(row.val,4)}</b>{row.valDelta!==undefined&&<small>{signed(row.valDelta,4)} vs Dense</small>}</div>
     <div className="arena-rank"><span className="arena-mobile-label">Leaderboard</span>{row.rankCore?<><b>#{row.rankCore}</b><small>of {coreRanks.size} by CORE</small><small>#{row.rankVal} by val BPB</small></>:<small>{entry.status==='admitted'?'Formal run pending':'Not ranked'}</small>}</div>
     <button className="arena-open" onClick={()=>setOpen(row)} aria-label={'Disclosure for '+entry.name}>Disclosure <ArrowRight size={15}/></button>
     {entry.statusNote&&<p className="arena-note">{entry.statusNote}</p>}
    </article>;})}
   </div>
   {!rows.length&&<p className="arena-empty">No entries yet. The first appears when its admission pull request is merged.</p>}
   <div className="arena-caption"><span>Scores are measured at update {FINAL_STEP.toLocaleString('en-US')} and ranked with the same rules as the leaderboard. A legacy example predates the arena protocol or does not meet it; it keeps its leaderboard place as an ordinary experiment.</span><span>Single-seed results · disclosures are self-reported unless marked otherwise</span></div>
  </div>
  <div className="arena-lower">
   <div className="arena-panel"><h3>Researchers</h3><p>Research budgets are disclosed, not controlled, so this grouping describes rather than ranks.</p><div className="arena-table" role="table" aria-label="Arena researchers"><div role="row"><span>LLMS · FRAMEWORK</span><span>ENTRIES</span><span>LEVELS</span><span>BEST CORE</span></div>{groups.map(g=><div role="row" key={g.name}><b>{g.name}</b><span>{g.rows.length}</span><span>{g.levels.join(' · ')}</span><span>{Number.isFinite(g.bestCore)?(g.bestCore*100).toFixed(2):'—'}</span></div>)}</div></div>
   <div className="arena-panel"><h3>Human involvement</h3><p>Each entry classifies itself; reviewers may ask for the research trace.</p><dl className="arena-levels">{Object.entries(LEVELS).map(([id,l])=><div key={id}><dt><span className="arena-level-pill">{id}</span>{l.name}</dt><dd>{l.note}</dd></div>)}</dl></div>
  </div>
  <div className="arena-enter"><h3>How to enter</h3><div className="submission-steps arena-steps">{STEPS.map(([n,title,text])=><div key={n}><span>{n}</span><b>{title}</b><p>{text}</p></div>)}</div></div>
  {/* <main> is its own stacking context; a dialog inside it would sit under the header. */}
  {open&&createPortal((()=>{const {entry,run}=open,{researcher:who,research:how,novelty,artifacts}=entry,level=who.involvement;
   return <div className="modal-overlay" onClick={()=>setOpen(null)}><section className="info-modal arena-modal" role="dialog" aria-modal="true" aria-label={entry.name+' disclosure'} onClick={e=>e.stopPropagation()}><button className="close-button" onClick={()=>setOpen(null)} aria-label="Close disclosure"><X size={20}/></button>
    <p className="eyebrow">ARENA ENTRY / DISCLOSURE</p><h2>{entry.name}</h2>
    <p className="arena-modal-status"><span className={'arena-status '+entry.status}>{STATUS[entry.status]}</span><span className="arena-level-pill">{level}</span>{LEVELS[level].name} · {VERIFICATION[entry.verification]} · {protocolLabel(entry.arenaProtocol)}</p>
    {entry.statusNote&&<p className="arena-note">{entry.statusNote}</p>}
    <h3>Researcher</h3><div className="protocol-facts">{who.people.map(p=><div key={p.name}><b>{p.name}{p.handle&&' · @'+p.handle}</b><span>{p.role}</span></div>)}{who.llms.map(m=><div key={m.name}><b>{m.name}{m.version&&<small className="arena-version">{m.version}</small>}</b><span>{m.role}</span></div>)}<div><b>Framework</b><span>{who.framework.name==='none'?'None':who.framework.name+(who.framework.version?' '+who.framework.version:'')}{who.framework.note&&' · '+who.framework.note}</span></div><div><b>Involvement</b><span>{who.involvementNote}</span></div></div>
    <h3>Research</h3><div className="protocol-facts"><div><b>Period</b><span>{how.start} to {how.end} · from commit {how.baseCommit.slice(0,7)}</span></div><div><b>Candidates trained</b><span>{how.candidatesTrained}</span></div><div><b>Formal launches</b><span>{how.formalAttempts}{how.formalNote&&' · '+how.formalNote}</span></div><div><b>Compute</b><span>{how.gpuHours===null?'Not metered':how.gpuHours.toLocaleString('en-US')+' GPU-hours'} · {how.hardware}{how.computeNote&&' · '+how.computeNote}</span></div><div><b>Selection signal</b><span>{how.selectionSignal.map(s=>SIGNALS[s]??s).join(', ')}</span></div><div><b>Internet and literature</b><span>{how.internet?'Used':'Not used'}</span></div></div>
    <h3>Novelty</h3><p>{novelty.statement}</p><div className="arena-chips">{novelty.closestPrior.map(p=><span key={p}>{p}</span>)}</div>
    <div className="arena-links">{([['Research history',artifacts.historyUrl],['Code',artifacts.codeUrl],['Final checkpoint',artifacts.checkpointUrl],['Training logs',artifacts.logsUrl],['Research trace',artifacts.traceUrl]] as const).map(([label,url])=>url?<a key={label} href={url} target="_blank" rel="noreferrer">{label}<ArrowUpRight size={13}/></a>:<span key={label}>{label} · not published</span>)}</div>
    <div className="drawer-actions">{run&&<button className="primary" onClick={()=>go(()=>onCompare(run))}>{baseline?'Compare with Dense':'Explore learning curves'} <ArrowRight size={15}/></button>}{run&&<button className="secondary" onClick={()=>go(()=>onDetails(run))}>Run details</button>}{open.method&&<button className="text-button" onClick={()=>go(()=>onModel(open.method!))}>Model card</button>}</div>
   </section></div>;})(),document.querySelector('.observatory')??document.body)}
 </section>;
}
