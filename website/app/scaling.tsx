'use client';
import {memo,useRef,useState} from 'react';
import {ArrowUpRight,BookOpen,ChevronDown,RotateCcw} from 'lucide-react';
import {REPO_URL,docs} from './lib/links';
import {TONES} from './lib/palette';
import {scalingLadder,scalingRecords} from './lib/results';
import {FINAL_STEP,LADDER_STATUS,VIEWS,cellStatus,compact,familyKey,plannedPoints,scalingPoints} from './lib/scaling';
import type {ScalingMetric,View} from './lib/scaling';
import type {LadderCell,Method,Run,ScalingRecord} from './lib/types';
import ScalingPlot from './scaling-plot';
import type {FamilyTone} from './scaling-plot';
import {Flops,PointInspector} from './scaling-inspector';
import type {MapPoint} from './scaling-inspector';
import './scaling.css';

const TONE:Record<string,FamilyTone>={...TONES,baseline:{...TONES.baseline,name:'Dense'},moe:{name:'MoE',ink:'#806952',mid:'#BEA98F',soft:'#F7F3EE'}};
const tones=(family:string)=>TONE[family]??TONE.baseline;
const ORDER=['baseline','moe','looped','sparse','linear','residual','hybrid','others'];
const VIEW_LABELS:[string,string][]=[['angled','3D'],['parameters','Parameters'],['tokens','Tokens'],['floor','Floor plan']];
const TIER_NOTES:Record<string,string>={T3:'Fit five sizes per family.',T4:'Fit five sizes per family at the 49B protocol budget.',C200:'Test registered forecasts at the budget of the Dense 200B-token reference run.'};
const TASKS=[['Train a ladder cell','Claim an open cell and train it under the frozen recipe.','About 10²⁰ training FLOPs at T3'],['Add an architecture family','A method with a size rule; its cells join the ladder.','Configurations and cost proofs'],['Register a forecast','Predict the C200 optimum before the test runs start.','No training compute'],['Evaluate released checkpoints','Long-context and downstream evaluations on public weights.','Inference only'],['Improve the fits','Estimators and uncertainty for the compute-optimal frontier.','A laptop is enough'],['Reproduce a cell','Retrain a recorded cell with another seed, or re-run its checks.','One cell, or none']];
const STEPS=[['01','Claim','Open the claim issue for one cell; a maintainer marks it claimed on this board.'],['02','Prepare','Pinned ClimbMix revision, tokenizer and CORE bundle, and the frozen scaling recipe.'],['03','Configure','Write the cell configuration and check its training FLOPs against the tier budget.'],['04','Train','Train with the scaling trainer, on any hardware; resume as often as needed.'],['05','Export','nexttransformer.scaling.export checks the run and writes its record.'],['06','Pull request','The record, the checkpoint link and the W&B run. The board and the plot update on merge.']];


type RecordRow={id:string;name:string;stage:string;parameters:number;tokens:number;flops?:number;bpb:number;context:number;core:number;links:[string,string][]};

function Records({records,runs,methods,families}:{records:ScalingRecord[];runs:Run[];methods:Method[];families:string[]}){
 const run=runs.find(r=>r.methodId==='dense_full'),baseCore=run?.core.find(p=>p.step===FINAL_STEP)?.value;
 const matching=records.filter(r=>!families.length||families.includes(familyKey(r.family)));
 const rows:RecordRow[]=matching.map(r=>({id:r.id,name:r.name,stage:`Scaling · ${r.protocol}`,parameters:r.parameters.total,tokens:r.training.inputTokens,flops:r.training.trainingFlopsEstimate,bpb:r.finalBpb.val,context:r.model.sequenceLength,core:r.finalCore.value,
  links:[['Record',`${REPO_URL}/blob/main/results/scaling/runs/${r.id}.json`],...(r.links.wandb?[['W&B',r.links.wandb] as [string,string]]:[]),...(r.links.checkpoint?[['Checkpoint',r.links.checkpoint] as [string,string]]:[])]}));
 const references=scalingPoints(runs,methods,[]).filter(p=>families.length?families.includes(p.family):p.id===run?.id);
 for(const p of references){
  const source=runs.find(r=>r.id===p.id)!;
  rows.push({id:p.id,name:p.id===run?.id?'Dense Baseline':p.name,stage:'Leaderboard · 49B protocol',parameters:p.params,tokens:p.tokens,flops:p.flops,bpb:p.bpb,context:p.context,core:p.core,links:[['Audit',source.url]]});
 }
 return <div className="scaling-records"><div className="sc-section-head"><div><h2>Recorded runs</h2><span>{matching.length} scaling-stage {matching.length===1?'record':'records'}{families.length?` · ${references.length} matching 49B runs`:references.length?' · Dense Baseline shown for reference':''}</span></div><a href={REPO_URL+'/tree/main/results/scaling/runs'} target="_blank" rel="noreferrer">All records <ArrowUpRight size={14}/></a></div>
  {rows.length>0?<div className="records-scroll"><table className="records-table"><thead><tr><th>Run</th><th>Parameters</th><th>Training tokens</th><th>Training FLOPs</th><th>Validation BPB ↓</th><th>CORE ×100 ↑</th><th>Links</th></tr></thead>
   <tbody>{rows.map(r=><tr key={r.id}><td><b>{r.name}</b><small>{r.stage}</small></td><td data-label="Parameters">{compact(r.parameters)}</td><td data-label="Training tokens">{compact(r.tokens)}</td><td data-label="Training FLOPs">{r.flops?<Flops value={r.flops}/>:'—'}</td><td data-label="Validation BPB ↓">{r.bpb.toFixed(4)}<small>{r.context.toLocaleString('en-US')}-token context</small></td>
    <td data-label="CORE ×100 ↑">{(r.core*100).toFixed(2)}{baseCore!==undefined&&r.id!==run?.id&&<small className="gain">{r.core>=baseCore?'+':'−'}{Math.abs((r.core-baseCore)*100).toFixed(2)} vs the Dense Baseline</small>}</td>
    <td className="record-links"><div>{r.links.map(([label,href])=><a key={label} href={href} target="_blank" rel="noreferrer">{label}<ArrowUpRight size={12}/></a>)}</div></td></tr>)}</tbody></table></div>:<p className="sc-tier-note">No finished runs have been published for this selection.</p>}
  <p className="records-note">Validation BPB is measured at each run's own context, so compare it within a stage; CORE uses the same 22 tasks everywhere.</p>
 </div>;
}

function Cell({cell,status}:{cell:LadderCell;status:string}){
 const moe=cell.family==='moe';
 return <li className={'ladder-cell '+status} title={LADDER_STATUS[status]}>
  <b>{cell.layers?`L${cell.layers}`:'Optimum'}</b>
  <span>{cell.parameters?<>{compact(moe?cell.activeBlockParameters!:cell.blockParameters!)}{moe?' active block params':' block params'} · {compact(cell.tokens!)} tokens</>:'Size from the registered forecast'}</span>
  <em>{LADDER_STATUS[status]}{cell.claim&&<> · <a href={cell.claim.issue} target="_blank" rel="noreferrer">{cell.claim.by}</a></>}</em>
 </li>;
}

function Scaling({runs,methods}:{runs:Run[];methods:Method[]}){
 const [metric,setMetric]=useState<ScalingMetric>('core'),[view,setView]=useState<View>(VIEWS.angled),[planned,setPlanned]=useState(true),[picked,setPicked]=useState<string[]>([]),[multiple,setMultiple]=useState(false);
 const [selection,setSelection]=useState(''),[tierId,setTierId]=useState(scalingLadder?.tiers[0]?.id);
 const board=useRef<HTMLHeadingElement>(null);
 const ladder=scalingLadder,cells=ladder?.cells??[],tier=ladder?.tiers.find(t=>t.id===tierId);
 const all=scalingPoints(runs,methods,scalingRecords),rings=plannedPoints(ladder,scalingRecords);
 const families=ORDER.filter(f=>all.some(p=>p.family===f)||rings.some(p=>p.family===f));
 const matches=(f:string)=>!picked.length||picked.includes(f);
 const points=all.filter(p=>matches(p.family)),floor=rings.filter(p=>matches(p.family));
 const ladderFamilies=ladder?.families.filter(f=>matches(familyKey(f.id)))??[];
 const tierCells=cells.filter(c=>c.tier===tierId&&matches(familyKey(c.family)));
 const choices:MapPoint[]=[...points,...(planned?floor:[])];
 const selected=choices.find(p=>p.id===selection)??points.find(p=>p.stage==='scaling')??choices[0];
 const sourceRun=runs.find(r=>r.id===selected?.id),record=scalingRecords.find(r=>'scaling/'+r.id===selected?.id);
 const samples=(record??sourceRun)?.[metric==='core'?'core':'val']??[];
 const url=record?REPO_URL+'/blob/main/results/scaling/runs/'+record.id+'.json':sourceRun?.url??(selected?REPO_URL+'/blob/main/results/scaling/ladder.json':undefined);
 const preset=Object.entries(VIEWS).find(([,v])=>v.yaw===view.yaw&&v.pitch===view.pitch)?.[0]??'custom';
 function pick(f:string){setPicked(prev=>!multiple?[f]:prev.includes(f)?prev.filter(x=>x!==f):[...prev,f]);}
 function setCompare(enabled:boolean){setMultiple(enabled);if(!enabled)setPicked(prev=>prev.slice(0,1));}
 function jumpTier(id:string){setTierId(id);board.current?.focus({preventScroll:true});board.current?.scrollIntoView({block:'start'});}
 return <section className="scaling" id="scaling" aria-labelledby="sc-title">
  <header className="sc-intro"><div><p className="sc-kicker">04 / Scaling</p><h1 id="sc-title">Scaling at fixed compute</h1>
   <p>How should each architecture split its training budget between parameters and tokens?</p></div>
   {ladder&&<div className="sc-budget-nav"><p className="sc-kicker">Training budgets <span>/ FLOPs</span></p>
    <div>{ladder.tiers.map(t=><button key={t.id} onClick={()=>jumpTier(t.id)} aria-label={'Explore '+t.id+' budget'}><span><strong>{t.id}</strong> {t.role==='fit'?'Fit':'Test'}</span><b><Flops value={t.flops}/></b></button>)}</div></div>}
  </header>

  <section className="scaling-surface" aria-labelledby="sc-map-title">
   <div className="sc-map-head"><div className="sc-map-title"><span className="sc-figure-no">FIG. 01</span><h2 id="sc-map-title">Run map</h2></div>
    <div className="sc-controls">
     <div className="sc-switch" role="group" aria-label="Height metric">{(['core','bpb'] as const).map(m=><button key={m} aria-pressed={metric===m} onClick={()=>setMetric(m)}>{m==='core'?'CORE ↑':'Validation BPB ↓'}</button>)}</div>
     <label className="sc-view"><span>View</span><select aria-label="Viewpoint" value={preset} onChange={e=>setView(VIEWS[e.target.value])}>{preset==='custom'&&<option value="custom">Custom</option>}{VIEW_LABELS.map(([id,label])=><option key={id} value={id}>{label}</option>)}</select></label>
     <button className="sc-reset" title="Reset view" aria-label="Reset view" onClick={()=>setView(VIEWS.angled)}><RotateCcw size={15}/></button>
    </div></div>
   <div className="sc-family-bar"><span className="sc-kicker">Family</span>
    <div role="group" aria-label="Architecture family"><button aria-pressed={!picked.length} onClick={()=>setPicked([])}>All</button>{families.map(f=><button key={f} aria-pressed={picked.includes(f)} onClick={()=>pick(f)}><i style={{background:tones(f).ink}}/>{tones(f).name}</button>)}</div>
    <label><input type="checkbox" checked={multiple} onChange={e=>setCompare(e.target.checked)}/>Select multiple</label></div>
   <div className="sc-map-body"><div className="sc-stage">
    <div className="sc-keys"><span><i className="dot"/>49B protocol</span><span><i className="diamond"/>Scaling stage</span><label><input type="checkbox" checked={planned} onChange={e=>setPlanned(e.target.checked)}/><i className="ring"/>Planned cells</label><span className="sc-map-count" role="status">{points.length} measured · {planned?floor.length:0} planned</span></div>
    {choices.length?<ScalingPlot points={points} planned={floor} metric={metric} view={view} onView={setView} onReset={()=>setView(VIEWS.angled)} showPlanned={planned} tones={tones} selected={selected?.id} onSelect={setSelection}/>:
     <div className="sc-empty"><b>No points in view</b><p>No measured runs match this selection.</p>{floor.length>0&&!planned?<button onClick={()=>setPlanned(true)}>Show planned cells</button>:<button onClick={()=>setPicked([])}>Show all families</button>}</div>}
    <p className="sc-map-hint">Select a point for details. Drag or use arrow keys to rotate.</p>
   </div><PointInspector selected={selected} options={choices} onSelect={setSelection} metric={metric} samples={samples} url={url} family={selected?tones(selected.family).name:''}/></div>
  </section>
  <Records records={scalingRecords} runs={runs} methods={methods} families={picked}/>

  {ladder&&tier&&<section className="scaling-board"><div className="sc-section-head"><div><h2 tabIndex={-1} ref={board}>Budget ladder</h2><span>Fit each family at two smaller budgets, then test its forecast at C200.</span></div>
   <a href={docs('scaling/contribute/')}>How to claim a cell <ArrowUpRight size={14}/></a></div>
   {ladderFamilies.length>0?<><div className="sc-tier-head"><div className="sc-switch" role="group" aria-label="Budget tier">{ladder.tiers.map(t=><button key={t.id} aria-pressed={tierId===t.id} onClick={()=>setTierId(t.id)}>{t.id}</button>)}</div><b><Flops value={tier.flops}/> FLOPs</b><span>{tierCells.filter(c=>cellStatus(c,scalingRecords)==='recorded').length} / {tierCells.length} recorded</span></div>
   <p className="sc-tier-note">{TIER_NOTES[tier.id]} Token counts follow the theoretical training FLOPs of each size.</p>
   <div className="sc-ladder-grid">{ladderFamilies.map(f=><section className="sc-family" key={f.id} aria-label={f.name+' at '+tier.id}>
    <h3><i style={{background:tones(familyKey(f.id)).ink}}/>{f.name}<span>{tier.id}</span></h3>
    <ul>{cells.filter(c=>c.family===f.id&&c.tier===tier.id).map(c=><Cell key={c.id} cell={c} status={cellStatus(c,scalingRecords)}/>)}</ul>
    <details><summary>Family recipe{f.gate?' & gate':''}<ChevronDown size={14}/></summary><p>{f.rule}</p>{f.gate&&<p>{f.gate}</p>}</details></section>)}</div></>:<p className="sc-tier-note">No ladder cells are registered for the selected families yet.</p>}
  </section>}
  <div className="sc-resources"><div className="sc-section-head"><div><h2>Take part</h2><span>Claim a cell, register a forecast or evaluate a checkpoint.</span></div><a href={docs('scaling/overview/')}><BookOpen size={15}/> Scaling protocol <ArrowUpRight size={14}/></a></div>
   <div className="sc-guides"><details><summary>Ways to contribute <ChevronDown size={16}/></summary><ul>{TASKS.map(([title,text,cost])=><li key={title}><b>{title}</b><p>{text} <small>{cost}.</small></p></li>)}</ul></details>
   <details><summary>Train a cell, step by step <ChevronDown size={16}/></summary><ol>{STEPS.map(([n,title,text])=><li key={n}><b>{title}</b><p>{text}</p></li>)}</ol><a className="sc-link" href={docs('scaling/contribute/')}>Contribution guide <ArrowUpRight size={14}/></a></details></div>
  </div>
 </section>;
}

// The dashboard replay advances every frame. Only the actual run and method objects affect this view.
export default memo(Scaling,(a,b)=>a.methods===b.methods&&a.runs.length===b.runs.length&&a.runs.every((r,i)=>r===b.runs[i]));
