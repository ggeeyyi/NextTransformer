'use client';
import {useEffect,useLayoutEffect,useRef,useState} from 'react';
import {TONES} from './lib/palette';
import type {CSSProperties} from 'react';
import {ArrowDown,ArrowRight,ArrowUp,Pause,Play,RotateCcw} from 'lucide-react';
import {chartName} from './learning-chart';
import {FINAL_STEP,coreLengthGroups,leaderboardEntries,rankingCheckpoint,scoreAt} from './lib/leaderboard';
import type {RankingMetric,RankingMode} from './lib/leaderboard';
import {LEVELS,coreOnly,originOf,provenance} from './lib/arena';
import type {Origin} from './lib/arena';
import {arenaEntries} from './lib/results';
import coreLengths from '../../nexttransformer/evaluation/core-lengths.json';
import type {Method,Run} from './lib/types';
import type {MoeTrack} from './lib/moe';
import './leaderboard.css';
import {ComputeValue} from './training-compute';
import RankChart from './rank-chart';

const number=(value:number)=>new Intl.NumberFormat('en-US').format(Math.round(value));
const families:Record<string,{name:string;color:string}>=Object.fromEntries(Object.entries(TONES).map(([id,t])=>[id,{name:t.name,color:t.ink}]));

function AnimatedScore({value,decimals}:{value:number;decimals:number}){
 const [display,setDisplay]=useState(value),previous=useRef(value);
 useEffect(()=>{
  const from=previous.current;
  if(window.matchMedia('(prefers-reduced-motion: reduce)').matches){previous.current=value;setDisplay(value);return;}
  let frame:number,start:number|undefined;
  const tick=(time:number)=>{start??=time;const t=Math.min(1,(time-start)/500);const next=from+(value-from)*(1-(1-t)**3);previous.current=next;setDisplay(next);if(t<1)frame=requestAnimationFrame(tick);};
  frame=requestAnimationFrame(tick);return()=>cancelAnimationFrame(frame);
 },[value]);
 return <><span aria-hidden="true">{display.toFixed(decimals)}</span><span className="leaderboard-sr">{value.toFixed(decimals)}</span></>;
}

export default function Leaderboard<M extends Pick<Method,'id'|'name'|'family'>>({runs,methods,baseline,cursor,playing,onReplay,onPause,onCursor,onCompare,onModel,onDetails,moe}:{runs:Run[];methods:M[];baseline?:Run;moe?:MoeTrack;cursor:number;playing:boolean;onReplay:()=>void;onPause:()=>void;onCursor:(step:number)=>void;onCompare:(run:Run)=>void;onModel:(method:M)=>void;onDetails:(run:Run)=>void}){
 const [metric,setMetric]=useState<RankingMetric>('core'),[mode,setMode]=useState<RankingMode>('final'),[origin,setOrigin]=useState<Origin>('all');
 const wrapper=useRef<HTMLElement>(null),previousPositions=useRef(new Map<string,number>());
 const finalStep=moe?.updates ?? FINAL_STEP, coreLabel=moe?'MoE-CORE':'CORE';
 const rules={finalStep, separateFinalEvaluation:!!moe};
 const baselineLabel=moe?'MoE baseline':'Dense';
 const checkpoint=mode==='final'?finalStep:rankingCheckpoint(baseline,metric,cursor,runs,finalStep);
 // Ranks always come from every run; the provenance filter only hides rows. The Dense Baseline stays as the reference.
 const ranking=leaderboardEntries(runs,metric,mode,checkpoint,run=>!moe&&coreOnly(run.methodId,arenaEntries),rules),ranked=ranking.filter(e=>e.rank!==undefined);
 const entries=ranking.filter(e=>origin==='all'||e.run.methodId==='dense_full'||originOf(e.run.methodId,arenaEntries)===origin);
 const values=ranked.map(e=>e.value!),min=Math.min(...values),max=Math.max(...values);
 const orderKey=entries.map(e=>e.run.id).join('|');
 useLayoutEffect(()=>{
  const rows=wrapper.current?.querySelectorAll<HTMLElement>('.leaderboard-row');
  if(!rows)return;
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  rows.forEach(row=>{const key=row.dataset.run!,before=previousPositions.current.get(key),after=row.offsetTop;row.getAnimations().forEach(a=>a.cancel());if(!reduced&&before!==undefined&&before!==after){const distance=before-after,largeMove=Math.abs(distance)>row.offsetHeight*2;row.animate(largeMove?[{opacity:.25,transform:'translateY(8px)'},{opacity:1,transform:'translateY(0)'}]:[{transform:`translateY(${distance}px)`},{transform:'translateY(0)'}],{duration:largeMove?400:850,easing:'cubic-bezier(.22,1,.36,1)'});}previousPositions.current.set(key,after);});
 },[orderKey]);
 useEffect(()=>{
  if(!wrapper.current)return;
  let lastWidth=-1;
  const observer=new ResizeObserver(([entry])=>{if(entry.contentRect.width===lastWidth)return;lastWidth=entry.contentRect.width;wrapper.current?.querySelectorAll<HTMLElement>('.leaderboard-row').forEach(row=>{row.getAnimations().forEach(a=>a.cancel());previousPositions.current.set(row.dataset.run!,row.offsetTop);});});
  observer.observe(wrapper.current);return()=>observer.disconnect();
 },[]);
 const referenceScore=(kind:RankingMetric)=>baseline ? scoreAt(baseline,kind,mode,checkpoint,rules) : undefined;
 const label=(run:Run)=>chartName(methods.find(m=>m.id===run.methodId))||run.methodId;
 const lengthGroups=coreLengthGroups(entries.filter(e=>e.core!==undefined).map(e=>e.run),label);
 const longTasks=Object.keys(coreLengths.overTrainingContext);
 const primaryLabel=metric==='core'?coreLabel:'validation BPB';
 function replay(){setMode('replay');onReplay();}
 function renderScore(value:number|undefined,latest:Run['core'][number]|undefined,kind:RankingMetric,run:Run){
  const scale=kind==='core'?100:1,digits=kind==='core'?2:4,reference=referenceScore(kind),gap=value!==undefined&&reference!==undefined&&run.id!==baseline?.id?(value-reference)*scale:undefined;
  const even=gap!==undefined&&Math.abs(gap)<.5*10**-digits,better=gap!==undefined&&!even&&(kind==='core'?gap>0:gap<0);
  return value!==undefined?<><b><AnimatedScore value={value*scale} decimals={digits}/></b>{gap!==undefined&&<small className={'leaderboard-delta'+(even?'':better?' better':' worse')}>{(even?'±':gap>0?'+':'−')+Math.abs(gap).toFixed(digits)} vs {baselineLabel}</small>}{run.id===baseline?.id&&<small className="leaderboard-delta">Reference</small>}</>:<><b>—</b>{latest&&<small>Latest {(latest.value*scale).toFixed(digits)} · {number(latest.step)} steps</small>}</>;
 }
 return <section className="leaderboard" id="leaderboard" ref={wrapper} aria-label="Model leaderboard">
  <div className="leaderboard-heading"><div><p className="eyebrow">02 / LEADERBOARD</p><h2>Leaderboard.<span>{moe?moe.title:'Measured evidence. Every contender.'}</span></h2></div><div className="leaderboard-heading-meta"><span>{runs.length} {runs.length===1?'experiment':'experiments'}</span><span>{ranked.length} ranked {mode==='final'?'at the final step':checkpoint===undefined?'':'at step '+number(checkpoint)}</span></div></div>
  <div className="leaderboard-surface">
   <div className="leaderboard-toolbar"><div className="leaderboard-metric" role="group" aria-label="Leaderboard ranking metric"><button aria-pressed={metric==='core'} onClick={()=>setMetric('core')}>{coreLabel} <ArrowUp size={16}/></button><button aria-pressed={metric==='val'} onClick={()=>setMetric('val')}>{mode==='final'?'Final val BPB':'Val BPB'} <ArrowDown size={16}/></button></div><div className="leaderboard-view" role="group" aria-label="Leaderboard results view"><button aria-pressed={mode==='final'} onClick={()=>{setMode('final');onCursor(finalStep);}}>Final results</button><button aria-pressed={mode==='replay'} onClick={()=>{if(mode!=='replay')replay();}}>Training replay</button></div>{!moe&&<div className="leaderboard-view leaderboard-origin" role="group" aria-label="Leaderboard provenance">{([['all','All'],['human','Human-designed'],['agent','Auto-research']] as const).map(([id,name])=><button key={id} aria-pressed={origin===id} onClick={()=>setOrigin(id)}>{name}</button>)}</div>}</div>
   <div className="leaderboard-story"><div className="leaderboard-podium"><small>{mode==='final'?'LEADING AT THE FINAL CHECKPOINT':checkpoint===undefined?'WAITING FOR AN EVALUATION':'LEADING AT STEP '+number(checkpoint)}</small>{ranked.slice(0,3).map((e,i)=>{const pm=methods.find(x=>x.id===e.run.methodId);return <button key={e.run.id} className={'place-'+(i+1)} onClick={()=>pm&&onModel(pm)}><span>{e.rank}</span><b>{label(e.run)}</b><em><AnimatedScore value={e.value!*(metric==='core'?100:1)} decimals={metric==='core'?2:4}/> {metric==='core'?coreLabel:'BPB'}</em></button>;})}{!ranked.length&&<b>No shared checkpoint yet</b>}</div><button className="leaderboard-replay-button" aria-label={mode==='replay'&&playing?'Pause ranking replay':'Replay the ranking'} onClick={mode==='replay'&&playing?onPause:replay}>{mode==='replay'&&playing?<Pause size={17}/>:mode==='replay'?<RotateCcw size={17}/>:<Play size={17}/>}<span>{mode==='replay'&&playing?'Pause replay':mode==='replay'?'Replay again':'Replay the ranking'}</span></button></div>
   {mode==='replay'&&<div className="leaderboard-replay"><label htmlFor="leaderboard-replay-step">Training step <b>{number(cursor)}</b></label><input id="leaderboard-replay-step" aria-label="Leaderboard replay step" type="range" min={0} max={finalStep} value={Math.min(cursor,finalStep)} onChange={e=>onCursor(+e.target.value)}/><span>{checkpoint===undefined?'Awaiting first score':'Scores at '+number(checkpoint)}</span></div>}
   {mode==='replay'&&<RankChart runs={metric==='val'?runs.filter(r=>!coreOnly(r.methodId,arenaEntries)):runs} methods={methods} metric={metric} cursor={cursor} name={label} finalStep={finalStep} coreLabel={coreLabel} baselineId={baseline?.methodId}/>}
   <div className="leaderboard-columns" aria-hidden="true"><span>RANK</span><span>MODEL</span><span className={metric==='core'?'selected':''}>{mode==='final'?'FINAL ':''}{coreLabel.toUpperCase()} ↑</span><span className={metric==='val'?'selected':''}>{mode==='final'?'FINAL VAL BPB':'VAL BPB'} ↓</span><span>{moe?'REL. THROUGHPUT':<>AVG TRAIN<br/>GFLOPs / TOKEN</>}</span><span/></div>
   <div className="leaderboard-rows" role="list" aria-label={`Models ranked by ${primaryLabel}`}>
    {entries.map(entry=>{
     const {run,rank}=entry,m=methods.find(m=>m.id===run.methodId),family=moe&&m?.family==='moe'?{name:'MoE',color:'#526B85'}:families[m?.family??'baseline']??families.baseline,arena=moe?undefined:provenance(run.methodId,arenaEntries);
     const relative=moe?.models.find(model=>model.id===run.methodId)?.record?.throughput.harness?.relative_throughput;
     const observedStep=Math.max(run.step,...run.val.map(p=>p.step),...run.core.map(p=>p.step));
     const chip=entry.differentBudget?'Different budget':metric==='val'&&entry.coreOnly?'Ranked on CORE only':mode==='replay'&&rank===undefined?'No score at this step':observedStep<finalStep?`${number(observedStep)} / ${number(finalStep)}`:'';
     const merit=entry.value===undefined?0:max===min?1:metric==='core'?(entry.value-min)/(max-min):(max-entry.value)/(max-min);
     return <article role="listitem" key={run.id} data-run={run.id} className={'leaderboard-row'+(rank===1?' leading':'')+(rank===undefined?' pending':'')} style={{minHeight:'var(--leaderboard-row-height)','--leaderboard-accent':family.color} as CSSProperties} aria-label={`${rank===undefined?'Unranked':'Rank '+rank}, ${label(run)}${run.methodId==='dense_full'?', Dense Baseline':''}`}>
      <span className="leaderboard-rank">{rank===undefined?'—':String(rank).padStart(2,'0')}</span>
      <div className="leaderboard-model"><button onClick={()=>m&&onModel(m)}>{label(run)}<ArrowRight size={15}/></button><span><i style={{background:family.color}}/>{family.name}{arena&&<em className="leaderboard-provenance" title={`Arena entry · ${LEVELS[arena.researcher.involvement].name}: ${LEVELS[arena.researcher.involvement].note}`}>Auto-research · {arena.researcher.involvement}</em>}<button className="leaderboard-run-details" onClick={()=>onDetails(run)} aria-label={'Experiment details for '+label(run)}>Run details</button>{runs.filter(r=>r.methodId===run.methodId).length>1&&<small> · {run.name}</small>}{chip&&<em className="leaderboard-progress-chip">{chip}</em>}</span></div>
      <div className={'leaderboard-score core'+(metric==='core'?' active':'')}><span className="leaderboard-mobile-label">{mode==='final'?'Final ':''}{coreLabel} ↑</span>{renderScore(entry.core,entry.latestCore,'core',run)}{metric==='core'&&entry.value!==undefined&&<i className="leaderboard-merit" style={{width:(18+merit*72)+'%'}}/>}</div>
      <div className={'leaderboard-score val'+(metric==='val'?' active':'')}><span className="leaderboard-mobile-label">{mode==='final'?'Final val BPB':'Val BPB'} ↓</span>{renderScore(entry.val,entry.latestVal,'val',run)}{metric==='val'&&entry.value!==undefined&&<i className="leaderboard-merit" style={{width:(18+merit*72)+'%'}}/>}</div>
      <div className="leaderboard-compute"><span className="leaderboard-compute-label">{moe?'Relative throughput':'Avg train GFLOPs / token'}</span>{moe?<span className="compute-value"><b>{relative===undefined?'—':relative.toFixed(3)+'×'}</b><small>{relative===undefined?'No paired measurement':'Same-node harness'}</small></span>:<ComputeValue run={run} baseline={baseline} cursor={mode==='replay'?cursor:undefined}/>}</div>
      <button className="leaderboard-compare" onClick={()=>run.id===baseline?.id&&m?onModel(m):onCompare(run)} aria-label={run.id===baseline?.id?'Open '+label(run)+' model card':'Explore '+label(run)}>{run.id===baseline?.id?'Model card':'Compare'} <ArrowRight size={15}/></button>
     </article>;
    })}
   </div>
   {!entries.length&&<p className="leaderboard-empty">{origin==='agent'?'No auto-research entries yet. Arena entries appear here after review.':'No measured experiments yet. New results appear here after review.'}</p>}
   <div className="leaderboard-caption">{moe?<span>{mode==='final'?'Scores use the final 14-task evaluation and full six-source validation after '+number(finalStep)+' updates.':'Replay shows scheduled MoE-CORE and sampled validation at matching measured steps.'} Deltas use this budget’s MoE Baseline.</span>:<><span>{mode==='final'?<>Final rankings use measured results at step {number(FINAL_STEP)}. Final val BPB is the last periodic validation checkpoint.</>:<>Replay rankings use exactly the same evaluation step for every ranked model. Rows move as measured checkpoints arrive.</>}</span><span>Training compute is estimated per input token; configuration targets are not substituted. </span><span>{mode==='final'?'In-progress results stay visible and unranked.':'Missing evaluations stay unranked.'}</span><span>Auto-research entries rank on CORE only; their validation BPB is shown, not ranked.</span>
    <div className="leaderboard-core-context">
     <p>Only {longTasks.length} of {coreLengths.taskCount} CORE tasks contain prompts longer than the {number(coreLengths.trainingTokens)}-token training context ({new Intl.ListFormat('en').format(longTasks)}).</p>
     <p><strong>Methods evaluated within the training context length:</strong> {lengthGroups.within.join(', ') || 'None'}</p>
     <p><strong>Methods evaluated beyond the training context length:</strong> {lengthGroups.beyond.join(', ') || 'None'}</p>
    </div>
   </>}</div>
  </div>
 </section>;
}
