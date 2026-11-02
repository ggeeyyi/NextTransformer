'use client';
import {useEffect, useId, useRef, useState} from 'react';
import {Activity} from 'lucide-react';
import type {Method, Run, Point} from './lib/types';

export const SERIES_COLORS=['#35485F','#7356A6','#B37037','#397AB4','#A85B75','#597DA0','#8C6040','#7E6CAD','#BB617C','#4C779C','#997549'];
export const chartName=(m:Method|undefined)=>({sssl:'SSSL',dense_full:'Dense Baseline',ut_act:'UT · ACT',fixed_recursive:'Fixed Recursive',huginn:'Huginn',ouro:'Ouro · LoopLM',deeploop:'DeepLoop',dsa:'DSA',qsa:'QSA',nsa:'NSA',seer:'SeerAttention',moba:'MoBA',mor:'MoR',loopformer:'LoopFormer',circle:'Circle Loop',residual_standard:'Standard Residual',attnres:'Block AttnRes'} as Record<string,string>)[m?.id??'']??m?.name??'';
export const seriesColor=(id:string,methods:Method[]=[])=>({sssl:'#7E7462',dense_full:'#5E6166',ut_act:'#74638C',fixed_recursive:'#8C759F',huginn:'#9570A0',ouro:'#B07794',deeploop:'#6E7294',mor:'#88739D',loopformer:'#A48DB3',circle:'#AE748C',dsa:'#A96D7B',qsa:'#BC8A96',nsa:'#965C66',seer:'#AB7E87',moba:'#986A7D',deltanet:'#607E97',gla:'#6E8A9D',kda:'#526B85',retnet:'#8295AD',elu_linear:'#718A9E',elu_qknorm:'#9DB2C5',gdn:'#3E6A8A',mqa:'#A88C4F',gqa:'#926F23',mla_qknorm:'#695019',mla_nope_qknorm:'#C0AB7F',residual_standard:'#6D7591',attnres:'#2F3C62',gdn_hybrid:'#B56754',gdn_hybrid_swa:'#764337'} as Record<string,string>)[id]??SERIES_COLORS[Math.max(0,methods.findIndex(m=>m.id===id))%SERIES_COLORS.length];
const number=(v:number)=>new Intl.NumberFormat('en-US').format(Math.round(v));

export default function Chart({runs,methods,metric,cursor,max=46875,focus='',wide=false,compact=false,small=false,overview=false,domain,allRange=false,delta=false,allowNegative=false,gapFill=false,ariaLabel}:{runs:Run[];methods:Method[];metric:'val'|'core';cursor:number;max?:number;focus?:string;wide?:boolean;compact?:boolean;small?:boolean;overview?:boolean;domain?:[number,number];allRange?:boolean;delta?:boolean;allowNegative?:boolean;gapFill?:boolean;ariaLabel?:string}){
 const uid=useId().replace(/[^a-zA-Z0-9_-]/g,'');
 const wrap=useRef<HTMLDivElement>(null);
 const [width,setWidth]=useState(wide?1000:600),[hover,setHover]=useState<{step:number;y:number}|null>(null);
 useEffect(()=>{if(!wrap.current)return;const observer=new ResizeObserver(([e])=>setWidth(Math.max(220,e.contentRect.width)));observer.observe(wrap.current);return()=>observer.disconnect();},[]);
 const W=width,H=small?180:wide?(compact?(W<600?280:310):(W<600?310:370)):270,L=small?48:delta?66:58,R=small?16:28,T=small?14:22,B=small?34:45;
 max=Math.max(max,...runs.flatMap(r=>r[metric].map(p=>p.step)),1);
 const first=metric==='val'&&!allRange?Math.min(8000,max*.5):0;
 const value=(p:Point)=>p.value*(metric==='core'?100:1);
 const values=runs.flatMap(r=>r[metric].filter(p=>p.step>=first&&p.step<=max).map(value));
 const min=values.length?Math.min(...values):0,high=values.length?Math.max(...values):1;
 const pad=(high-min)*.15||(delta?.005:.02);
 const lo=domain?.[0]??(metric==='core'&&!allowNegative?Math.max(0,min-pad):min-pad),hi=domain?.[1]??(high+pad);
 const x=(step:number)=>L+(step-first)/(max-first)*(W-L-R), y=(v:number)=>H-B-(v-lo)/(hi-lo)*(H-T-B);
 const color=(r:Run)=>seriesColor(r.methodId,methods);
 const points=(r:Run)=>r[metric].filter(p=>p.step>=first&&p.step<=Math.min(cursor,max));
 const candidates=hover?runs.map(r=>({r,p:points(r).reduce<Point|undefined>((a,p)=>!a||Math.abs(p.step-hover.step)<Math.abs(a.step-hover.step)?p:a,undefined)})).filter(item=>item.p):[];
 const nearest=candidates.toSorted((a,b)=>Math.abs(y(value(a.p!))-hover!.y)-Math.abs(y(value(b.p!))-hover!.y))[0];
 const active=focus||(nearest&&Math.abs(y(value(nearest.p!))-hover!.y)<26?nearest.r.methodId:'');
 const tips=active?candidates.filter(({r})=>r.methodId===active||r.methodId==='dense_full'):[];
 const hasData=runs.some(r=>r[metric].length);
 const tickCount=small?2:W<500?3:5;
 return <div className={'plot-wrap'+(wide?' plot-wide':'')+(small?' plot-small':'')} ref={wrap}>
  <svg viewBox={`0 0 ${W} ${H}`} className="plot" role="img" aria-label={ariaLabel??(delta?'BPB difference from Dense at matching measured checkpoints':metric==='val'?'Validation BPB learning curves; lower is better':'CORE learning curves; higher is better')} onMouseMove={e=>{const r=e.currentTarget.getBoundingClientRect();setHover({step:Math.max(first,Math.min(max,first+((e.clientX-r.left)/r.width*W-L)/(W-L-R)*(max-first))),y:(e.clientY-r.top)/r.height*H});}} onMouseLeave={()=>setHover(null)}>
   <defs><clipPath id={uid+'-plot'}><rect x={L-7} y={T-7} width={W-L-R+14} height={H-T-B+14}/></clipPath></defs>
   {[0,1,2,3,4].map(i=>{const v=lo+(hi-lo)*i/4;return <g key={i}><line x1={L} x2={W-R} y1={y(v)} y2={y(v)} className="grid-line"/><text x={L-13} y={y(v)+4} textAnchor="end" className="axis-text">{delta?(v>0?'+':'')+v.toFixed(3):v.toFixed(metric==='core'?0:3)}</text></g>;})}
   {Array.from({length:tickCount+1},(_,i)=>first+(max-first)*i/tickCount).map(step=><text key={step} x={x(step)} y={H-17} textAnchor="middle" className="axis-text">{Number((step/1000).toFixed(1))}k</text>)}
   <g clipPath={`url(#${uid}-plot)`}>
    {delta&&lo<=0&&hi>=0&&<line x1={L} x2={W-R} y1={y(0)} y2={y(0)} stroke="#7F8DA1" strokeDasharray="5 6"/>}
    {gapFill&&(()=>{const dense=runs.find(r=>r.methodId==='dense_full'),run=runs.find(r=>r.methodId!=='dense_full');if(!dense||!run)return null;
     const at=new Map(points(dense).map(p=>[p.step,p])),shared=points(run).filter(p=>at.has(p.step));if(shared.length<2)return null;
     const d=shared.map((p,i)=>`${i?'L':'M'}${x(p.step)},${y(value(p))}`).join(' ')+' '+[...shared].reverse().map(p=>`L${x(p.step)},${y(value(at.get(p.step)!))}`).join(' ')+' Z';
     return <path d={d} fill={color(run)} opacity={.16}/>;})()}
    {[...runs].sort((a,b)=>Number(a.methodId===active)-Number(b.methodId===active)).map(r=>{const pts=points(r),last=pts.at(-1),emphasis=!active||r.methodId===active||r.methodId==='dense_full';return <g key={r.id} opacity={emphasis?1:.13} style={{transition:'opacity 160ms'}}>
     <path d={pts.map((p,i)=>`${i?'L':'M'}${x(p.step)},${y(value(p))}`).join(' ')} fill="none" stroke={color(r)} strokeWidth={r.methodId==='dense_full'?2:overview?(r.methodId===active?2.8:1.7):wide?2.8:2.3} strokeDasharray={r.methodId==='dense_full'?'7 6':undefined} strokeLinejoin="round" strokeLinecap="round"/>
     {metric==='core'&&pts.map(p=><circle key={p.step} cx={x(p.step)} cy={y(value(p))} r={overview?3:4} fill="white" stroke={color(r)} strokeWidth={2}/>)}
     {last&&<>{(!overview||r.methodId===active||r.methodId==='dense_full')&&<circle cx={x(last.step)} cy={y(value(last))} r={small?5:8} fill={color(r)} opacity={.1} className="head-pulse"/>}<circle cx={x(last.step)} cy={y(value(last))} r={overview&&r.methodId!==active?2.5:3.5} fill={color(r)}/></>}
    </g>;})}
    {cursor>=first&&cursor<=max&&<line x1={x(cursor)} x2={x(cursor)} y1={T} y2={H-B} className="cursor-line"/>}
    {hover&&<line x1={x(hover.step)} x2={x(hover.step)} y1={T} y2={H-B} stroke="#A7B6C8" strokeDasharray="3 5"/>}
   </g>
  </svg>
  {!hasData&&<div className="empty-chart"><Activity size={24}/><strong>Waiting for experiment records</strong><span>Submit measured checkpoints to add a curve.</span></div>}
  {tips.length>0&&<div className="chart-tooltip"><small>{delta?'Δ BPB vs Dense':'Measured checkpoint'}</small>{tips.map(({r,p})=><div key={r.id}><span style={{color:color(r)}}>● {chartName(methods.find(m=>m.id===r.methodId))}</span><b>{delta&&p!.value>0?'+':''}{value(p!).toFixed(metric==='core'?2:4)}</b><em>step {number(p!.step)}</em></div>)}</div>}
 </div>;
}
