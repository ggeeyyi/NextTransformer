'use client';
import {useEffect,useRef,useState} from 'react';
import {seriesColor} from './learning-chart';
import {rankSteps,rankTrajectories} from './lib/ranks';
import type {Method,Run} from './lib/types';

const short=(step:number)=>step>=46875?'46.9k':Math.round(step/1000)+'k';

// The rank race: each model's place at every exact evaluation step up to the replay cursor.
export default function RankChart({runs,methods,metric,cursor,name}:{runs:Run[];methods:Method[];metric:'core'|'val';cursor:number;name:(run:Run)=>string}){
 const wrap=useRef<HTMLDivElement>(null),[width,setWidth]=useState(900),[hover,setHover]=useState('');
 useEffect(()=>{if(!wrap.current)return;const observer=new ResizeObserver(([e])=>setWidth(Math.max(300,e.contentRect.width)));observer.observe(wrap.current);return()=>observer.disconnect();},[]);
 const steps=rankSteps(runs,metric),tracks=rankTrajectories(runs,metric,steps);
 const count=Math.max(2,...[...tracks.values()].flatMap(t=>t.map(p=>p.rank)));
 const narrow=width<560,W=width,row=narrow?14:15,L=26,R=narrow?92:150,T=10,B=24,H=T+B+(count-1)*row;
 const x=(step:number)=>L+steps.indexOf(step)/Math.max(1,steps.length-1)*(W-L-R),y=(rank:number)=>T+(rank-1)*row;
 const shown=steps.filter(s=>s<=cursor),current=shown.at(-1);
 if(steps.length<2)return null;
 return <div className="rank-chart" ref={wrap}>
  <div className="rank-chart-heading"><span><b>The rank race</b> · places by {metric==='core'?'CORE':'validation BPB'} at each exact evaluation step</span><span>{current===undefined?'Waiting for the first evaluation':'Through step '+current.toLocaleString('en-US')}</span></div>
  <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Rank of each model by ${metric==='core'?'CORE':'validation BPB'} at every evaluation step through the replay cursor`} onMouseLeave={()=>setHover('')}>
   {steps.map(s=><g key={s}><line x1={x(s)} x2={x(s)} y1={T-4} y2={H-B+4} className={'rank-step'+(s===current?' current':'')}/><text x={x(s)} y={H-6} textAnchor="middle" className="rank-axis">{short(s)}</text></g>)}
   {[1,count].map(r=><text key={r} x={L-10} y={y(r)+4} textAnchor="end" className="rank-axis">{r}</text>)}
   {runs.map(run=>{const pts=(tracks.get(run.id)??[]).filter(p=>p.step<=cursor);if(!pts.length)return null;const color=seriesColor(run.methodId,methods),last=pts.at(-1)!,dim=hover&&hover!==run.id;
    return <g key={run.id} opacity={dim?.18:1} onMouseEnter={()=>setHover(run.id)} style={{cursor:'default'}}>
     <path d={pts.map((p,i)=>`${i?'L':'M'}${x(p.step)},${y(p.rank)}`).join(' ')} stroke={color} strokeDasharray={run.methodId==='dense_full'?'6 5':undefined}/>
     {pts.map(p=><circle key={p.step} cx={x(p.step)} cy={y(p.rank)} r={p.step===current?4:2.6} fill={p.step===current?color:'#FFFEFD'} stroke={color} strokeWidth="1.6"/>)}
     {last.step===current&&<text x={x(last.step)+9} y={y(last.rank)+4} className="rank-label" fill={color}>{last.rank}. {name(run).split(' · ').at(-1)}</text>}
    </g>;})}
  </svg>
 </div>;
}
