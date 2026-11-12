'use client';
import {useEffect,useRef,useState} from 'react';
import type {KeyboardEvent,PointerEvent} from 'react';
import {compact,linearExtent,logExtent,logTicks,project} from './lib/scaling';
import type {PlannedPoint,ScalingMetric,ScalingPoint,View} from './lib/scaling';

export type FamilyTone={name:string;ink:string;mid:string;soft:string};
const clamp=(v:number,lo:number,hi:number)=>Math.max(lo,Math.min(hi,v));
export default function ScalingPlot({points,planned,metric,view,onView,onReset,showPlanned,tones,selected,onSelect}:{points:ScalingPoint[];planned:PlannedPoint[];metric:ScalingMetric;view:View;onView:(view:View)=>void;onReset:()=>void;showPlanned:boolean;tones:(family:string)=>FamilyTone;selected?:string;onSelect:(id:string)=>void}){
 const wrap=useRef<HTMLDivElement>(null),drag=useRef<{x:number;y:number;view:View;moved:boolean}|null>(null);
 const [width,setWidth]=useState(900),[hover,setHover]=useState('');
 useEffect(()=>{if(!wrap.current)return;const observer=new ResizeObserver(([e])=>setWidth(Math.max(200,e.contentRect.width)));observer.observe(wrap.current);return()=>observer.disconnect();},[]);
 const W=width,H=Math.round(clamp(W*.5,245,382)),M={l:W<560?43:65,r:W<560?34:64,t:33,b:W<560?42:48};
 const value=(p:ScalingPoint)=>metric==='core'?p.core*100:p.bpb;
 const floor=showPlanned?planned:[];
 const xd=logExtent([...points,...floor].map(p=>p.params)),yd=logExtent([...points,...floor].map(p=>p.tokens)),zd=linearExtent(points.map(value));
 const nx=(v:number)=>(Math.log10(v)-xd[0])/(xd[1]-xd[0])*2-1,ny=(v:number)=>1-(Math.log10(v)-yd[0])/(yd[1]-yd[0])*2,nz=(v:number)=>(v-zd[0])/(zd[1]-zd[0])*2-1;
 const corners=[-1,1].flatMap(x=>[-1,1].flatMap(y=>[-1,1].map(z=>project(x,y,z,view))));
 const [x0,x1]=[Math.min(...corners.map(c=>c.x)),Math.max(...corners.map(c=>c.x))],[y0,y1]=[Math.min(...corners.map(c=>c.y)),Math.max(...corners.map(c=>c.y))];
 const k=Math.min((W-M.l-M.r)/(x1-x0||1),(H-M.t-M.b)/(y1-y0||1));
 const kx=Math.min((W-M.l-M.r)/(x1-x0||1),k*1.96);
 const ox=M.l+(W-M.l-M.r-(x1-x0)*kx)/2-x0*kx,oy=M.t+(H-M.t-M.b-(y1-y0)*k)/2-y0*k;
 const at=(x:number,y:number,z:number)=>{const p=project(x,y,z,view);return {x:ox+p.x*kx,y:oy+p.y*k,depth:p.depth};};
 const line=(a:number[],b:number[])=>{const p=at(a[0],a[1],a[2]),q=at(b[0],b[1],b[2]);return {x1:p.x,y1:p.y,x2:q.x,y2:q.y};};
 const length=(a:number[],b:number[])=>{const l=line(a,b);return Math.hypot(l.x2-l.x1,l.y2-l.y1);};
 // The walls farthest from the viewer carry the grid; tick labels sit on the floor edges nearest to it.
 const xs=at(1,0,0).depth<at(-1,0,0).depth?1:-1,ys=at(0,1,0).depth<at(0,-1,0).depth?1:-1;
 const xTicks=logTicks(xd),yTicks=logTicks(yd),zTicks=Array.from({length:5},(_,i)=>zd[0]+(zd[1]-zd[0])*i/4);
 const showX=length([-1,-ys,-1],[1,-ys,-1])>60,showY=length([-xs,-1,-1],[-xs,1,-1])>60,showZ=length([xs,ys,-1],[xs,ys,1])>60;
 const edges=[[xs,-ys],[-xs,ys],[xs,ys]].map(([x,y])=>({x,y,screen:at(x,y,0).x}));
 const zEdge=edges.reduce((a,b)=>b.screen<a.screen?b:a);
 // Offset labels in screen pixels, not world coordinates: edge-on views otherwise stack
 // the title on its ticks. Cull crowded ticks on the projected axes as the view rotates.
 type XY={x:number;y:number};
 const center=at(0,0,-1);
 function outward(a:XY,b:XY){
  const dx=b.x-a.x,dy=b.y-a.y,len=Math.hypot(dx,dy)||1;
  let x=-dy/len,y=dx/len;
  const dot=x*((a.x+b.x)/2-center.x)+y*((a.y+b.y)/2-center.y);
  if(dot<-.01||(Math.abs(dot)<=.01&&y<0)){x=-x;y=-y;}return {x,y};
 }
 const offset=(pt:XY,n:XY,d:number)=>({x:pt.x+n.x*d,y:pt.y+n.y*d});
 const xNormal=outward(at(-1,-ys,-1),at(1,-ys,-1)),yNormal=outward(at(-xs,-1,-1),at(-xs,1,-1));
 const labels:{text:string;x:number;y:number;half:number;title:boolean}[]=[];
 function label(text:string,pos:XY,title=false){
  const half=text.length*(title?3.7:3.4)+3;
  const x=clamp(pos.x,half+2,W-half-2),y=clamp(pos.y,10,H-9);
  if(!title&&labels.some(l=>Math.abs(l.x-x)<l.half+half+3&&Math.abs(l.y-y)<14))return;
  labels.push({text,x,y,half,title});
 }
 if(showX)label('Parameters',offset(at(0,-ys,-1),xNormal,31),true);
 if(showY)label('Training tokens',offset(at(-xs,0,-1),yNormal,33),true);
 if(showZ)label(metric==='core'?'CORE ×100 ↑':'Validation BPB ↓',offset(at(zEdge.x,zEdge.y,1),{x:0,y:-1},17),true);
 if(showZ)for(const t of zTicks){const txt=metric==='core'?t.toFixed(1):t.toFixed(3);label(txt,offset(at(zEdge.x,zEdge.y,nz(t)),{x:-1,y:0},txt.length*3.4+7));}
 if(showX)for(const t of xTicks)label(compact(t),offset(at(nx(t),-ys,-1),xNormal,13));
 if(showY)for(const t of yTicks)label(compact(t),offset(at(-xs,ny(t),-1),yNormal,13));
 const measured=points.map(p=>({p,top:at(nx(p.params),ny(p.tokens),nz(value(p))),base:at(nx(p.params),ny(p.tokens),-1)})).sort((a,b)=>a.top.depth-b.top.depth);
 const rings=floor.map(p=>({p,base:at(nx(p.params),ny(p.tokens),-1)}));
 const tilted=view.pitch>.06;
 const active=measured.find(m=>m.p.id===selected),ring=rings.find(r=>r.p.id===selected);
 function locate(e:PointerEvent<SVGSVGElement>){const r=e.currentTarget.getBoundingClientRect();return {x:(e.clientX-r.left)/r.width*W,y:(e.clientY-r.top)/r.height*H};}
 function nearest(e:PointerEvent<SVGSVGElement>){
  const m=locate(e),targets=[...measured.slice().reverse().map(x=>({id:x.p.id,...x.top})),...rings.map(x=>({id:x.p.id,...x.base}))];
  return targets.reduce<{id:string;d:number}|null>((best,t)=>{const d=Math.hypot(t.x-m.x,t.y-m.y);return d<15&&(!best||d<best.d)?{id:t.id,d}:best;},null)?.id;
 }
 function move(e:PointerEvent<SVGSVGElement>){
  const d=drag.current;
  if(d){const dx=e.clientX-d.x,dy=e.clientY-d.y;if(Math.hypot(dx,dy)>4)d.moved=true;
   if(d.moved){setHover('');onView({yaw:d.view.yaw-dx*.008,pitch:clamp(d.view.pitch+dy*.008,0,Math.PI/2)});}return;}
  setHover(nearest(e)??'');
 }
 function key(e:KeyboardEvent<SVGSVGElement>){
  const step={ArrowLeft:[.08,0],ArrowRight:[-.08,0],ArrowUp:[0,-.08],ArrowDown:[0,.08]}[e.key];
  if(step){e.preventDefault();onView({yaw:view.yaw+step[0],pitch:clamp(view.pitch+step[1],0,Math.PI/2)});}
  if(e.key==='0'||e.key==='Home'){e.preventDefault();onReset();}
 }
 const mark=active?.top??ring?.base;
 const markLabel=active?.p.name??ring?.p.id??'';
 const tagText=markLabel.length>27?markLabel.slice(0,26)+'…':markLabel;
 const tagWidth=tagText.length*7.3+14;
 const tagX=mark?clamp(mark.x+20+tagWidth<=W?mark.x+16:mark.x-tagWidth-16,4,W-tagWidth-4):0;
 const tagY=mark?clamp(mark.y-20,3,H-24):0;
 return <div className="scaling-plot" ref={wrap}>
  <svg viewBox={`0 0 ${W} ${H}`} role="img" tabIndex={0} aria-label={`Finished runs by parameters, training tokens and ${metric==='core'?'CORE':'validation BPB'}; drag or use the arrow keys to rotate`}
   onPointerDown={e=>{if(!e.isPrimary||e.button!==0)return;e.currentTarget.setPointerCapture(e.pointerId);drag.current={x:e.clientX,y:e.clientY,view,moved:false};}}
   onPointerMove={move} onPointerUp={e=>{if(!e.isPrimary)return;if(drag.current&&!drag.current.moved){const id=nearest(e);if(id)onSelect(id);}drag.current=null;}} onLostPointerCapture={()=>{drag.current=null;}} onPointerCancel={()=>{drag.current=null;setHover('');}} onPointerLeave={()=>{if(!drag.current)setHover('');}}
   onDoubleClick={onReset} onKeyDown={key}>
   <polygon className="scaling-floor" points={[[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y])=>{const p=at(x,y,-1);return `${p.x},${p.y}`;}).join(' ')}/>
   {xTicks.map(t=><line key={'fx'+t} className="scaling-grid" {...line([nx(t),-1,-1],[nx(t),1,-1])}/>)}
   {yTicks.map(t=><line key={'fy'+t} className="scaling-grid" {...line([-1,ny(t),-1],[1,ny(t),-1])}/>)}
   {view.pitch<1.5?zTicks.map(t=><g key={'z'+t}><line className="scaling-grid" {...line([xs,-1,nz(t)],[xs,1,nz(t)])}/><line className="scaling-grid" {...line([-1,ys,nz(t)],[1,ys,nz(t)])}/></g>):null}
   {[[xs,ys],[xs,-ys],[-xs,ys]].map(([x,y])=><line key={x+','+y} className="scaling-edge" {...line([x,y,-1],[x,y,1])}/>)}
   {labels.map((l,i)=><text key={i} x={l.x} y={l.y} textAnchor="middle" dominantBaseline="central" className={l.title?'scaling-axis-title':'axis-text'}>{l.text}</text>)}
   {rings.map(({p,base})=>{const t=tones(p.family);return <circle key={p.id} cx={base.x} cy={base.y} r={hover===p.id?6:4.2} data-point={p.id} className="scaling-planned" stroke={t.ink} fill={hover===p.id?t.soft:'#FFFEFD'}/>;})}
   {measured.map(({p,top,base})=>{const t=tones(p.family),on=hover===p.id,dim=hover&&!on;return <g key={p.id} data-point={p.id} opacity={dim?.68:1} style={{transition:'opacity 140ms'}}>
    {tilted&&<><line x1={base.x} y1={base.y} x2={top.x} y2={top.y} stroke={t.mid} strokeDasharray="2 4"/><circle cx={base.x} cy={base.y} r={2.2} fill={t.mid} opacity={.55}/></>}
    {p.stage==='scaling'?<rect x={top.x-(on?7:5.6)} y={top.y-(on?7:5.6)} width={on?14:11.2} height={on?14:11.2} transform={`rotate(45 ${top.x} ${top.y})`} fill={t.ink} stroke="#FFFEFD" strokeWidth={1.6}/>:<circle cx={top.x} cy={top.y} r={on?7.5:5.4} fill={t.ink} stroke="#FFFEFD" strokeWidth={1.6}/>}
   </g>;})}
   {mark&&<g className="sc-selected-mark" pointerEvents="none" aria-hidden="true">
    <circle cx={mark.x} cy={mark.y} r={10} fill="none" stroke="#526B85" strokeWidth="1.4"/>
    <line x1={mark.x+(tagX>mark.x?8:-8)} y1={mark.y-5} x2={tagX>mark.x?tagX:tagX+tagWidth} y2={tagY+10} stroke="#8396AA"/>
    <rect x={tagX} y={tagY} width={tagWidth} height={21} rx="5" fill="#F3F6FA" stroke="#C8D4E0"/>
    <text x={tagX+tagWidth/2} y={tagY+14} textAnchor="middle" fill="#415873" fontSize="11.5" fontWeight="600">{tagText}</text>
   </g>}
  </svg>
 </div>;
}
