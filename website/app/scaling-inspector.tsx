import {ArrowUpRight} from 'lucide-react';
import type {Point} from './lib/types';
import {compact,LADDER_STATUS} from './lib/scaling';
import type {PlannedPoint,ScalingMetric,ScalingPoint} from './lib/scaling';

export type MapPoint=ScalingPoint|PlannedPoint;
export const pointName=(p:MapPoint)=>'name' in p?p.name:p.id;
export function Flops({value}:{value:number}){
 const exponent=Math.floor(Math.log10(value));
 return <>{Number((value/10**exponent).toPrecision(3))} × 10<sup>{exponent}</sup></>;
}

function Trace({samples,metric}:{samples:Point[];metric:ScalingMetric}){
 if(samples.length<2)return null;
 const lo=Math.min(...samples.map(p=>p.value)),hi=Math.max(...samples.map(p=>p.value));
 const start=samples[0].step,end=samples[samples.length-1].step;
 const coords=samples.map(p=>[3+204*(p.step-start)/(end-start||1),4+35*(hi-p.value)/(hi-lo||1)]);
 const label=metric==='core'?'CORE':'Validation BPB';
 return <div className="sc-trace"><span>{label} over training</span>
  <svg viewBox="0 0 210 44" preserveAspectRatio="none" role="img" aria-label={label+' across '+samples.length+' measured checkpoints'}>
   <path d="M3 39H207 M3 21H207 M3 4H207" fill="none" stroke="#DEE5EE" strokeDasharray="2 4"/>
   <polyline points={coords.map(p=>p.join(',')).join(' ')} fill="none" stroke="currentColor" strokeWidth="1.7" vectorEffect="non-scaling-stroke"/>
   {coords.filter((_,i)=>i===0||i===coords.length-1||coords.length<=12).map(([x,y],i)=><circle key={i} cx={x} cy={y} r="2.3" fill="currentColor"/>)}
  </svg><div><small>Step {start.toLocaleString('en-US')}</small><small>{end.toLocaleString('en-US')}</small></div>
 </div>;
}

export function PointInspector({selected,options,onSelect,metric,samples,url,family}:{selected?:MapPoint;options:MapPoint[];onSelect:(id:string)=>void;metric:ScalingMetric;samples:Point[];url?:string;family:string}){
 const run=selected&&'stage' in selected?selected:undefined;
 const plan=selected&&'tier' in selected?selected:undefined;
 const groups=[['Scaling stage',options.filter(p=>'stage' in p&&p.stage==='scaling')],['49B protocol',options.filter(p=>'stage' in p&&p.stage==='reference')],['Planned cells',options.filter(p=>'tier' in p)]] as const;
 return <aside className="sc-inspector" aria-label="Point details">
  <label className="sc-kicker" htmlFor="sc-point">Point details</label>
  {selected?<><select id="sc-point" value={selected.id} onChange={e=>onSelect(e.target.value)}>{groups.map(([label,items])=>items.length>0&&<optgroup key={label} label={label}>{items.map(p=><option key={p.id} value={p.id}>{pointName(p)}</option>)}</optgroup>)}</select>
    <p className="sc-provenance">{family} <span>·</span> {run?(run.stage==='scaling'?'Scaling stage':'49B protocol'):`Planned · ${plan?.tier}`}</p>
    {run?<div className="sc-readings"><div><span>CORE ×100 ↑</span><b>{(run.core*100).toFixed(2)}</b></div><div><span>Validation BPB ↓</span><b>{run.bpb.toFixed(4)}</b></div></div>
    :<p className="sc-plan-note"><strong>{plan&&LADDER_STATUS[plan.status]}</strong>This cell has no measured scores yet.</p>}
    <dl className="sc-specs"><div><dt>Total parameters</dt><dd>{compact(selected.params)}</dd></div>
     {selected.active!==undefined&&<div><dt>{plan?'Active block parameters':'Block parameters'}</dt><dd>{compact(selected.active)}</dd></div>}
     <div><dt>Training tokens</dt><dd>{compact(selected.tokens)}</dd></div>
     {run&&<><div><dt>Context length</dt><dd>{run.context.toLocaleString('en-US')}</dd></div><div><dt>Training FLOPs</dt><dd>{run.flops?<Flops value={run.flops}/>: '—'}</dd></div></>}
    </dl>
    {run&&<Trace samples={samples} metric={metric}/>}
    {url&&<a className="sc-link" href={url} target="_blank" rel="noreferrer">{plan?'View ladder entry':'View run record'} <ArrowUpRight size={14}/></a>}
   </>:<p className="sc-plan-note">Enable a family or show planned cells to inspect a point.</p>}
 </aside>;
}
