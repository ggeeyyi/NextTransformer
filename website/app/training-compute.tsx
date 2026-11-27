'use client';
import {getTrainingCompute,trainingComputeDelta,LEGACY_ACCOUNTING} from './lib/training-compute';
import {ledgerLabel} from './lib/labels';
import type {Run} from './lib/types';
const number=(v:number)=>v.toLocaleString('en-US');
export function ComputeValue({run, baseline, cursor, baselineLabel = 'Dense'}: {
 run?: Run; baseline?: Run; cursor?: number; baselineLabel?: string;
}){
 const c=getTrainingCompute(run),delta=trainingComputeDelta(run,baseline);
 if(!c)return <span className="compute-value missing"><b>—</b><small>Not reported</small></span>;
 if(cursor!==undefined&&c.step>cursor)return <span className="compute-value missing"><b>—</b><small>Later checkpoint</small></span>;
 return <span className="compute-value"><b>{c.averageGflopsPerInputToken.toFixed(6)}</b><small>{c.final?'Full run':'Through step '+number(c.step)}{delta!==undefined&&run?.id!==baseline?.id?' · '+(delta>=0?'+':'')+delta.toFixed(2)+'% / '+baselineLabel:''}</small></span>;
}
export default function ComputeDetails({run,baseline,target}:{run?:Run;baseline?:Run;target?:number}){
 const c=getTrainingCompute(run),delta=trainingComputeDelta(run,baseline);
 return <details className="compute-details"><summary>Training compute &amp; accounting</summary><p>Average estimated training work per input token, in GFLOPs. Values come from executed training paths.</p>{c?<><dl><div><dt>Average training cost</dt><dd>{c.averageGflopsPerInputToken.toFixed(6)} GFLOPs / input token{delta!==undefined&&run?.id!==baseline?.id?' · '+(delta>=0?'+':'')+delta.toFixed(2)+'% vs Dense':''}</dd></div><div><dt>Measurement</dt><dd>{c.final?'Full training run':'Training in progress'} · step {number(c.step)} · {number(c.inputTokens)} input tokens</dd></div><div><dt>Accounting</dt><dd>{ledgerLabel(c.accounting)}</dd></div><div><dt>Source</dt><dd>{c.source}</dd></div>{c.trainingFlopsEstimate!==undefined&&<div><dt>Cumulative FLOPs estimate</dt><dd>{c.trainingFlopsEstimate.toExponential(8)}</dd></div>}</dl><code>training_flops_estimate / input_tokens / 1e9</code>{c.accounting===LEGACY_ACCOUNTING&&<p>The historical matrix estimate includes 3 × forward FLOPs plus recompute FLOPs. Full-run costs vary across methods; these runs are not strictly compute-matched.</p>}</>:<p>A cumulative training estimate has not been reported for this experiment.</p>}{target!==undefined&&target>0&&<p>Configuration cost target: {target.toFixed(6)} GFLOPs / token. The completed training average is reported separately above.</p>}</details>;
}
