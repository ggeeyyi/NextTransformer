import {z} from 'zod';
import type {Run,TrainingCompute} from './types';
export const LEGACY_ACCOUNTING='Historical matrix FLOPs · 3 × forward + recompute';
export function getTrainingCompute(run:Run|undefined):TrainingCompute|undefined {
 const c=run?.trainingCompute;
 return c&&Number.isFinite(c.averageGflopsPerInputToken)&&c.averageGflopsPerInputToken>0&&Number.isFinite(c.inputTokens)&&c.inputTokens>0&&Number.isInteger(c.step)&&c.step>=0&&c.accounting?c:undefined;
}
/** Cost deltas require the same completed budget AND accounting definition. */
export function trainingComputeDelta(run:Run|undefined,baseline:Run|undefined):number|undefined {
 const a=getTrainingCompute(run),b=getTrainingCompute(baseline);
 if(!a||!b||!a.final||!b.final||a.accounting!==b.accounting||a.step!==b.step||a.inputTokens!==b.inputTokens)return undefined;
 return (a.averageGflopsPerInputToken/b.averageGflopsPerInputToken-1)*100;
}
/** Only explicitly reported cumulative fields; configuration cost proofs are not averages. */
export function computeFromSummary(summary:Record<string,any>,step:number,final:boolean,accounting:string,source:string):TrainingCompute|undefined {
 // Native tracker prefixes event fields with train/; JSON exports use bare counters.
 // Read each pair together so counters from different event scopes cannot be mixed.
 const pair=[['train/training_flops_estimate','train/input_tokens'],['training_flops_estimate','input_tokens']].find(([f,t])=>Number.isFinite(summary[f])&&summary[f]>0&&Number.isFinite(summary[t])&&summary[t]>0);
 if(!pair)return undefined;
 const [flops,tokens]=pair.map(key=>summary[key]);
 if(!Number.isFinite(flops)||flops<=0||!Number.isFinite(tokens)||tokens<=0||!Number.isInteger(step)||step<0)return undefined;
 return {averageGflopsPerInputToken:flops/tokens/1e9,trainingFlopsEstimate:flops,inputTokens:tokens,step,final,accounting,source};
}

// Validates a reported training-compute record.
const point=z.object({step:z.number().int().min(0).max(10000000),value:z.number().finite().min(0).max(100)});
const link=z.string().url().max(1000).refine(s=>/^https:\/\//.test(s),'Use an HTTPS URL');
export const trainingComputeSchema=z.object({
 averageGflopsPerInputToken:z.number().finite().positive().optional(),
 trainingFlopsEstimate:z.number().finite().positive().optional(),inputTokens:z.number().finite().positive(),
 step:z.number().int().min(0).max(10000000),final:z.boolean().default(false),
 accounting:z.string().trim().min(3).max(200),source:z.string().trim().min(3).max(300).default('Contributor-reported training logs'),
}).superRefine((d,c)=>{
 if(d.averageGflopsPerInputToken===undefined&&d.trainingFlopsEstimate===undefined)c.addIssue({code:'custom',message:'Report an average or a cumulative FLOPs estimate'});
 if(d.averageGflopsPerInputToken!==undefined&&d.trainingFlopsEstimate!==undefined){const expected=d.trainingFlopsEstimate/d.inputTokens/1e9;if(Math.abs(expected-d.averageGflopsPerInputToken)>Math.max(5e-7,expected*1e-6))c.addIssue({code:'custom',message:'Average compute does not match cumulative FLOPs / input tokens / 1e9'});}
}).transform(d=>({...d,averageGflopsPerInputToken:d.trainingFlopsEstimate!==undefined?d.trainingFlopsEstimate/d.inputTokens/1e9:d.averageGflopsPerInputToken!}));
export const submissionSchema=z.object({
 kind:z.enum(['model','result']),parameters:z.number().int().min(0).max(1e14).default(0),physicalBlocks:z.number().int().min(0).max(10000).default(0),title:z.string().trim().min(2).max(160),author:z.string().trim().min(1).max(100),
 family:z.enum(['looped','sparse','linear']),methodId:z.string().max(100),description:z.string().trim().min(10).max(4000),
 paperUrl:link.or(z.literal('')),codeUrl:link.or(z.literal('')),evidenceUrl:link.or(z.literal('')),
 commit:z.string().max(100),hardware:z.string().max(200),cohort:z.enum(['reference-v1','native-v3','external']),
 tokens:z.number().finite().min(0).max(1e15),flops:z.number().finite().min(0).max(1e30),seed:z.number().int().min(0),
 trainingCompute:trainingComputeSchema.optional(),
 val:z.array(point).max(10000),core:z.array(point).max(10000),
 architecture:z.object({kind:z.enum(['loop','sparse','state','dense','custom']),nodes:z.array(z.string().trim().min(1).max(40)).min(2).max(8),loop:z.boolean()}),
 figureUrl:z.string().max(300).refine(s=>s===''||/^\/api\/figures\/[a-f0-9-]+$/.test(s)),
}).superRefine((d,c)=>{
 if(d.trainingCompute&&!d.evidenceUrl)c.addIssue({code:'custom',message:'Training compute needs an evidence URL'});
 if(d.kind==='model'&&!d.paperUrl&&!d.codeUrl)c.addIssue({code:'custom',message:'A model needs a paper or implementation URL'});
 if((d.kind==='result'||d.val.length||d.core.length)&&(!d.evidenceUrl||(!d.val.length&&!d.core.length)))c.addIssue({code:'custom',message:'Results need an evidence URL and at least one measured checkpoint'});
 for(const key of ['val','core'] as const){const a=d[key];for(let i=1;i<a.length;i++)if(a[i].step<=a[i-1].step)c.addIssue({code:'custom',message:'Checkpoint steps must be strictly increasing'});}
 if(d.core.some(p=>p.value>1))c.addIssue({code:'custom',message:'Use the original 0–1 CORE scale; the chart displays ×100'});
});
