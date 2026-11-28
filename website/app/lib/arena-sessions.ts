import type {ArenaEntry,Run} from './types';

// One autonomous research session in the research gym, exported from its retained state: results/arena/sessions/.
export type SessionCandidate={id:string;method:string;title:string;createdAt:string;parents:string[];layers:number|null;mlpWidth:number|null;parameters:number|null;anchorRatio:number|null;publishedMethod:boolean;trained:boolean;submitted:boolean;selected:boolean;control:boolean;
 basis:{candidate:string;changes:Record<string,unknown>;codeChanged:boolean}|null;hypothesis:string};
export type SessionRun={candidate:string;from:number;to:number;state:string;reachedUpdate:number|null;queuedAt:string;startedAt:string|null;endedAt:string;trainingFlops:number;devBpb:number|null;trainingLoss:[number,number][]};
export type ArenaSession={id:string;order:number;
 agent:{name:string;model:string;effort:string|null;harness:string;harnessVersion:string|null};
 gym:{name:string;commit:string;url:string;instruction:string};hardware:string;start:string;end:string;submittedAt:string;humanIntervention:string;
 budget:{limitFlops:number;spentFlops:number};tokens:{input:number;cachedInput:number;output:number};jobs:Record<string,number>;
 candidatesRegistered:number;candidatesTrained:number;
 submission:{candidate:string;selected:string|null;method:string;arenaEntry:string|null};
 candidates:SessionCandidate[];runs:SessionRun[];spend:[string,number][]};

export type Role='selected'|'control'|'candidate';
export type DevPoint={update:number;bpb:number;at:string};
export type Standing={candidate:SessionCandidate;bpb:number;delta?:number;rank:number};

const time=(iso:string)=>Date.parse(iso);
export const hoursBetween=(a:string,b:string)=>(time(b)-time(a))/3.6e6;
export const budgetShare=(s:ArenaSession)=>s.budget.spentFlops/s.budget.limitFlops;
export const roleOf=(c:SessionCandidate):Role=>c.selected?'selected':c.control?'control':'candidate';
export const trainedCandidates=(s:ArenaSession)=>s.candidates.filter(c=>c.trained);

/** A candidate's measured dev BPB, one point per training job that returned a result, in update order. */
export function devPath(s:ArenaSession,id:string):DevPoint[]{
 return s.runs.filter(r=>r.candidate===id&&r.devBpb!==null).map(r=>({update:r.to,bpb:r.devBpb!,at:r.endedAt})).sort((a,b)=>a.update-b.update);
}

/** Updates at which at least two candidates were measured: the steps the agent compared designs at. */
export function screeningSteps(s:ArenaSession):number[]{
 const seen=new Map<number,Set<string>>();
 for(const r of s.runs)if(r.devBpb!==null){const ids=seen.get(r.to)??new Set<string>();ids.add(r.candidate);seen.set(r.to,ids);}
 return [...seen].filter(([,ids])=>ids.size>=2).map(([step])=>step).sort((a,b)=>a-b);
}

/** Candidates measured at exactly this update, best first; the difference is from the session's own control there. */
export function standings(s:ArenaSession,update:number):Standing[]{
 const at=new Map<string,number>();
 for(const r of s.runs)if(r.to===update&&r.devBpb!==null)at.set(r.candidate,Math.min(r.devBpb,at.get(r.candidate)??Infinity));
 const control=s.candidates.find(c=>c.control&&at.has(c.id)),reference=control?at.get(control.id):undefined;
 const rows=s.candidates.filter(c=>at.has(c.id)).map(c=>({candidate:c,bpb:at.get(c.id)!,delta:reference===undefined?undefined:at.get(c.id)!-reference}));
 rows.sort((a,b)=>a.bpb-b.bpb);
 return rows.map((r,i)=>({...r,rank:i+1}));
}

/** The default comparison step: the one with the most measured candidates, later steps winning ties. */
export function mainStep(s:ArenaSession):number|undefined{
 const steps=screeningSteps(s);
 return steps.reduce<number|undefined>((best,step)=>best===undefined||standings(s,step).length>=standings(s,best).length?step:best,undefined);
}

/** Cumulative research compute as a share of the budget, against hours since the session started. */
export function spendCurve(s:ArenaSession):{hours:number;share:number}[]{
 return [{hours:0,share:0},...s.spend.map(([at,flops])=>({hours:hoursBetween(s.start,at),share:flops/s.budget.limitFlops}))];
}

/** The formal run of the session's submission, through its arena card, if both are published. */
export function formalResult(s:ArenaSession,entries:ArenaEntry[],runs:Run[]):{entry?:ArenaEntry;run?:Run}{
 const entry=entries.find(e=>e.id===s.submission.arenaEntry);
 const run=entry&&runs.filter(r=>entry.runIds.includes(r.id)).sort((a,b)=>(b.lastRecordedAt??0)-(a.lastRecordedAt??0))[0];
 return {entry,run};
}

const shown=(v:unknown)=>v===null||v===undefined?'removed':Array.isArray(v)?'['+v.join(', ')+']':typeof v==='number'?v.toLocaleString('en-US',{maximumFractionDigits:4}):String(v);
/** What sets a candidate apart from the closest candidate queued before it: changed settings, then changed code. */
export function changeList(c:SessionCandidate):string[]{
 if(!c.basis)return [];
 return [...Object.entries(c.basis.changes).map(([k,v])=>k+' '+shown(v)),...(c.basis.codeChanged?['plugin code']:[])];
}
export function changeText(c:SessionCandidate,limit=3):string{
 if(!c.basis)return 'First candidate';
 const all=changeList(c);
 if(!all.length)return 'Same design as '+c.basis.candidate;
 return 'vs '+c.basis.candidate+' · '+all.slice(0,limit).join(', ')+(all.length>limit?` +${all.length-limit} more`:'');
}
