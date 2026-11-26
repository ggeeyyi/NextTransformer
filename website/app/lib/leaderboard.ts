import type {Point,Run} from './types';

export const FINAL_STEP=46875;
export type RankingMetric='core'|'val';
export type RankingMode='final'|'replay';
export type RankingRules = {finalStep: number; separateFinalEvaluation?: boolean};
const denseRules: RankingRules = {finalStep: FINAL_STEP};

export function scoreAt(run: Run, metric: RankingMetric, mode: RankingMode, step: number | undefined, rules: RankingRules): number | undefined {
 const value = mode === 'final' && rules.separateFinalEvaluation
  ? run.finalEvaluation?.[metric] : run[metric].find(p => p.step === step && Number.isFinite(p.value))?.value;
 return Number.isFinite(value) ? value : undefined;
}
export type LeaderboardEntry={run:Run;rank?:number;value?:number;core?:number;val?:number;latestCore?:Point;latestVal?:Point;differentBudget:boolean;coreOnly:boolean};

// Pick the most recently recorded attempt, never the attempt with the best score.
export function preferredRuns(runs:Run[],allAttempts=false):Run[]{
 if(allAttempts)return runs;
 const chosen=new Map<string,Run>();
 for(const run of runs){
  const previous=chosen.get(run.methodId);
  if(!previous||(run.lastRecordedAt??0)>(previous.lastRecordedAt??0))chosen.set(run.methodId,run);
 }
 return [...chosen.values()];
}

export function rankingCheckpoint(baseline:Run|undefined,metric:RankingMetric,cursor:number,runs:Run[]=[],finalStep=FINAL_STEP):number|undefined{
 return (baseline?.[metric]??runs.flatMap(r=>r[metric])).filter(p=>Number.isFinite(p.value)&&p.step<=cursor&&p.step<=finalStep).map(p=>p.step).sort((a,b)=>a-b).at(-1);
}

// `coreOnly` marks runs that rank on CORE only (auto-research entries): they keep their measured values but take no
// place in the validation BPB ranking.
export function leaderboardEntries(runs:Run[],metric:RankingMetric,mode:RankingMode,step:number|undefined,coreOnly:(run:Run)=>boolean=()=>false,rules:RankingRules=denseRules):LeaderboardEntry[]{
 const at=mode==='final'?rules.finalStep:step;
 const entries=runs.map(run=>{
  const point=(kind:RankingMetric)=>scoreAt(run,kind,mode,at,rules);
  const latest=(kind:RankingMetric)=>run[kind].filter(p=>Number.isFinite(p.value)&&(mode==='final'||(at!==undefined&&p.step<=at))).sort((a,b)=>a.step-b.step).at(-1);
  const target=Number(run.proof?.total_steps??rules.finalStep),differentBudget=target!==rules.finalStep;
  const core=point('core'),val=point('val');
  const only=coreOnly(run);
  return {run,core,val,value:(mode==='final'&&differentBudget)||(metric==='val'&&only)?undefined:metric==='core'?core:val,latestCore:latest('core'),latestVal:latest('val'),differentBudget,coreOnly:only};
 });
 entries.sort((a,b)=>{
  if(a.value===undefined&&b.value!==undefined)return 1;
  if(a.value!==undefined&&b.value===undefined)return -1;
  if(a.value!==undefined&&b.value!==undefined){const delta=metric==='core'?b.value-a.value:a.value-b.value;if(Math.abs(delta)>1e-12)return delta;}
  return a.run.methodId.localeCompare(b.run.methodId)||a.run.id.localeCompare(b.run.id);
 });
 let previous:number|undefined,rank=0;
 return entries.map((entry,index)=>{
  if(entry.value===undefined)return entry;
  if(previous===undefined||Math.abs(entry.value-previous)>1e-12)rank=index+1;
  previous=entry.value;
  return {...entry,rank};
 });
}

/** Group runs that have a CORE measurement by the longest input actually evaluated. */
export function coreLengthGroups(runs: Run[], name: (run: Run) => string) {
 const within = new Set<string>(), beyond = new Set<string>();
 for (const run of runs) {
  const context = run.coreContext;
  if (!context || !run.core.length) continue;
  const group = context.maxEvaluatedTokens > context.trainingTokens ? beyond : within;
  group.add(name(run));
 }
 return {within: [...within].sort(), beyond: [...beyond].sort()};
}
