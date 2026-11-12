/// <reference types="vite/client" />
// Published records: every JSON file under results/ in the repository, bundled at build time.
import type {ArenaEntry,Method,Run,ScalingLadder,ScalingRecord} from './types';
const runFiles=import.meta.glob('../../../results/runs/*.json',{eager:true,import:'default'}) as Record<string,Run>;
const methodFiles=import.meta.glob('../../../results/methods/*.json',{eager:true,import:'default'}) as Record<string,Method>;
const arenaFiles=import.meta.glob('../../../results/arena/*.json',{eager:true,import:'default'}) as Record<string,ArenaEntry>;
const scalingFiles=import.meta.glob('../../../results/scaling/runs/*.json',{eager:true,import:'default'}) as Record<string,ScalingRecord>;
const ladderFiles=import.meta.glob('../../../results/scaling/ladder.json',{eager:true,import:'default'}) as Record<string,ScalingLadder>;
const byOrder=(a:any,b:any)=>(a.order??Number.MAX_SAFE_INTEGER)-(b.order??Number.MAX_SAFE_INTEGER)||String(a.id).localeCompare(String(b.id));
export const runs:Run[]=Object.values(runFiles).sort(byOrder);
export const methods:Method[]=Object.values(methodFiles).sort(byOrder);
export const arenaEntries:ArenaEntry[]=Object.values(arenaFiles).sort(byOrder);
export const scalingRecords:ScalingRecord[]=Object.values(scalingFiles).sort((a,b)=>a.training.trainingFlopsEstimate-b.training.trainingFlopsEstimate);
export const scalingLadder:ScalingLadder|undefined=Object.values(ladderFiles)[0];
const latest=Math.max(0,...runs.map(r=>r.lastRecordedAt??0));
export const updatedAt=latest?new Date(latest*1000).toISOString():'';
export const SOURCE='Repository records · results/';
