import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {runs,methods} from './results.mjs';
import {leaderboardEntries} from '../app/lib/leaderboard.ts';
import {bpbDifference} from '../app/lib/chart-series.ts';

const CORE_STEPS=[8000,16000,24000,32000,40000,46875];
const VAL_STEPS=[...Array.from({length:188},(_,i)=>i*250),46875];
const tasks=JSON.parse(readFileSync(new URL('../data/core-tasks.json',import.meta.url),'utf8')).tasks;
const ids=['mla_qknorm','mla_nope_qknorm'];

test('published H20 Others records retain the full protocol and agree with their cumulative audits',()=>{
 const all=runs(),cards=methods();
 for(const id of ids){
  const run=all.find(r=>r.methodId===id);
  const audit=JSON.parse(readFileSync(new URL('../../evidence/checks/others-20261028/h20-'+id.replaceAll('_','-')+'-formal.json',import.meta.url),'utf8'));
  assert.ok(run,id+' run missing');
  assert.equal(run.id,id+'_h20_seed42');
  assert.equal(cards.find(m=>m.id===id).family,'others');
  assert.equal(run.status,'finished');assert.equal(run.complete,true);assert.equal(run.reportedAudit,'matched');
  assert.equal(run.step,audit.optimizer_steps);
  assert.deepEqual(run.val.map(p=>p.step),VAL_STEPS);
  assert.deepEqual(run.core.map(p=>p.step),CORE_STEPS);
  assert.deepEqual(run.taskEvaluations.map(e=>e.step),CORE_STEPS);
  assert.equal(run.finalBpb,audit.final_bpb.val);
  assert.equal(run.val.at(-1).value,audit.periodic_val_bpb['46875']);
  assert.notEqual(run.finalBpb,run.val.at(-1).value);
  assert.equal(run.core.at(-1).value,audit.core_final);
  assert.equal(run.trainingCompute.inputTokens,audit.input_tokens);
  assert.equal(run.trainingCompute.trainingFlopsEstimate,audit.theory_flops);
  assert.equal(run.trainingCompute.accounting,'reference-aten-math-v2');
  assert.equal(run.proof.model_hash,audit.fingerprints.model_hash);
  assert.equal(run.proof.science_hash,audit.fingerprints.science_hash);
  assert.equal(run.proof.src_head,'a0c795c47eff3bf6fabb50034ddf84ab12028817');
  assert.equal(run.proof.code_snapshot,audit.segment_provenance[0].code);
  assert.match(run.comparisonNote,/frozen Others branch/);
  for(const e of run.taskEvaluations){
   assert.deepEqual(Object.keys(e.raw).sort(),tasks.map(t=>t.id).sort());
   assert.deepEqual(e.centered,audit.core_tasks_centered[e.step]);
   for(const t of tasks)assert.ok(Math.abs((e.raw[t.id]-t.randomBaseline)/(1-t.randomBaseline)-e.centered[t.id])<1e-7);
   assert.ok(Math.abs(Object.values(e.centered).reduce((a,b)=>a+b,0)/22-audit.core[e.step])<1e-7);
  }
 }
});

test('H20 Others results appear in final and replay rankings with measured Dense differences',()=>{
 const all=runs(),dense=all.find(r=>r.baselineAcceptance);
 for(const id of ids){
  const run=all.find(r=>r.methodId===id);
  for(const metric of ['val','core'])for(const mode of ['final','replay']){
   const entry=leaderboardEntries(all,metric,mode,46875).find(e=>e.run.id===run.id);
   assert.ok(entry.rank>0);assert.equal(entry.value,run[metric].at(-1).value);
  }
  assert.equal(bpbDifference([run],dense)[0].val.at(-1).value,run.val.at(-1).value-dense.val.at(-1).value);
 }
});
