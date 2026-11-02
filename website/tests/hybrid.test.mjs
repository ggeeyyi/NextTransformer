import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {runs,methods} from './results.mjs';
import {leaderboardEntries} from '../app/lib/leaderboard.ts';
import {bpbDifference} from '../app/lib/chart-series.ts';

const CORE_STEPS=[8000,16000,24000,32000,40000,46875];
const VAL_STEPS=[...Array.from({length:188},(_,i)=>i*250),46875];
const evidence=path=>JSON.parse(readFileSync(new URL(`../../evidence/checks/${path}`,import.meta.url),'utf8'));
const RECORDS=[
 ['H02','gdn_hybrid',evidence('h02-b300-20261030/h02-b300-formal-20261027T0035-formal.json')],
 ['H03','gdn_hybrid_swa',evidence('h03-b300-20261031/h03-b300-fix-20261028T1845-formal.json')],
];

for(const [code,methodId,audit] of RECORDS){
 test(`the published ${code} record retains the full protocol and agrees with its frozen-source audit`,()=>{
  const run=runs().find(r=>r.methodId===methodId);
  assert.ok(run,`${methodId} run missing`);
  assert.equal(audit.ok,true);
  assert.equal(methods().find(m=>m.id===methodId).family,'hybrid');
  assert.equal(run.id,audit.run_id);
  assert.equal(run.status,'finished');assert.equal(run.complete,true);assert.equal(run.reportedAudit,audit.final_status);
  assert.equal(run.step,audit.optimizer_steps);
  assert.deepEqual(run.val.map(p=>p.step),VAL_STEPS);
  assert.deepEqual(run.core.map(p=>p.step),CORE_STEPS);
  assert.deepEqual(run.taskEvaluations.map(e=>e.step),CORE_STEPS);
  assert.equal(run.finalBpb,audit.final_bpb.val);
  assert.equal(run.val.at(-1).value,audit.periodic_val_bpb);
  assert.notEqual(run.finalBpb,run.val.at(-1).value);
  assert.equal(run.core.at(-1).value,audit.core_final);
  assert.equal(run.trainingCompute.inputTokens,audit.input_tokens);
  assert.equal(run.trainingCompute.trainingFlopsEstimate,audit.theory_flops);
  assert.equal(run.proof.science_hash,audit.provenance.science_hash);
  assert.equal(run.proof.src_head,audit.provenance.source_commit);
  assert.match(run.comparisonNote,/prerequisite/);
 });

 test(`${code} appears in final and replay rankings with its audited paired differences`,()=>{
  const all=runs(),dense=all.find(r=>r.baselineAcceptance),run=all.find(r=>r.methodId===methodId);
  for(const metric of ['val','core'])for(const mode of ['final','replay']){
   const entry=leaderboardEntries(all,metric,mode,46875).find(e=>e.run.id===run.id);
   assert.ok(entry.rank>0);assert.equal(entry.value,run[metric].at(-1).value);
  }
  assert.equal(bpbDifference([run],dense)[0].val.at(-1).value,run.val.at(-1).value-dense.val.at(-1).value);
  for(const [peer,comparison] of Object.entries(audit.comparisons)){
   const other=all.find(r=>r.methodId===peer);
   assert.ok(other,`${peer} run missing`);
   assert.equal(run.finalBpb-other.finalBpb,comparison.delta_final_val_bpb);
  }
 });
}
