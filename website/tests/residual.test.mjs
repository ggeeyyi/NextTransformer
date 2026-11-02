import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {runs,methods} from './results.mjs';
import {leaderboardEntries} from '../app/lib/leaderboard.ts';
import {bpbDifference} from '../app/lib/chart-series.ts';

const CORE_STEPS=[8000,16000,24000,32000,40000,46875];
const VAL_STEPS=[...Array.from({length:188},(_,i)=>i*250),46875];
const audits={residual_standard:'a800-residual-standard-formal.json',attnres:'a800-attnres-formal.json'};

test('published Residual records retain the full protocol and agree with their cumulative audits',()=>{
 const all=runs(),cards=methods();
 for(const [id,file] of Object.entries(audits)){
  const run=all.find(r=>r.methodId===id);
  const audit=JSON.parse(readFileSync(new URL('../../evidence/checks/residual-20261028/'+file,import.meta.url),'utf8'));
  assert.ok(run,id+' run missing');
  assert.equal(audit.ok,true);
  assert.equal(run.id,'nextt-v2-residual-20261023_'+id+'_a800_seed42');
  assert.equal(cards.find(m=>m.id===id).family,'residual');
  assert.equal(run.status,'finished');assert.equal(run.complete,true);assert.equal(run.reportedAudit,'matched');
  assert.equal(run.step,audit.optimizer_steps);
  assert.deepEqual(run.val.map(p=>p.step),VAL_STEPS);
  assert.deepEqual(run.core.map(p=>p.step),CORE_STEPS);
  assert.deepEqual(run.taskEvaluations.map(e=>e.step),CORE_STEPS);
  assert.equal(run.finalBpb,audit.final_bpb.val);
  assert.equal(run.val.at(-1).value,audit.periodic_val_bpb['46875']);
  for(const p of run.core)assert.equal(p.value,audit.core[p.step]);
  assert.equal(run.trainingCompute.inputTokens,audit.input_tokens);
  assert.equal(run.trainingCompute.trainingFlopsEstimate,audit.theory_flops);
  assert.equal(run.proof.model_hash,audit.fingerprints.model_hash);
  assert.equal(run.proof.science_hash,audit.fingerprints.science_hash);
  assert.equal(run.proof.src_head,audit.segments[0].commit);
  assert.equal(run.proof.src_head,'4dd6ebb9f4f1f8b4571963c5df2aaa384eea7825');
  assert.equal(run.baselineAcceptance,undefined);
 }
});

test('Residual results appear in final and replay rankings with measured Dense differences',()=>{
 const all=runs(),dense=all.find(r=>r.baselineAcceptance);
 for(const id of Object.keys(audits)){
  const run=all.find(r=>r.methodId===id);
  for(const metric of ['val','core'])for(const mode of ['final','replay']){
   const entry=leaderboardEntries(all,metric,mode,46875).find(e=>e.run.id===run.id);
   assert.ok(entry.rank>0);assert.equal(entry.value,run[metric].at(-1).value);
  }
  assert.equal(bpbDifference([run],dense)[0].val.at(-1).value,run.val.at(-1).value-dense.val.at(-1).value);
 }
});
