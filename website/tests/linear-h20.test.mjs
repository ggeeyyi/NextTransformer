import assert from 'node:assert/strict';
import {test} from 'node:test';
import {runs as allRuns,methods as allMethods} from './results.mjs';

const methods=allMethods(),linearIds=new Set(methods.filter(m=>m.family==='linear').map(m=>m.id));
const runs=allRuns().filter(r=>r.id.includes('_h20_')&&linearIds.has(r.methodId));
const VAL_STEPS=[...Array.from({length:188},(_,i)=>i*250),46875],CORE_STEPS=[8000,16000,24000,32000,40000,46875];

test('audited H20 linear runs are complete protocol records with final reference accounting',()=>{
 assert.deepEqual(runs.map(r=>r.methodId).sort(),['elu_qknorm','gdn','kda','retnet']);
 for(const run of runs){
  assert.equal(run.status,'finished');assert.equal(run.complete,true);assert.equal(run.reportedAudit,'matched');assert.equal(run.step,46875);
  assert.deepEqual(run.val.map(p=>p.step),VAL_STEPS);assert.deepEqual(run.core.map(p=>p.step),CORE_STEPS);
  assert.deepEqual(run.taskEvaluations.map(e=>e.step),CORE_STEPS);
  const c=run.trainingCompute,p=run.proof;
  assert.ok(c.final&&c.step===46875&&c.inputTokens===49152000000&&c.accounting==='reference-aten-math-v2');
  assert.ok(Math.abs(c.averageGflopsPerInputToken/4.67239832544785-1)<=.02);
  // Every update charges the admitted proof, so the cumulative average equals it.
  assert.ok(Math.abs(c.averageGflopsPerInputToken/p.cost_proof_gflops_per_token-1)<1e-9);
  assert.ok(/^[a-f0-9]{40}$/.test(p.src_head)&&p.total_input_tokens===49152000000&&p.global_batch_tokens===1048576);
  // The independent final BPB is a separate pass, not the last periodic validation.
  assert.ok(Number.isFinite(run.finalBpb)&&run.finalBpb!==run.val.at(-1).value);
  const method=methods.find(m=>m.id===run.methodId);
  assert.ok(method,run.methodId+' has no model card');
  assert.ok(Math.abs(method.flops-p.cost_proof_gflops_per_token)<5e-7&&method.costModel==='reference-aten-math-v2');
 }
});
