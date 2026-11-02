import assert from 'node:assert/strict';
import {test} from 'node:test';
import {runs} from './results.mjs';
import {computeFromSummary,getTrainingCompute,trainingComputeDelta,LEGACY_ACCOUNTING,trainingComputeSchema} from '../app/lib/training-compute.ts';
const native=runs().filter(r=>r.cohort==='native-v3');
test('owner-verified final costs preserve six-decimal values and Dense-relative differences',()=>{
 const dense=native.find(r=>r.methodId==='dense');
 for(const [id,value,delta]of [['dense',4.775226,0],['loopformer',4.775756,.01],['mor',4.957654,3.82],['circle',4.976652,4.22]]){
  const run=native.find(r=>r.methodId===id),cost=getTrainingCompute(run);
  assert.equal(cost.averageGflopsPerInputToken,value);assert.equal(cost.inputTokens,49152000000);assert.equal(cost.step,46875);assert.equal(cost.final,true);assert.equal(cost.trainingFlopsEstimate,undefined);
  assert.equal(Number(trainingComputeDelta(run,dense).toFixed(2)),delta);
 }
});
test('cumulative compute requires real counters, and config targets never stand in for an average',()=>{
 assert.equal(computeFromSummary({cost_proof_gflops_per_token:4.67,total_input_tokens:49152000000},46875,true,'definition','source'),undefined);
 assert.equal(computeFromSummary({training_flops_estimate:4e18,input_tokens:0},46875,true,'definition','source'),undefined);
 const c=computeFromSummary({training_flops_estimate:4e18,input_tokens:1e9},100,false,'definition','source');assert.equal(c.averageGflopsPerInputToken,4);assert.equal(c.step,100);assert.equal(c.final,false);
 const tracked=computeFromSummary({'train/training_flops_estimate':4e18,'train/input_tokens':1e9},100,false,'definition','source');assert.equal(tracked.averageGflopsPerInputToken,4);
 assert.equal(computeFromSummary({'train/training_flops_estimate':4e18,input_tokens:1e9},100,false,'definition','source'),undefined);
 const dense=native[0],run={...dense,trainingCompute:{...dense.trainingCompute,accounting:'another definition'}};
 assert.equal(trainingComputeDelta(run,dense),undefined);assert.equal(trainingComputeDelta({...dense,trainingCompute:{...dense.trainingCompute,final:false}},dense),undefined);
});
test('submissions validate compute units, positive denominators and cumulative/average consistency',()=>{
 const good={averageGflopsPerInputToken:4,trainingFlopsEstimate:4e18,inputTokens:1e9,step:46875,final:true,accounting:LEGACY_ACCOUNTING};
 assert.equal(trainingComputeSchema.parse(good).averageGflopsPerInputToken,4);
 assert.equal(trainingComputeSchema.safeParse({...good,inputTokens:0}).success,false);
 assert.equal(trainingComputeSchema.safeParse({...good,averageGflopsPerInputToken:5}).success,false);
 assert.equal(trainingComputeSchema.safeParse({...good,averageGflopsPerInputToken:NaN}).success,false);
});
