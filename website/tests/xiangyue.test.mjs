import assert from 'node:assert/strict';
import {test} from 'node:test';
import {runs} from './results.mjs';
import {leaderboardEntries} from '../app/lib/leaderboard.ts';
import {trainingComputeDelta} from '../app/lib/training-compute.ts';
import {bpbDifference} from '../app/lib/chart-series.ts';
import {constellation} from '../app/lib/constellation.ts';
import {rankTrajectories} from '../app/lib/ranks.ts';

test('Xiangyue results use the standard final/replay rankings and measured Dense comparisons',()=>{
 const all=runs(),historical=all.filter(r=>r.cohort==='xiangyue-20261019'),dense=all.find(r=>r.baselineAcceptance);
 assert.equal(historical.length,2);assert.ok(dense);
 for(const r of historical){
  assert.equal(r.complete,true);assert.equal(r.val.length,189);assert.equal(r.taskEvaluations.length,6);
  assert.ok(r.comparisonNote.includes('all-backends-replay'));
  // Compute definitions remain distinct; quality scores use the same leaderboard rules as other runs.
  assert.equal(trainingComputeDelta(r,dense),undefined);
  const differences=bpbDifference([r],dense)[0].val;
  assert.equal(differences.at(-1).value,r.val.at(-1).value-dense.val.at(-1).value);
  const star=constellation([r],[],dense,()=>'')[0];
  assert.equal(star.dBpb,r.val.at(-1).value-dense.val.at(-1).value);
  assert.ok(Math.abs(star.dCore-(r.core.at(-1).value-dense.core.at(-1).value)*100)<1e-9);
  for(const metric of ['val','core'])for(const mode of ['final','replay']){
   const entry=leaderboardEntries(all,metric,mode,46875).find(e=>e.run.id===r.id);
   assert.ok(entry.rank>0);assert.equal(entry.value,r[metric].at(-1).value);
   assert.equal(entry[metric],r[metric].at(-1).value);
   assert.equal(rankTrajectories(all,metric,[46875]).get(r.id).at(-1).rank,entry.rank);
  }
 }
 const final=leaderboardEntries(all,'core','final',46875);
 const rank=id=>final.find(e=>e.run.methodId===id).rank;
 // New published methods can shift absolute ranks; GLA's higher measured CORE keeps it above DeltaNet.
 assert.ok(rank('gla')<rank('deltanet'));
 assert.equal(dense.trainingCompute.accounting,'reference-aten-math-v2');
});
