import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {buildMoeTracks, moeAddress} from '../app/lib/moe.ts';
import {leaderboardEntries, rankingCheckpoint} from '../app/lib/leaderboard.ts';

const small='d768-240tpp', large='d1024-240tpp';
const path=(area,track,key)=>'moe/'+area+'/'+track+'/'+key+'.json';
const read=file=>JSON.parse(readFileSync(new URL('../../'+file,import.meta.url),'utf8'));
function sample() {
  const files={configs:{}, costs:{}, audits:{}, results:{}, plugins:{}};
  for(const [track,key] of [[small,'baseline'],[small,'hc'],[large,'baseline']]){
    for(const [bucket,area] of [['configs','configs'],['costs','evidence/costs']]){
      const file=path(area,track,key);
      files[bucket][file]=read(file);
    }
  }
  for(const [bucket,area] of [['audits','evidence/audits'],['results','results']]){
    const file=path(area,small,'baseline');
    files[bucket][file]=read(file);
  }
  return files;
}
test('MoE budgets admit models separately from ranked runs and retain the original launch identity',()=>{
  const [s,l]=buildMoeTracks(sample());
  assert.equal(s.id,small); assert.equal(l.id,large);
  assert.equal(s.tokens,14073462784); assert.equal(l.tokens,34064564224);
  assert.equal(s.models.length,2); assert.equal(l.models.length,1);
  assert.equal(s.runs.length,1); assert.equal(l.runs.length,0);
  assert.notEqual(s.baseline.id,l.baseline.id);
  const rules={finalStep:s.updates,separateFinalEvaluation:true};
  for(const [metric,expected,periodic] of [['core','36.82','36.83'],['val','0.7708','0.7498']]){
    const format=v=>metric==='core'?(v*100).toFixed(2):v.toFixed(4);
    const step=rankingCheckpoint(s.baseline.run,metric,s.updates,s.runs,s.updates);
    assert.equal(step,s.updates);
    assert.equal(format(leaderboardEntries(s.runs,metric,'final',step,undefined,rules)[0].value),expected);
    assert.equal(format(leaderboardEntries(s.runs,metric,'replay',step,undefined,rules)[0].value),periodic);
  }
  assert.equal(moeAddress('leaderboard'),'#leaderboard');
  assert.equal(moeAddress('results',large,'baseline'),'#results?track=d1024-240tpp&method=baseline');
});
test('a new published arm automatically joins its budget once its matching audit is present',()=>{
  const f=sample(), hc=f.configs[path('configs',small,'hc')];
  const record=structuredClone(f.results[path('results',small,'baseline')]);
  const audit=structuredClone(f.audits[path('evidence/audits',small,'baseline')]);
  Object.assign(record,{method:'hc',name:'HC',model_hash:hc.model_hash,audit:path('evidence/audits',small,'hc')});
  Object.assign(audit,{method:'hc',model_hash:hc.model_hash});
  record.final.moe_core+=0.01;
  f.results[path('results',small,'hc')]=record;
  assert.equal(buildMoeTracks(f)[0].runs.length,1);
  f.audits[record.audit]={...audit,track:large};
  assert.equal(buildMoeTracks(f)[0].runs.length,1);
  f.audits[record.audit]={...audit,passed:false};
  assert.equal(buildMoeTracks(f)[0].runs.length,1);
  f.audits[record.audit]=audit;
  const [s,l]=buildMoeTracks(f);
  const entries=leaderboardEntries(s.runs,'core','final',undefined,undefined,{finalStep:s.updates,separateFinalEvaluation:true});
  assert.equal(entries[0].run.methodId,'moe/'+small+'/hc');
  assert.equal(entries[1].run.id,s.baseline.id);
  assert.equal(l.runs.length,0);
  delete record.final.val_macro_bpb;
  assert.equal(buildMoeTracks(f)[0].runs.length,1);
});
