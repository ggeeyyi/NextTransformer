import assert from 'node:assert/strict';
import {test} from 'node:test';
import {budgetShare,changeList,changeText,devPath,formalResult,mainStep,screeningSteps,spendCurve,standings} from '../app/lib/arena-sessions.ts';
import {arena,runs,sessions} from './results.mjs';

test('every research session names its candidates, one submission and a budget it did not exceed',()=>{
 const all=sessions();
 assert.ok(all.length>=1);
 assert.equal(new Set(all.map(s=>s.id)).size,all.length);
 for(const s of all){
  assert.equal(typeof s.order,'number',s.id);
  const ids=new Set(s.candidates.map(c=>c.id));
  assert.ok(s.runs.every(r=>ids.has(r.candidate)),s.id);
  assert.equal(s.candidates.filter(c=>c.submitted).length,1,s.id);
  const selected=s.candidates.filter(c=>c.selected);
  assert.equal(selected.length,1,s.id);
  assert.ok(selected[0].trained,s.id);
  assert.ok(s.candidates.filter(c=>c.control).every(c=>c.trained),s.id);
  assert.ok(budgetShare(s)>0&&budgetShare(s)<=1,s.id);
  assert.ok(s.start<=s.submittedAt&&s.submittedAt<=s.end,s.id);
  assert.ok(s.runs.every(r=>r.queuedAt<=r.endedAt&&(!r.startedAt||(r.queuedAt<=r.startedAt&&r.startedAt<=r.endedAt))),s.id);
 }
});

test('the spend curve rises to the metered total and dev paths follow update order',()=>{
 for(const s of sessions()){
  const curve=spendCurve(s);
  curve.slice(1).forEach((p,i)=>assert.ok(p.share>=curve[i].share&&p.hours>=curve[i].hours,s.id));
  assert.equal(curve.at(-1).share,budgetShare(s));
  for(const c of s.candidates.filter(c=>c.trained)){
   const path=devPath(s,c.id);
   assert.ok(path.length>=1,c.id);
   path.slice(1).forEach((p,i)=>assert.ok(p.update>=path[i].update));
  }
 }
});

test('standings compare candidates only at the same update, against the control measured there',()=>{
 for(const s of sessions()){
  const step=mainStep(s);
  assert.ok(screeningSteps(s).includes(step),s.id);
  const rows=standings(s,step),control=rows.find(r=>r.candidate.control);
  assert.ok(rows.length>=2,s.id);
  rows.slice(1).forEach((r,i)=>assert.ok(r.bpb>=rows[i].bpb));
  for(const r of rows){
   assert.ok(s.runs.some(x=>x.candidate===r.candidate.id&&x.to===step&&x.devBpb===r.bpb),s.id);
   if(control)assert.ok(Math.abs(r.delta-(r.bpb-control.bpb))<1e-12);
   else assert.equal(r.delta,undefined);
  }
 }
});

test('a session with a published arena card agrees with it and reaches its formal run',()=>{
 const cards=arena(),all=runs();
 for(const s of sessions()){
  const card=cards.find(c=>c.id===s.submission.arenaEntry);
  if(!card)continue;
  assert.equal(card.methodId,s.submission.method,s.id);
  assert.equal(card.research.candidatesTrained,s.candidatesTrained,s.id);
  assert.equal(card.research.baseCommit,s.gym.commit,s.id);
  const {run}=formalResult(s,cards,all);
  assert.ok(run&&card.runIds.includes(run.id),s.id);
 }
});

test('each candidate is described by what differs from the closest candidate queued before it',()=>{
 for(const s of sessions()){
  const ids=new Set(s.candidates.map(c=>c.id)),order=[...s.candidates].sort((a,b)=>a.createdAt.localeCompare(b.createdAt));
  assert.equal(order[0].basis,null,s.id);
  for(const c of order.slice(1)){
   assert.ok(c.basis&&ids.has(c.basis.candidate)&&c.basis.candidate!==c.id,c.id);
   assert.ok(s.candidates.find(o=>o.id===c.basis.candidate).createdAt<=c.createdAt,c.id);
  }
 }
 const c={basis:{candidate:'abc',changes:{ngram_rows:524287,ngram_orders:[2,3],moe:null,gate:true},codeChanged:true}};
 assert.deepEqual(changeList(c),['ngram_rows 524,287','ngram_orders [2, 3]','moe removed','gate true','plugin code']);
 assert.equal(changeText(c),'vs abc · ngram_rows 524,287, ngram_orders [2, 3], moe removed +2 more');
 assert.equal(changeText({basis:null}),'First candidate');
 assert.equal(changeText({basis:{candidate:'abc',changes:{},codeChanged:false}}),'Same design as abc');
});
