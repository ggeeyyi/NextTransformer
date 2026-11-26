import {useEffect, useRef, useState} from 'react';
import {ArrowRight, ArrowUpRight, X} from 'lucide-react';
import Chart, {seriesColor} from './learning-chart';
import Leaderboard from './leaderboard';
import {docs, REPO_FILE} from './lib/links';
import {defaultMoeTrack, type MoeModel, type MoeTrack} from './lib/moe';
import {moeTracks} from './lib/moe-data';
import './moe-observatory.css';

const number = (n: number) => n.toLocaleString('en-US');
const score = (n: number) => (100 * n).toFixed(2);
const taskLabels: Record<string,string> = {
  arc_easy: 'ARC-E', arc_challenge: 'ARC-C', hellaswag: 'HellaSwag', piqa: 'PIQA',
  winogrande: 'WinoGrande', commonsense_qa: 'CSQA', openbook_qa: 'OBQA', lambada_openai: 'LAMBADA',
  squad: 'SQuAD', triviaqa: 'TriviaQA', ruler_niah_2k: 'RULER S-NIAH-1 · 2K',
  ruler_niah_4k: 'RULER S-NIAH-1 · 4K', fda: 'FDA', swde: 'SWDE',
};

export function TrackSelector({track, onChange}:{track?: MoeTrack; onChange:(id?:string)=>void}) {
  return <div className="track-selector">
    <div className="track-primary" role="group" aria-label="Benchmark track">
      <span className="track-title">Track</span>
      <div className="track-options">
        <button aria-pressed={!track} onClick={()=>onChange()}>Dense</button>
        <button aria-pressed={!!track} onClick={()=>onChange(track?.id ?? defaultMoeTrack)}>MoE</button>
      </div>
    </div>
    {track && <div className="track-control" role="group" aria-label="MoE budget">
      <span>240 TPP</span>
      {moeTracks.filter(t=>t.main).map(t=><button key={t.id} aria-pressed={track.id===t.id} onClick={()=>onChange(t.id)}>d{t.baseline.config.track.width}</button>)}
      <label className="reference-budget"><span className="leaderboard-sr">Marin reference budget</span>
        <select value={track.main?'':track.id} onChange={e=>e.target.value&&onChange(e.target.value)}>
          <option value="" disabled>Reference budgets</option>
          {moeTracks.filter(t=>!t.main).map(t=><option key={t.id} value={t.id}>{t.title}</option>)}
        </select>
      </label>
    </div>}
  </div>;
}

function Sources({model}:{model:MoeModel}) {
  const suffix = model.config.track.name + '/' + model.key + '.json';
  return <div className="moe-links">
    <a href={REPO_FILE+'moe/configs/'+suffix}>Configuration</a>
    <a href={REPO_FILE+model.config.calibration.cost_record}>Cost record</a>
    {model.record && <><a href={REPO_FILE+'moe/results/'+suffix}>Run record</a><a href={REPO_FILE+model.record.audit}>Run audit</a></>}
    {model.plugin?.paper && <a href={model.plugin.paper.url}>{model.plugin.paper.title} ↗</a>}
  </div>;
}

function ModelCard({model, onClose, onExplore}:{model:MoeModel;onClose:()=>void;onExplore:()=>void}) {
  const ref=useRef<HTMLDialogElement>(null);
  useEffect(()=>{const dialog=ref.current;dialog?.showModal();return ()=>dialog?.close();},[]);
  const {model: shape, track}=model.config, f=model.record?.final;
  return <dialog ref={ref} className="moe-dialog" aria-labelledby="moe-model-title" onCancel={onClose}
    onClick={e=>{if(e.target===e.currentTarget)onClose();}}>
    <div className="moe-dialog-body">
      <button className="icon-button" aria-label="Close model card" onClick={onClose}><X size={19}/></button>
      <p className="eyebrow">{track.name} / MODEL CARD</p><h2 id="moe-model-title">{model.name}</h2>
      {model.plugin && <p>{model.plugin.title}</p>}
      <p>{model.record?'Published · audit passed':'Admitted configuration · awaiting a published run'}</p>
      <dl className="moe-facts">
        <div><dt>Total parameters</dt><dd>{number(model.cost.parameters.total)}</dd></div>
        <div><dt>Active per token</dt><dd>{number(model.cost.parameters.active)}</dd></div>
        <div><dt>Backbone</dt><dd>{shape.layers} layers · width {track.width}</dd></div>
        <div><dt>Routing</dt><dd>Top {shape.top_k} of {shape.routed_experts} experts</dd></div>
        <div><dt>Calibrated FLOPs / token</dt><dd>{number(model.cost.flops_per_token)}</dd></div>
        {Object.entries(shape.options).map(([key,value])=><div key={key}><dt>{key.replaceAll('_',' ')}</dt><dd>{value}</dd></div>)}
      </dl>
      {f && <p><strong>{score(f.moe_core)}</strong> MoE-CORE · <strong>{f.val_macro_bpb.toFixed(4)}</strong> final val BPB</p>}
      <Sources model={model}/>
      {f && <button className="secondary" onClick={onExplore}>Explore results <ArrowRight size={15}/></button>}
    </div>
  </dialog>;
}

function Budget({track}:{track:MoeTrack}) {
  return <div className="moe-budget"><span>{track.title}</span><span>{number(track.context)} context</span>
    <span>{(track.tokens/1e9).toFixed(2)}B input tokens</span><span>{number(track.updates)} updates</span>
    <a href={docs('moe/protocol/')}>Protocol <ArrowUpRight size={13}/></a></div>;
}

function Models({models,open}:{models:MoeModel[]; open:(model:MoeModel)=>void}) {
  return <div className="moe-model-list">{models.map(m=><button key={m.id} onClick={()=>open(m)}>
    <span><strong>{m.name}</strong><small>{m.record?'Published result':'Admitted · awaiting result'}</small></span>
    <span>{(m.cost.parameters.total/1e9).toFixed(2)}B total <ArrowUpRight size={14}/></span>
  </button>)}</div>;
}

type Navigate = (view:string, model?:string)=>void;
function Results({track,focus,navigate,open}:{track:MoeTrack;focus:MoeModel;navigate:Navigate;open:(m:MoeModel)=>void}) {
  const [metric,setMetric]=useState<'core'|'val'>('core');
  const final=focus.record!.final;
  const pair=track.baseline.run && focus!==track.baseline ? [track.baseline,focus] : [focus];
  const options=track.models.filter(m=>m.run);
  return <section className="moe-results" aria-label="MoE results" id="trajectories">
    <div className="leaderboard-heading"><div><p className="eyebrow">01 / RESULTS</p>
      <h2>Learning curves.<span>{track.title}</span></h2></div>
      <label className="moe-picker">Model <select value={focus.key} onChange={e=>navigate('overview',e.target.value)}>
        {options.map(m=><option value={m.key} key={m.id}>{m.name}</option>)}</select></label></div>
    <div className="moe-summary">
      <div><small>Final MoE-CORE ↑</small><strong>{score(final.moe_core)}</strong></div>
      <div><small>Final val BPB ↓</small><strong>{final.val_macro_bpb.toFixed(4)}</strong></div>
      <div><small>Short / long · ×100</small><b>{score(final.short)} / {score(final.long)}</b></div>
      <button className="secondary" onClick={()=>open(focus)}>Model card <ArrowUpRight size={15}/></button>
    </div>
    <div className="chart-card moe-chart">
      <div className="leaderboard-metric" role="group" aria-label="MoE chart metric">
        <button aria-pressed={metric==='core'} onClick={()=>setMetric('core')}>MoE-CORE ↑</button>
        <button aria-pressed={metric==='val'} onClick={()=>setMetric('val')}>Validation BPB ↓</button></div>
      <div className="plot-key">{pair.map(m=><span key={m.id}><i style={{background:seriesColor(m.id,track.models)}}/>{m.name}</span>)}
        <small>{metric==='core'?'Scheduled MoE-CORE ×100':'Sampled validation · bits / byte'}</small></div>
      <Chart runs={pair.map(m=>m.run!)} methods={track.models} metric={metric} max={track.updates} cursor={track.updates}
        wide compact allRange ariaLabel={metric==='core'?'Scheduled MoE-CORE by training step':'Sampled validation BPB by training step'}/>
      <div className="plot-caption"><span>Training step</span><span>Measured checkpoints</span></div>
      <p>Training curves show scheduled evaluations. The headline scores come from a separate final pass over all 14 tasks and the full six-source validation set.</p>
    </div>
    <details className="moe-detail" open><summary>Final scores by task · 14 tasks</summary>
      <div className="moe-table-wrap"><table><thead><tr><th>Task</th>{pair.map(m=><th key={m.id}>{m.name} · ×100</th>)}</tr></thead>
        <tbody>{Object.entries(final.centered).map(([task])=><tr key={task}><th>{taskLabels[task]??task}</th>
          {pair.map(m=><td key={m.id}>{score(m.record!.final.centered[task])}</td>)}</tr>)}</tbody></table></div></details>
    <details className="moe-detail"><summary>Final validation by source · BPB</summary>
      <div className="moe-table-wrap"><table><thead><tr><th>Source</th>{pair.map(m=><th key={m.id}>{m.name} · BPB</th>)}</tr></thead>
        <tbody>{Object.keys(final.val_sources).map(source=><tr key={source}><th>{source}</th>
          {pair.map(m=><td key={m.id}>{m.record!.final.val_sources[source].bpb.toFixed(4)}</td>)}</tr>)}</tbody></table></div></details>
    <Sources model={focus}/>
    <button className="text-button" onClick={()=>navigate('leaderboard')}>Compare final rankings <ArrowRight size={15}/></button>
  </section>;
}

export default function MoeObservatory({track,view,method,navigate}:{track:MoeTrack;view:string;method:string|null;navigate:Navigate}) {
  const [card,setCard]=useState<MoeModel|null>(null);
  const [cursor,setCursor]=useState(track.updates),[playing,setPlaying]=useState(false);
  useEffect(()=>{
    if(!playing) return;
    const started=performance.now(), first=track.coreSteps[0]??0;
    const timer=window.setInterval(()=>{
      const fraction=Math.min(1,(performance.now()-started)/12000);
      setCursor(Math.round(first+(track.updates-first)*fraction));
      if(fraction===1) setPlaying(false);
    },80);
    return ()=>window.clearInterval(timer);
  },[playing,track]);
  const chosen=track.models.find(m=>m.key===method&&m.run) ?? track.models.find(m=>m.run);
  const missing=track.models.filter(m=>!m.run);
  const modelFor=(id:string)=>track.models.find(m=>m.id===id)!;
  const explore=(m:MoeModel)=>{setCard(null);navigate('overview',m.key);};
  return <div className="moe-workspace">
    <Budget track={track}/>
    {view==='atlas' ? <section aria-label="MoE model atlas"><div className="directory-heading"><div>
      <p className="eyebrow">05 / MODEL ATLAS</p><h2>Model atlas.<em>{track.title}</em></h2>
      <p>{track.models.length} admitted {track.models.length===1?'model':'models'} · {track.runs.length} published {track.runs.length===1?'result':'results'}</p>
      </div></div><Models models={track.models} open={setCard}/></section>
    : !chosen ? <section className="moe-awaiting" aria-label={view==='leaderboard'?'Model leaderboard':'MoE results'}>
      <h2>{view==='leaderboard'?'Leaderboard':'Results'} · {track.title}</h2>
      <p>No published runs yet. Open a model to see its admitted configuration and cost record.</p>
      <Models models={track.models} open={setCard}/></section>
    : view==='leaderboard' ? <>
      <Leaderboard moe={track} runs={track.runs} methods={track.models} baseline={track.baseline.run}
        cursor={cursor} playing={playing} onReplay={()=>{setCursor(track.coreSteps[0]??0);setPlaying(true);}}
        onPause={()=>setPlaying(false)} onCursor={step=>{setPlaying(false);setCursor(step);}} onModel={setCard}
        onDetails={run=>setCard(modelFor(run.id))} onCompare={run=>explore(modelFor(run.id))}/>
      {missing.length>0&&<details className="moe-detail"><summary>{missing.length} admitted models awaiting results</summary>
        <Models models={missing} open={setCard}/></details>}
    </> : <Results track={track} focus={chosen} navigate={navigate} open={setCard}/>}
    {card && <ModelCard model={card} onClose={()=>setCard(null)} onExplore={()=>explore(card)}/>}
  </div>;
}
