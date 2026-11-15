import { memo, useState } from 'react';
import { Activity, ArrowDownToLine, ArrowUpRight, BookOpen, Check, CircleDashed, Layers3 } from 'lucide-react';
import { docs, REPO_FILE } from './lib/links';
import { publishedT3 } from './lib/results';
import { budget, t3 } from './lib/scaling';
import type { Cell, Family, Hardware } from './lib/scaling';
import './scaling.css';

const colors: Record<string,string> = {dense:'#536e88',moe:'#967752','kda-dsa':'#7c6693',looped:'#a26c76'};
const roadmap = [
  ['T3','Size sweep','Public design'],
  ['T4','Refine the fit','Planned'],
  ['T5','Check the forecast','Planned'],
  ['C200','Final blind test','Held out'],
];
const configUrl = new URL('../../configs/scaling/t3.json', import.meta.url).href;
const number = (value:number,digits=0) => value.toLocaleString('en-US',{minimumFractionDigits:digits,maximumFractionDigits:digits});

function Tradeoff({family,cell,onPick}:{family:Family;cell:Cell;onPick:(depth:number)=>void}) {
  const cells=t3.cells.filter(c=>c.family===family.id);
  const points=cells.filter(c=>c.estimatedTokens!==null);
  const x=(depth:number)=>25+(depth-16)*16;
  const y=(tokens:number)=>85-65*Math.log(tokens/7e9)/Math.log(70/7);
  return <div className="ts-figure">
    <div className="ts-fig-top"><span>{points.length?'Training tokens by depth':'Mixer composition by depth'}</span><small>{points.length?'Planning estimate':'Layers per model'}</small></div>
    {points.length ? <svg viewBox="0 0 310 115" role="img" aria-label="Estimated training tokens decline with depth at the fixed T3 compute budget.">
      {[10,25,60].map(b=><g key={b}><line x1="24" x2="280" y1={y(b*1e9)} y2={y(b*1e9)} className="ts-gridline"/><text x="287" y={y(b*1e9)+3}>{b}B</text></g>)}
      <polyline points={points.map(c=>x(c.layers)+','+y(c.estimatedTokens!)).join(' ')} fill="none" stroke={colors[family.id]} strokeWidth="2"/>
      {points.map(c=><g key={c.id}><circle cx={x(c.layers)} cy={y(c.estimatedTokens!)} r={cell.id===c.id?5:3.5} fill={cell.id===c.id?colors[family.id]:'white'} stroke={colors[family.id]} strokeWidth="2"/><text x={x(c.layers)} y="107" textAnchor="middle">L{c.layers}</text></g>)}
    </svg>:<div className="ts-mix">{cells.map(c=><button key={c.id} onClick={()=>onPick(c.layers)} aria-pressed={c.id===cell.id}><b>L{c.layers}</b><span>{Array.from({length:c.layers/4},(_,i)=><i key={i}/>)}</span><small>{c.kdaLayers} + {c.dsaLayers}</small></button>)}</div>}
  </div>;
}

function Scaling() {
  const [hardware,setHardware]=useState<Hardware>('H200');
  const [unit,setUnit]=useState<'days'|'hours'>('days');
  const [scenario,setScenario]=useState(1);
  const [activeFamily,setFamily]=useState('dense');
  const [depth,setDepth]=useState(20);
  const family=t3.families.find(f=>f.id===activeFamily)!;
  const cell=t3.cells.find(c=>c.family===activeFamily&&c.layers===depth)!;
  const counts=(f:Family)=>t3.cells.filter(c=>c.family===f.id).length;
  const perRun=(f:Family)=>budget(f,hardware,scenario)[unit];
  const max=Math.max(...t3.families.map(perRun));
  const total=t3.families.reduce((a,f)=>a+perRun(f)*counts(f),0);
  const quantity=(v:number)=>number(v,unit==='days'?1:0);
  const unitText=unit==='days'?'node-days':'GPU-hours';
  const tokens=cell.estimatedTokens===null?'Pending proof':number(cell.estimatedTokens/1e9,2)+'B';
  const note=cell.estimatedTokens===null?'The dense burn-in and sparse phase need separate cost proofs.':'Token and update counts use the planning ledger, rounded to whole updates.';
  const shape=cell.preLayers!==undefined?cell.preLayers+' + '+cell.coreLayers+' × 2 + '+cell.postLayers:cell.expertWidth!==undefined?'256 / 8 / 1':cell.kdaLayers!==undefined?cell.kdaLayers+' KDA + '+cell.dsaLayers+' DSA':'4 × width';
  const shapeLabel=cell.preLayers!==undefined?'Prelude / core / coda':cell.expertWidth!==undefined?'Routed / selected / shared':cell.kdaLayers!==undefined?'Mixer layers':'FFN expansion';
  return <section className="t3-study" id="scaling" aria-labelledby="ts-title">
    <header className="ts-hero">
      <div><div className="ts-eyeline"><i/>Scaling study <span>/</span> 2K pretraining</div><h1 id="ts-title">Scaling at fixed compute<span>.</span></h1>
        <p>How should each architecture divide one training budget between model size and tokens?</p>
        <div className="ts-facts"><span><b>2,048</b> context</span><span><b>42</b> seed</span><span><b>4</b> families</span></div></div>
      <div className="ts-anchor"><div><Layers3 size={18}/><span>T3 budget per run</span></div><strong>10<sup>20</sup> <small>FLOPs</small></strong><p>One compute target across 16 planned runs.</p></div>
    </header>
    <ol className="ts-road" aria-label="Research stages">{roadmap.map(([id,title,status],i)=><li key={id} className={i===0?'current':''}><div><b>{id}</b><small>{status}</small></div><span>{title}</span></li>)}</ol>
    <div className="ts-progress" aria-label="Public study progress">
      <span><Check size={14}/> {t3.cells.length} T3 designs available</span>
      <span>{publishedT3.size} / {t3.cells.length} T3 results published</span>
      <a href={REPO_FILE+'results/scaling/runs/dense_200b.json'} target="_blank" rel="noreferrer">Completed 4K reference <ArrowUpRight size={12}/></a>
    </div>
    <section className="ts-sweep" aria-labelledby="ts-sweep-title">
      <div className="ts-section-heading"><div><h2 id="ts-sweep-title">T3 budget by family</h2><p>Each family spans 16, 20, 28 and 32 layers. Select a row to inspect its design.</p></div><a href={docs('scaling/protocol/')}><BookOpen size={15}/> Protocol <ArrowUpRight size={13}/></a></div>
      <div className="ts-toolbar">
        <div className="ts-control"><span>Hardware</span><div role="group" aria-label="Hardware">{(['A100','H200'] as const).map(g=><button aria-pressed={g===hardware} key={g} onClick={()=>setHardware(g)}>{g}</button>)}</div></div>
        <div className="ts-control"><span>Unit</span><div role="group" aria-label="Budget unit">{(['days','hours'] as const).map(u=><button aria-pressed={u===unit} key={u} onClick={()=>setUnit(u)}>{u==='days'?'8-GPU days':'GPU-hours'}</button>)}</div></div>
        <label className="ts-scenario">Throughput <select value={scenario} onChange={e=>setScenario(Number(e.target.value))}><option value={0}>Lower</option><option value={1}>Central</option><option value={2}>Higher</option></select></label>
      </div>
      <div className="ts-main-grid">
        <div className="ts-cost"><div className="ts-cost-label"><span>Architecture family</span><span>Per run</span><span>Four runs</span></div>
          {t3.families.map(f=><button key={f.id} className="ts-row" aria-pressed={activeFamily===f.id} onClick={()=>setFamily(f.id)}>
            <span className="ts-row-name"><i style={{background:colors[f.id]}}/>{f.name}</span><span className="ts-row-value">{quantity(perRun(f))}</span>
            <span className="ts-bar-value">{quantity(perRun(f)*counts(f))}<i><em style={{width:(perRun(f)/max*100)+'%',background:colors[f.id]}}/></i></span>
          </button>)}
          <div className="ts-total"><span>T3 total <small>16 planned runs</small></span><b>{quantity(total)} <small>{unitText}</small></b></div>
          <p className="ts-fine">Estimates include 15% operating overhead. One node-day is eight {hardware} GPUs for 24 hours. Pilots are budgeted separately.</p>
        </div>
        <aside className="ts-inspect" aria-live="polite">
          <div className="ts-inspect-heading"><div><small>T3 configuration</small><h3>{family.name}</h3></div><a href={configUrl} download="t3.json" aria-label="Download the public T3 configuration"><ArrowDownToLine size={17}/></a></div>
          <div className="ts-depth" role="group" aria-label="Execution depth">{t3.depths.map(l=><button key={l} onClick={()=>setDepth(l)} aria-pressed={l===depth}>L{l}</button>)}</div>
          <Tradeoff family={family} cell={cell} onPick={setDepth}/>
          <dl className="ts-values">{[['Width / heads',number(cell.width)+' / '+cell.heads],[shapeLabel,shape],['Training tokens',tokens],['Updates',cell.estimatedUpdates===null?'Pending proof':number(cell.estimatedUpdates)]].map(([k,v])=><div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
          <p className="ts-note">{note}</p>
          <div className="ts-cell-status"><span>{publishedT3.has(cell.id)
            ? <a href={REPO_FILE+'results/scaling/runs/'+cell.id+'.json'}>Published result ↗</a>
            : 'Formal result pending'}</span>
            <a href={'#contribute?cell='+encodeURIComponent(cell.id)}>Contribute to {cell.id} <ArrowUpRight size={12}/></a></div>
        </aside>
      </div>
    </section>
    <section className="ts-method"><div className="ts-section-heading"><div><h2>Reading the study</h2><p>Designs share ClimbMix, the tokenizer, seed and evaluation. Each family changes one size variable.</p></div></div>
      <div className="ts-notes"><article><Activity size={18}/><h3>Find the fit</h3><p>Use final validation BPB to select size and token count. Track CORE-22 as a downstream check.</p></article><article><CircleDashed size={18}/><h3>Forecast before training</h3><p>Publish the T5 predictions, score them, then include those observations in the final fit.</p></article><article><Check size={18}/><h3>Hold out C200</h3><p>Freeze forecasts across all four families before the final test. The historical 4K Dense run supplies the compute reference.</p></article></div>
      <details><summary>Recipe, budget assumptions and participation</summary><p>{family.description} The selected configuration uses a {cell.ffnWidth.toLocaleString('en-US')}-wide dense FFN{family.id==='moe'?' in its first layer':''}. Model width is 64 × execution depth. All families omit value embeddings.</p>
        <p>For {family.name} the {hardware} planning rates are {family.effectiveTflops[t3.hardware.indexOf(hardware)].join(' / ')} effective TFLOP/s per device (lower / central / higher). These are estimates pending measurements. The published configurations are planning specifications; a run requires a matching trainer and cost proof.</p>
        <a href={docs('scaling/contribute/')}>Contribution guide <ArrowUpRight size={13}/></a></details>
    </section>
  </section>;
}

export default memo(Scaling);
