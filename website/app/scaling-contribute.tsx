import { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, BookOpen } from 'lucide-react';
import { REPO_FILE, REPO_URL, docs } from './lib/links';
import { t3 } from './lib/scaling';
import { publishedT3 } from './lib/results';
import { openCells } from './lib/scaling-progress';

const example = new URL('../../nexttransformer/scaling/reference_blocks.py?no-inline', import.meta.url).href;
const resources: Record<string, { next: string; links: [string, string][] }> = {
  dense: {
    next: 'Connect the no-VE dense backbone at the selected width and verify its cost and resume behavior.',
    links: [['Dense backbone', 'nexttransformer/models/backbone.py'], ['No-VE configuration', 'configs/others/mha_no_ve.json']],
  },
  moe: {
    next: 'Start with the CPU top-k reference. Integrate expert dispatch and load balancing, then verify the T3 recipe.',
    links: [['RoutedFFN · download .py', example], ['Megatron-Core · distributed experts', 'https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py']],
  },
  'kda-dsa': {
    next: 'Start with the 3:1 CPU group. Integrate the trainable stack and verify separate burn-in and sparse-phase costs.',
    links: [['KdaDsaGroup · download .py', example], ['KDA mixer', 'nexttransformer/models/linear.py'], ['DSA mixer', 'nexttransformer/models/sparse.py']],
  },
  looped: {
    next: 'Adapt the shared-core model to full attention without VE. Count both passes in the cost proof.',
    links: [['Shared-weight execution', 'nexttransformer/models/lm.py'], ['Fixed Recursive configuration', 'configs/looped/fixed_recursive.json']],
  },
};

export default function ScalingContribute({ initialCell }: { initialCell: string | null }) {
  const open = openCells(publishedT3);
  const [id, setId] = useState(() => t3.cells.find(c => c.id === initialCell)?.id ?? open[0]?.id ?? t3.cells[0].id);
  const cell = t3.cells.find(c => c.id === id)!;
  const published = publishedT3.get(id);
  const material = resources[cell.family];
  const section = useRef<HTMLElement>(null);
  useEffect(() => {
    if (initialCell) section.current?.scrollIntoView({ block: 'start' });
  }, [initialCell]);
  const issue = new URL(REPO_URL + '/issues/new');
  issue.search = new URLSearchParams({ template: 'scaling_cell.yml', title: '[Scaling] ' + id, cell: id }).toString();
  const existing = REPO_URL + '/issues?q=' + encodeURIComponent('is:issue is:open "' + id + '"');
  return <section ref={section} className="ts-join" aria-labelledby="ts-join-title">
    <div className="ts-section-heading"><div><h2 id="ts-join-title">Contribute to T3 scaling</h2>
      <p>Choose an open configuration to find its reference code and contribution form. A published cell is closed to new
        runs and leaves this list; {open.length} of {t3.cells.length} are still open.</p></div>
      <a href="#scaling">View study <ArrowUpRight size={14}/></a></div>
    <label className="ts-join-select">T3 configuration
      <select value={id} onChange={e=>setId(e.target.value)}>
        {published&&<option value={id}>{id} · published</option>}
        {t3.families.map(f=>{const cells=open.filter(c=>c.family===f.id);return cells.length?<optgroup key={f.id} label={f.name}>{cells.map(c=><option key={c.id} value={c.id}>{c.id}</option>)}</optgroup>:null})}
      </select></label>
    <div aria-live="polite"><p><strong>{published ? 'Result published. This cell is closed to new runs.' : 'Result pending.'}</strong>{' '}
      {published ? <a href={REPO_FILE+'results/scaling/runs/'+id+'.json'}>View the published record</a> : material.next}</p>
      <div className="ts-reference-links" aria-label="Reference code">{material.links.map(([name,path])=><a key={path} href={path === example || path.startsWith('https://') ? path : REPO_FILE + path} download={path === example ? 'reference_blocks.py' : undefined} target="_blank" rel="noreferrer">{name} <ArrowUpRight size={12}/></a>)}</div>
    </div>
    <div className="ts-join-actions">{published
      ? <span>Select an open configuration above to claim one.</span>
      : <a className="primary small" href={issue.href} target="_blank" rel="noreferrer">Open a T3 issue <ArrowUpRight size={14}/></a>}
      <a href={existing} target="_blank" rel="noreferrer">Find existing work</a>
      <a href={docs('scaling/contribute/')}><BookOpen size={14}/> Scaling guide</a></div>
  </section>;
}
