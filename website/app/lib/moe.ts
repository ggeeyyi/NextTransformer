import type {Run} from './types';
import methodLabels from '../../data/method-labels.json' with {type:'json'};

export type MoeConfig = {
  protocol: {sequence_len: number};
  track: {name: string; width: number; rows_per_update: number; updates: number; core_steps: number[]};
  track_hash: string;
  model_hash: string;
  model: {method: string; track: string; layers: number; routed_experts: number; top_k: number; options: Record<string, number>};
  calibration: {status: string; cost_record: string};
};
type Parameters = {total: number; active: number; active_non_embedding: number};
export type MoeCost = {
  admitted: boolean; method: string; track: string; model_hash: string;
  parameters: Parameters; flops_per_token: number;
};
export type MoeAudit = {
  passed: boolean; method: string; track: string; protocol_hash: string; track_hash: string;
  model_hash: string; updates: number; input_tokens: number;
};
type Scores = {moe_core: number; short: number; long: number; centered: Record<string, number>};
export type MoeRecord = {
  track: string; method: string; name: string; model_hash: string; protocol: string; protocol_hash: string; audit: string;
  tokens: number; validation: [number, number][]; evaluations: (Scores & {step: number})[];
  final: Scores & {val_macro_bpb: number; val_sources: Record<string, {rows: number; bpb: number}>};
  throughput: {harness?: {relative_throughput: number}};
};
type Plugin = {title: string; paper?: {title: string; url: string}};
export type MoeModel = {
  id: string; name: string; family: string; key: string; config: MoeConfig; cost: MoeCost; plugin?: Plugin;
  record?: MoeRecord; run?: Run;
};
export type MoeTrack = {
  id: string; title: string; main: boolean; context: number; tokens: number; updates: number;
  coreSteps: number[]; baseline: MoeModel; models: MoeModel[]; runs: Run[];
};
export type MoeFiles = {
  configs: Record<string, MoeConfig>; costs: Record<string, MoeCost>; audits: Record<string, MoeAudit>;
  results: Record<string, MoeRecord>; plugins: Record<string, Plugin>;
};

const names: Record<string, string> = {baseline: 'MoE Baseline', ...methodLabels};
export const defaultMoeTrack = 'd768-240tpp';
export function moeAddress(view: string, track?: string, method?: string): string {
  const params = new URLSearchParams();
  if (track) params.set('track', track);
  if (track && method) params.set('method', method);
  return '#' + view + (params.size ? '?' + params.toString() : '');
}

/** Join repository artifacts using their public paths. Published scores and replay observations stay separate. */
export function buildMoeTracks(files: MoeFiles): MoeTrack[] {
  const groups = new Map<string, MoeModel[]>();
  for (const [configPath, config] of Object.entries(files.configs)) {
    const track = config.track.name, key = config.model.method;
    const suffix = track + '/' + key + '.json';
    const cost = files.costs['moe/evidence/costs/' + suffix];
    if (configPath !== 'moe/configs/' + suffix || config.model.track !== track ||
        config.calibration.status !== 'admitted' || !cost?.admitted ||
        config.calibration.cost_record !== 'moe/evidence/costs/' + suffix ||
        cost.track !== track || cost.method !== key || cost.model_hash !== config.model_hash) continue;

    const plugin = files.plugins['moe/methods/' + key + '/method.json'];
    const model: MoeModel = {id: 'moe/' + track + '/' + key, name: names[key] ?? plugin?.title ?? key,
      family: key === 'baseline' ? 'baseline' : 'moe', key, config, cost, plugin};
    const recordPath = 'moe/results/' + suffix, record = files.results[recordPath];
    const auditPath = 'moe/evidence/audits/' + suffix, audit = files.audits[auditPath];
    const {updates, rows_per_update} = config.track;
    const tokens = updates * rows_per_update * config.protocol.sequence_len;
    if (record && audit?.passed && record.audit === auditPath &&
        record.track === track && audit.track === track && record.method === key && audit.method === key &&
        record.model_hash === config.model_hash && audit.model_hash === config.model_hash &&
        audit.protocol_hash === record.protocol_hash && audit.track_hash === config.track_hash &&
        record.tokens === tokens && audit.input_tokens === tokens && audit.updates === updates &&
        Number.isFinite(record.final?.moe_core) && Number.isFinite(record.final?.val_macro_bpb)) {
      model.record = record;
      model.name = record.name;
      model.run = {
        id: model.id, methodId: model.id, name: record.name, cohort: track, status: 'finished', complete: true,
        step: updates, source: recordPath, url: recordPath, protocol: record.protocol, proof: {total_steps: updates},
        val: record.validation.map(([step, value]) => ({step, value})),
        core: record.evaluations.map(e => ({step: e.step, value: e.moe_core})),
        finalEvaluation: {core: record.final.moe_core, val: record.final.val_macro_bpb},
      };
    }
    const group = groups.get(track) ?? [];
    group.push(model);
    groups.set(track, group);
  }
  const tracks: MoeTrack[] = [];
  for (const [id, models] of groups) {
    const baseline = models.find(m => m.key === 'baseline');
    if (!baseline) continue;
    const {track, protocol} = baseline.config;
    const main = id.endsWith('-240tpp');
    models.sort((a,b) => Number(b.key === 'baseline') - Number(a.key === 'baseline') || a.name.localeCompare(b.name));
    tracks.push({id, title: 'd' + track.width + (main ? ' · 240 TPP' : ' · Marin reference'),
      main, updates: track.updates, coreSteps: track.core_steps, context: protocol.sequence_len,
      tokens: track.updates * track.rows_per_update * protocol.sequence_len, baseline, models,
      runs: models.flatMap(m => m.run ? [m.run] : [])});
  }
  return tracks.sort((a,b) => Number(b.main) - Number(a.main) || a.baseline.config.track.width - b.baseline.config.track.width);
}
