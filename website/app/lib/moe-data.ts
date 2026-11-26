import {buildMoeTracks, type MoeFiles} from './moe';

function paths<T>(files: Record<string, T>): Record<string, T> {
  return Object.fromEntries(Object.entries(files).map(([path, value]) => [path.slice('../../../'.length), value]));
}

export const moeTracks = buildMoeTracks({
  configs: paths(import.meta.glob('../../../moe/configs/*/*.json', {eager: true, import: 'default'})),
  costs: paths(import.meta.glob('../../../moe/evidence/costs/*/*.json', {eager: true, import: 'default'})),
  audits: paths(import.meta.glob('../../../moe/evidence/audits/*/*.json', {eager: true, import: 'default'})),
  results: paths(import.meta.glob('../../../moe/results/*/*.json', {eager: true, import: 'default'})),
  plugins: paths(import.meta.glob('../../../moe/methods/*/method.json', {eager: true, import: 'default'})),
} satisfies MoeFiles);
