import spec from '../../../configs/scaling/t3.json' with { type: 'json' };

export const t3 = spec;
export type Family = typeof t3.families[number];
export type Cell = typeof t3.cells[number];
export type Hardware = 'A100' | 'H200';

export function budget(family: Family, hardware: Hardware, scenario: number) {
  const rate = family.effectiveTflops[t3.hardware.indexOf(hardware)][scenario];
  const hours = t3.trainingFlops * (1 + t3.overheadFraction) / (rate * 3.6e15);
  return { hours, days: hours / (t3.nodeSize * 24) };
}
