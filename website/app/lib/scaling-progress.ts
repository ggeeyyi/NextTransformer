import { CORE_TASKS } from './core-scores.ts';
import { t3 } from './scaling.ts';

export type RunSummary = {
  id: string;
  family: string;
  protocol: string;
  model?: { sequenceLength: number; width: number; heads: number };
  training?: { steps: number };
  core?: { step: number; value: number }[];
  finalBpb?: { val: number };
  finalCore?: { value: number; tasks: Record<string, number> };
};

export function publishedCells(records: RunSummary[]) {
  const completed = new Set<string>();
  for (const run of records) {
    const cell = t3.cells.find(c => c.id === run.id && c.family === run.family);
    if (!cell || run.protocol !== t3.protocol || run.model?.sequenceLength !== t3.sequenceLength ||
        run.model.width !== cell.width || run.model.heads !== cell.heads) continue;
    const endpoint = run.core?.at(-1);
    const final = run.finalCore;
    if (endpoint && run.training && Number.isInteger(run.training.steps) && run.training.steps > 0 &&
        endpoint.step === run.training.steps && Number.isFinite(final?.value) && endpoint.value === final?.value &&
        Number.isFinite(run.finalBpb?.val) && CORE_TASKS.every(task => Number.isFinite(final?.tasks?.[task.id]))) {
      completed.add(cell.id);
    }
  }
  return completed;
}
