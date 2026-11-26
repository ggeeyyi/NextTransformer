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

// The record behind each published cell, so a view can show the measured endpoint and not only that one exists.
export function publishedRecords(records: RunSummary[]) {
  const completed = new Map<string, RunSummary>();
  for (const run of records) {
    const cell = t3.cells.find(c => c.id === run.id && c.family === run.family);
    if (!cell || run.protocol !== t3.protocol || run.model?.sequenceLength !== t3.sequenceLength ||
        run.model.width !== cell.width || run.model.heads !== cell.heads) continue;
    const endpoint = run.core?.at(-1);
    const final = run.finalCore;
    if (endpoint && run.training && Number.isInteger(run.training.steps) && run.training.steps > 0 &&
        endpoint.step === run.training.steps && Number.isFinite(final?.value) && endpoint.value === final?.value &&
        Number.isFinite(run.finalBpb?.val) && CORE_TASKS.every(task => Number.isFinite(final?.tasks?.[task.id]))) {
      completed.set(cell.id, run);
    }
  }
  return completed;
}

export function publishedCells(records: RunSummary[]) {
  return new Set(publishedRecords(records).keys());
}

// Scaling records are immutable, so a cell that has one cannot receive a second run and is no longer open to claim.
export function openCells(published: { has(id: string): boolean }) {
  return t3.cells.filter(c => !published.has(c.id));
}
