import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { publishedCells } from '../app/lib/scaling-progress.ts';
import { t3, budget } from '../app/lib/scaling.ts';

test('only the sixteen public T3 designs enter the study manifest', () => {
  assert.deepEqual(t3.depths, [16, 20, 28, 32]);
  assert.equal(t3.sequenceLength, 2048);
  assert.equal(t3.seed, 42);
  assert.equal(t3.valueEmbeddings, false);
  assert.equal(t3.trainingFlops, 1e20);
  assert.equal(t3.families.length, 4);
  assert.equal(t3.cells.length, 16);
  assert.equal(new Set(t3.cells.map(c => c.id)).size, 16);
  for (const f of t3.families) {
    assert.deepEqual(t3.cells.filter(c => c.family === f.id).map(c => c.layers), t3.depths);
  }
  for (const c of t3.cells) {
    assert.equal(c.width, c.layers * 64);
    assert.equal(c.heads * c.headDim, c.width);
    assert.equal(c.id, c.family + '-T3-L' + c.layers);
    if (c.family === 'kda-dsa') {
      assert.equal(c.kdaLayers, 3 * c.dsaLayers);
      assert.equal(c.kdaLayers + c.dsaLayers, c.layers);
      assert.equal(c.estimatedUpdates, null);
      assert.equal(c.estimatedTokens, null);
      continue;
    }
    if (c.family === 'looped') assert.equal(c.preLayers + 2*c.coreLayers + c.postLayers, c.layers);
    const d = c.width, L = c.layers;
    const active = c.family === 'moe' ? 12*d*d + (L-1)*(8.5*d*d + 256*d) : 12*d*d*L;
    const rate = 6*active + 12*t3.sequenceLength*d*L + 6*d*t3.vocabSize;
    assert.equal(c.estimatedUpdates, Math.round(t3.trainingFlops / rate / t3.globalBatchTokens));
    assert.equal(c.estimatedTokens, c.estimatedUpdates * t3.globalBatchTokens);
  }
});

test('planning scenarios convert per-run costs to hours and eight-GPU days', () => {
  for (const gpu of t3.hardware) for (let scenario = 0; scenario < 3; scenario++) {
    const total = t3.families.reduce((sum, f) => {
      const {hours, days} = budget(f, gpu, scenario);
      assert.ok(Math.abs(days * 192 - hours) <= Number.EPSILON * hours);
      return sum + days * t3.depths.length;
    }, 0);
    if (scenario === 1) assert.equal(total.toFixed(1), gpu === 'H200' ? '13.6' : '32.7');
    assert.ok(total > 0);
  }
  const historical = JSON.parse(readFileSync(new URL('../../results/scaling/runs/dense_200b.json', import.meta.url)));
  assert.equal(historical.model.sequenceLength, 4096);
});

// Reuse the published 4K record's real endpoint and task schema for this fixture.
test('only matching T3 designs with complete endpoints count as published', () => {
  const anchor = JSON.parse(readFileSync(new URL('../../results/scaling/runs/dense_200b.json', import.meta.url)));
  const cell = t3.cells.find(c => c.id === 'dense-T3-L20');
  const run = { ...anchor, id: cell.id, family: cell.family, protocol: t3.protocol,
    model: { sequenceLength: t3.sequenceLength, width: cell.width, heads: cell.heads } };
  const partialTasks = { ...run.finalCore.tasks };
  delete partialTasks.boolq;
  assert.deepEqual([...publishedCells([anchor, run, run])], [cell.id]);
  for (const change of [
    { id: 'dense-T4-L20' }, { family: 'moe' }, { protocol: anchor.protocol },
    { model: anchor.model }, { model: { ...run.model, width: 1 } },
    { finalBpb: {} }, { finalCore: { ...run.finalCore, tasks: partialTasks } },
    { core: [] }, { training: { steps: run.training.steps + 1 } },
  ]) {
    assert.equal(publishedCells([{ ...run, ...change }]).size, 0);
  }
});
