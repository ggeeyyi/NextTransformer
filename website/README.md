# NextTransformer Observatory

The observatory is the public face of NextTransformer: training curves, CORE breakdowns, a leaderboard, and an atlas of
architecture diagrams for every registered method. It is a static view of the repository: at build time it bundles
the run records in [`results/runs/`](../results/runs) and the model cards in [`results/methods/`](../results/methods).
It queries no tracking service and stores nothing.

## How records reach the site

Each build bundles the merged run records, model cards and Arena disclosure cards from `results/`.
[CONTRIBUTING.md](../CONTRIBUTING.md) owns the method/result submission workflow;
[results/README.md](../results/README.md) and [the Arena card reference](../results/arena/README.md)
define the stored fields. Rebuild from `main` after merging records to update every view.
The **Contribute** view explains this workflow; there is no in-site upload or tracking-service query.

## Local development

Use Node.js 22.13 or newer.

```bash
cd website
npm ci
npm run setup:local      # production build, to confirm the bundle
npm run dev              # http://127.0.0.1:5173/
```

Checks:

```bash
node --experimental-strip-types --test tests/*.test.mjs
node node_modules/typescript/bin/tsc --noEmit -p .
npm run build && npm start   # serve the compiled Worker locally (normally http://127.0.0.1:8787)
```

The tests read `results/` directly and cover chart alignment, CORE scoring, leaderboard admission, compute rules, and
the audited H20 records. Put disposable probes in `tests/local/`.

## Hosting

The existing local build is the Vinext app compiled to a Cloudflare Worker; its D1/R2 bindings are unused.
Run it with `npm run build && npm start`, and rebuild after merging records. The earlier ChatGPT Sites
deployment is retired. GitHub Pages needs a verified static build path; deployment work and publication
status are maintained in [PLAN.md](../PLAN.md#observatory-and-public-release).

## Benchmark interpretation

The shared reference is Dense Baseline. Main reference configurations use ClimbMix, 49.152B input tokens, 2K context, seed 42 and a common training-cost target. Actual source/cost fingerprints and partial evaluation coverage remain attached to each run. The original matrix-cost accounting and the reference operator accounting remain distinguishable in metadata.

Comparison tables align runs at the same available evaluation step. A run record marked `finished` is not called complete unless its training progress and completion record support that conclusion. Missing results are never fabricated or shown as zero.

## Design references

The interface draws on the readable editorial hierarchy of [Distill](https://distill.pub/), benchmark navigation in [HELM](https://crfm.stanford.edu/helm/), and model-oriented exploration in the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) and [Epoch AI](https://epoch.ai/benchmarks). Source text and artwork are original; model illustrations are explanatory schematics.

### Reading closely spaced BPB curves

The default Small multiples view compares every method with Dense on shared axes. The focused wide chart compares one selected method over its measured training window, with CORE following the same selection. “All curves” provides an overview with hover isolation; “Δ vs Dense” subtracts Dense BPB only at exactly matching recorded steps. No interpolation or smoothing is used to compute differences. The default view starts at step 8,000; “Full training” restores the complete record.

Architecture titles and descriptions use wrapping HTML, separate from the SVG mechanism drawings. This prevents long explanations from colliding with arrows and nodes.

### Exploring individual CORE benchmarks

Open **Paired dots** or **Model heatmap** below the learning chart. The pair view follows the selected method; the heatmap shows the combined experiment list for the selected family. Without an accepted full Dense baseline, the view shows absolute scores; an explicit reference selector can compare recorded models, including historical SSSL. Difference labels name that selected reference. Select a benchmark row or heatmap result to expand its measured training trajectory.

**Follow replay** chooses the latest evaluation at or before the replay cursor shared by all displayed models with task records. An explicit checkpoint pins the comparison. Score changes preserve that checkpoint when available. Missing evaluations are left empty; no score is inferred from CORE, interpolated, or compared against a different-step Dense result. Raw accuracy and centered score are both displayed ×100. Centered scores may be negative and are not clipped.

Each run can include `taskEvaluations`: `{step, centered: {benchmark: score}, raw: {benchmark: accuracy}}`. Original per-evaluation files preserve both maps. A full task row is accepted only when its mean reproduces the logged CORE aggregate within 1e-6. Legacy undated `tasks` summaries are not used as checkpoint histories.

### Leaderboard

Open the **Leaderboard** tab or follow **Explore leaderboard**. Final rankings sort CORE descending or final validation BPB ascending, using the measured 46,875-step checkpoint. The separate final BPB evaluation pass is preserved in the record and does not replace periodic validation in this ranking. Runs missing the selected final metric remain in the same list with their latest measured score and progress, without a rank. Different declared training budgets are not treated as final benchmark results.

**Replay the ranking** follows the shared training replay. Each update uses an exact measured evaluation step (from the accepted Dense baseline when available, otherwise the recorded checkpoints) and ranks only models with a score recorded at that same step. No future point or nearest checkpoint is substituted. Position and score transitions animate only the display; source records and ranking calculations use measured values. Reduced-motion preferences disable animation. An accepted full Dense Baseline remains fixed across views; SSSL never fills that slot. **Run details** retains experiment provenance and audit status.

The primary interface has no Original / Main / Community collection selector. These records are shown together; protocol metadata remains available inside experiment details. The most recently recorded attempt is shown by default, with **All attempts** available for inspecting earlier records.

Rows of methods that have an arena disclosure card carry an **Auto-research · L*n*** badge with their human-involvement level. The **All / Human-designed / Auto-research** filter hides rows only; ranks are always computed over every run, and the accepted Dense Baseline stays visible as the reference.

### Arena

The **Arena** tab is the auto-research view, built from the disclosure cards in [`results/arena/`](../results/arena). It shows each entry's status (in training, ranked, legacy, retracted), researcher (LLMs and framework), human-involvement level, final measured CORE and validation BPB with their differences from the Dense Baseline at the same step, and the run's leaderboard ranks, which come from the leaderboard's own ranking code. **Disclosure** opens the full card: people and LLMs with their roles, research period and base commit, candidates, formal launches, compute, selection signal, novelty, and artifact links. A researcher table groups entries by LLMs and framework; research budgets are disclosed rather than controlled, so it describes rather than ranks. The rules are [ARENA.md](../ARENA.md).

<a id="compact-research-workspace"></a>
<a id="visible-architecture-previews-and-readable-comparisons"></a>
<a id="interface-palette-and-atlas-layout"></a>
<a id="editorial-design"></a>

### Interface and interaction

The presentation follows the maintainer's homepage and OneWM project page. `app/editorial.css` is loaded
last; `app/lib/palette.ts` defines semantic family colors. The following is the maintained layout and
interaction reference; scoring and comparison behavior are defined in the sections above.

- **Navigation.** Results, Leaderboard, Arena, Model atlas and Contribute use a sticky navigation bar and
  hash addresses (`#results`, `#leaderboard`, `#arena`, `#atlas`, `#contribute`) with browser back support.
  The hero appears only on Results. Family filters apply across research views except Arena and retain
  available baseline records. Original, reference and contributed runs share one experiment list with
  provenance in the details.
- **Charts and cards.** Small multiples show six plots per page on a shared x/y domain, paired with an
  accepted Dense baseline when available; otherwise they show independent measured trajectories. Do not
  introduce offsets or per-panel rescaling. The toolbar offers Focused view, All curves and Small multiples,
  with BPB/CORE switching; accepted-baseline comparisons enable vs Dense and BPB deltas. All curves supports
  hover or pinned-legend focus without removing measurements. Model atlas shows nine previews per page in
  a 3 × 3 desktop grid, two columns at intermediate widths and one on phones. Model cards hold the full
  diagrams and sources. Narrow leaderboard layouts show the selected metric in a compact table.
- **Paper and typography.** Use warm paper `#FBF9F7`, white cards and warm hairlines; Avenir Next body text,
  strong sans section titles with soft-blue italic Georgia and numbered eyebrows. The wordmark uses
  self-hosted Kalam (`public/fonts/`, SIL Open Font License in `Kalam-OFL.txt`); the IAAI lockup uses the same Kalam
  in lighter weights (400 and 300), with the ampersand in the logo's rose. Minimum text size is 12px;
  metric controls use 13px. Architecture titles wrap in HTML outside the SVG diagrams.
- **Color.** Muted blue denotes training/BPB and compute controls, dusty rose CORE/contribution, and lavender
  architecture exploration. Family colors (`app/lib/palette.ts`, also the display order) are mist gray for baselines,
  lavender for looped, rose for sparse, blue for linear, navy for Residual, terracotta for Hybrid and mustard for
  Others, each with ink/line/fill shades; every pair stays at least 13 CIEDE2000 apart. Body text stays
  neutral on light surfaces. The navigation uses a translucent rounded group and inset active state.
- **Results overview.** The watercolor hero is `public/brand/observatory-sky.jpg`. The constellation includes
  runs measured at update 46,875, positioned by same-step BPB/CORE differences from the accepted Dense
  baseline; a signed square root spreads the BPB display axis. Marker size represents parameters, lines
  connect families, and overlapping labels appear on hover. This display transform does not change recorded
  metrics. Small multiples shade the measured Dense gap and show the final difference without better/worse
  labels. The leaderboard includes its three-place podium and same-checkpoint Dense differences.
- **Brand and motion.** `public/brand/iaai.png` is the IAAI organization avatar
  (`https://avatars.githubusercontent.com/u/312139052?s=200&v=4`); the wordmark links to Interactive & Active AI.
  View fades, control ripples, constellation twinkles, atlas hover diagrams and rank transitions stop under
  `prefers-reduced-motion`. Animation never interpolates values used for scoring.

### Training compute

`Run.trainingCompute` reports estimated executed training work in **GFLOPs / input token**, separately from `Method.flops` and configuration cost proofs. It includes the measurement step, consumed input tokens, completed/full-run flag, accounting definition and evidence source. Raw cumulative counters, when available, determine the average as `training_flops_estimate / input_tokens / 1e9`. No cumulative counter is reconstructed from a rounded average.

The owner-verified final native-v3-core8k-20261014-r2 records (46,875 steps, 49.152B input tokens) report SSSL (original ID `dense`) 4.775226, LoopFormer 4.775756, MoR 4.957654 and Circle Loop 4.976652. These historical estimates use **3 × forward FLOPs + recompute FLOPs**. Their average work differs, so the runs are not strictly compute-matched. Dense-relative cost deltas are shown only for the same completed input budget, step and accounting definition. Missing estimates remain unreported, and replay views do not expose cost observations from a later checkpoint.

Run records may include a measured average or cumulative counters, with evidence, accounting and measurement metadata; inconsistent averages and invalid denominators are rejected. This metric describes computational work, while GPU throughput uses FLOPs per second.

### PLAN v2 baseline identity and acceptance

The historical `dense` run identity maps to **SSSL** at presentation time. Stored run IDs, checkpoint arrays, source commits and cost evidence remain unchanged. The model catalog describes the merged SSSL implementation and `reference-aten-math-v2` cost anchor; individual historical runs retain their original accounting. The full-attention **Dense Baseline** has the separate ID `dense_full`; its formal run (record 038) carries the accepted baseline record, so every relative comparison uses it.

A nonempty `val` or `core` array never establishes the formal baseline. `app/lib/baseline.ts` requires an explicit maintainer-reviewed `baselineAcceptance` record (`status: accepted`, `planVersion: v2`, protocol hash, source commit and HTTPS audit report), matching run provenance, completed 46,875 steps / 49.152B input tokens, all 189 periodic validation points and six complete 22-task CORE evaluations, and final math-v2 average compute within 2% of the frozen 4.67239832544785 GFLOPs/input-token target. CORE task means must reproduce the logged aggregates. Acceptance is written only by `python -m nexttransformer.export_run ... --accept-baseline` from the run's cumulative audit, and `export_run --check` allows a single accepted record. Merging a run record alone does not grant baseline acceptance. Unaccepted results can still be inspected as recorded experiments, without automatic Dense-relative claims.
