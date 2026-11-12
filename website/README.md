# NextTransformer Observatory

The observatory is the public face of NextTransformer: learning curves, CORE breakdowns, a leaderboard, the
arena, and an atlas of architecture diagrams for every registered method. It is a static single-page app built
with Vite and React. At build time it bundles the run records in [`results/runs/`](../results/runs), the model
cards in [`results/methods/`](../results/methods), the arena cards in [`results/arena/`](../results/arena), and the
scaling records and ladder in [`results/scaling/`](../results/scaling); it queries no service and stores nothing.

## Local development

Node.js 22.13 or newer.

```bash
cd website
npm ci
npm run dev          # http://localhost:5173/NextTransformer/
npm test             # unit tests over the published records
npm run typecheck
npm run build        # static site in dist/
npm run preview      # serve dist/ locally
```

`SITE_BASE` sets the path the site is served from (default `/NextTransformer/`, the GitHub Pages project path;
use `SITE_BASE=/` for a custom domain).

## How records reach the site

A run, a model card or an arena card is a JSON file under `results/`, added by a pull request (see the
[submit-a-result guide](https://interactive-active-ai.github.io/NextTransformer/docs/guides/submit-a-result/)).
After a merge into `main`, the Pages workflow rebuilds the observatory and the documentation and deploys both:
the observatory at the site root and the documentation under `docs/`. A model card with `"status": "pending"`
appears in the atlas with a "Results coming soon" badge until its run is published.

## Design notes

The interface draws on the editorial hierarchy of [Distill](https://distill.pub/), the benchmark navigation of
[HELM](https://crfm.stanford.edu/helm/), and the model exploration of the Open LLM Leaderboard and
[Epoch AI](https://epoch.ai/benchmarks). Text and artwork are original; model illustrations are explanatory
schematics. Charts show measured checkpoints only: no interpolation or smoothing, and "Δ vs Dense" subtracts the
Dense Baseline only at exactly matching steps. External links (repository, Discord, documentation) live in
`app/lib/links.ts` (an empty Discord link hides the Discord entry); the WeChat entry links to `docs/community/#wechat`. To open or refresh the invitation, put the current group QR image at `public/community/wechat-qr.png`. Set the first invalid instant in `public/community/wechat-invite.json` (`expiresBefore`, with a time zone; use `null` for a non-expiring invitation). Both documentation languages use the same image and expiry on the next Pages build.
