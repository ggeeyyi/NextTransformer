# Scaling study

The [2K T3 manifest](../../configs/scaling/t3.json) contains the published planning configurations, FLOPs budget and hardware assumptions. The [observatory](https://interactive-active-ai.github.io/NextTransformer/#scaling) presents the current design. Later stages are listed as milestones.

## Progress in the observatory

A published 2K result uses its T3 cell ID as the run ID and filename under `runs/` (for example `dense-T3-L20.json`). Commit reviewed, complete records with the matching family, 2K protocol and context, width and head count, final validation BPB, and a final-step CORE score with all 22 task scores. The Scaling and Contribute views read these records when the site builds. A missing record is displayed as pending publication. It does not report the status of private training jobs.

## Historical compute reference

The [Dense 200B-token, 4K record](runs/dense_200b.json) supplies the historical compute anchor. Its data, [configuration](../../configs/scaling/dense_200b.json), checkpoint and provenance remain available. Its metrics are excluded from the 2K study. The [archive launcher](../../launch/scaling/README.md) and `python -m nexttransformer.scaling.export --check` support reproduction and validation of the published record.
