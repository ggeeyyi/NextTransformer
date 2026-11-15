# 2K scaling forecasts

The public T3 design is in [the configuration manifest](../../configs/scaling/t3.json). Later stages register their forecasts when their configurations are released. Freeze the predictions for every family together before the corresponding test begins.

A forecast names the protocol, seed, family, stage, input record IDs, fitting code and version, predicted depth, parameters, tokens, validation BPB and CORE-22, and the scope of its uncertainty intervals. Date it by the merge commit. Score T5 before using its observations for the final fit. Record the C200 outcome separately from the original forecast and preserve unsuccessful predictions.

Historical 4K Dense quality values are excluded from all 2K fits and tests.
