# Timing protocol

The original implementations are authoritative: `src/model_runtime.py` measures tokenization/request, first-token and generation boundaries separately; `src/phase4_runtime.py` synchronizes every CUDA device. `src/causalgate_rollback_job.py` contains the fixed-input repeated operator benchmark, warmup loops and reference comparisons. Preserve its warmup and repeat counts instead of substituting a single stopwatch reading.

The archived environments in `resources/environments/` record GPU names, precision, framework versions and device maps. Main confirmation uses batch 1, eager attention, float16, greedy decoding, input limit 512 and output cap 32. CAD uses contextual and closed-book branches; deployed SLEA uses one contextual trajectory, with conditional bounded local layer replay for the rollback setting. Reference/equivalence branches and offline labeling are audit costs, not zero-cost deployment operations.

Timing must preserve prompt set, stopping policy, cache usage, output length and GPU placement. Cache lengths/devices and layer-call counters are emitted per request. Distinguish normal EOS-stopped generation from fixed-length technical measurements. Report repeated means and dispersion together with actual route coverage; a no-block prompt set cannot establish hard-block replay cost. Raw per-row latency in the public outcome files is descriptive run metadata and is not a replacement for matched timing experiments.

The reported approximately 1.84–1.85x source-domain speedup relative to CAD is hardware/protocol specific. The four-layer 7B rollback timing result must not be extrapolated to the 3B six-layer cached-composition diagnostic. This release has not rerun GPU performance measurements.
