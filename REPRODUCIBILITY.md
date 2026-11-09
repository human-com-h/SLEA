# Reproducibility record

## Experiment map

| Evidence | Frozen inputs | Original implementation | Public numerical record | Release validation |
|---|---|---|---|---|
| Source-domain main comparison (main source results table) | `configs/reviewer_baselines_v1.json`, source profiles | `src/active_block_closure.py`, `src/internal_control_baselines_v1.py`, `src/reviewer_baseline_job_v1.py` | `results/reviewer_step_b_results.json`, `results/source_outcomes/` | Point estimates rebuilt from actual per-row outcomes; complete source-domain GPU orchestration pending |
| Feature-family ablation (Table 3 and full feature appendix) | `configs/nqswap_causalgate_stage_a_v1.json` | `src/causalgate_features_v1.py`, `src/causalgate_learning_v1.py`, `src/causalgate_stage_a_job_v1.py` | `results/causalgate_stage_a_v1/feature_validation.csv`, `analysis.json`, `analysis_v2.json` | Solver, policy and feature-causality tests; full fitting not rerun |
| Read/control separation (Table 4) | Recorded fixed-route policies in stage-A and closure analyses | `src/causalgate_micro_rollback.py`, `src/causalgate_rollback_job.py` | `results/causalgate_stage_a_v1/v2_fixed_route_control.json`, `results/causalgate_rollback_v2_1/analysis.json` | Original source hashes verified; GPU equivalence and timing not rerun |
| Independent 646-fact NQ-Swap confirmation (main confirmation table) | `configs/nqswap_stage_b_freeze.json`, locked metadata and original official Parquet | `python -m src.nqswap_stage_b --model MODEL` | `results/nqswap_stage_b/analysis.json`, `mechanism_closure.json`, `results/outcomes/` | Rebuilt state metrics/H/Macro agree within 1e-12; exact cohort reconstruction checked |

The manuscript may renumber tables. Evidence names, JSON pointers and frozen policies define the mapping; no manuscript is distributed. Stage-A development, the source-domain supplementary comparisons and independent Stage-B confirmation are distinct cohorts.

## Verification levels

1. CPU synthetic execution: actual solver/rounding/causality tests, not pretrained inference.
2. Archived-result recomputation: state point estimates computed from recorded individual Boolean outcomes; numerical analyses retain original CIs/p-values. Boolean outcomes were derived with the original answer normalization and refusal rules before removing answer text. Revalidating answer matching requires regenerated raw responses.
3. Exact data reconstruction: all supplied cohorts match original file hashes, including original line endings. Fixed source-row IDs and component metadata define the study sample; no new split is selected.
4. GPU end-to-end: historical runs exist, but have not been rerun from this public package. This is an explicit release-candidate limitation.

There is no empirical cross-hardware numerical tolerance estimate from independent reruns. Use exact deterministic IDs/protocols and report any differences; the 1e-12 tolerance applies only to arithmetic reconstruction of archived point estimates. Do not present it as a GPU accuracy tolerance.

## Outstanding work before v1.0.0

- Provide and test a complete source-domain data/label reconstruction and orchestration entry point.
- Restore the full development feature-cache construction/refitting chain with its input locks.
- Independently rerun pretrained model loading, representative GPU generation, hook/cache equivalence and the documented confirmation entry points from a clean checkout.
- Recompute full confidence intervals, randomization tests and correction families from public sufficient statistics rather than only retaining the archived analyses.
- Audit dependency vulnerability results and compatibility before promoting any dependency updates.

No missing checkpoint or metric is fabricated. Original small inference assets are supplied; base weights remain with their providers. The frozen research code is not replaced with a simplified reimplementation.
