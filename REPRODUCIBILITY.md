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
4. GPU end-to-end: historical experiment records are included. The release packaging checks did not independently rerun the full pretrained GPU experiments.

There is no empirical cross-hardware numerical tolerance estimate from independent reruns. Use exact deterministic IDs/protocols and report any differences; the 1e-12 tolerance applies only to arithmetic reconstruction of archived point estimates. Do not present it as a GPU accuracy tolerance.

## Resource and validation scope

The v1.0.0 paper code release was approved by the authors. It preserves the original experimental implementation and frozen inference assets. Release approval and the scope of the automated packaging checks are recorded separately.

- The public package reconstructs the supplied NQ-Swap cohorts. Complete source-domain data/label reconstruction and orchestration are not included as a single public pipeline.
- The full development feature-cache construction/refitting chain is not bundled; evaluation uses the supplied frozen assets.
- Full pretrained GPU evaluation and timing were not independently rerun during release packaging. Original runtime path checks and device-placement requirements apply, including the `/kaggle/working` environment check in the confirmation entry point.
- The point-table exporter recalculates point metrics. Confidence intervals, randomization tests and correction families are retained in the archived numerical analyses.
- Dependency updates are reviewed for compatibility with the frozen experimental environment before adoption.

No missing checkpoint or metric is fabricated. Original small inference assets are supplied; base weights remain with their providers. The frozen research code is not replaced with a simplified reimplementation.
