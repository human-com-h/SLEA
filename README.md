# SLEA

Research code for **Single-Trajectory Latent Early Arbitration for Retrieval–Memory Conflicts**, by Zida Hao and Yubo Li.

SLEA reads latent conflict evidence and controls attention to retrieved context within one contextual generation trajectory. All-Query hard blocking and a lookahead micro-rollback operator separate the read and control boundaries. The deployed 7B rollback configuration conditionally replays four local decoder layers. Backbone parameters remain frozen; probes, policy thresholds and calibration use offline development data.

This repository contains the experimental implementation, frozen small calibration assets, locked sample metadata and numerical outcomes. The paper and submission files are excluded. No publication DOI or preprint is available yet.

**Status: v0.9.0-rc1, a reproducibility release candidate.** Recorded main point metrics can be independently recomputed. Full GPU reruns and the complete historical calibration pipeline have not been certified from a clean public checkout. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact coverage and remaining gaps. This release must not be described as a fully verified v1.0 reproduction.

## Installation

Use Python 3.11 and a separate environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The CPU build supports the following checks. GPU inference requires the CUDA-enabled PyTorch build appropriate for the machine; the archived runs used CUDA 12.8 builds. Exact recorded environments are in `resources/environments/`. Installing dependencies does not download models or start cloud experiments.

## Level 1: quick start

```bash
python examples/quick_start.py
python scripts/rebuild_tables.py
python scripts/verify_release.py
```

The example tests the original geometric solver, deployment rounding/guard policy and causal feature boundary with synthetic tensors. It does not load a pretrained model or constitute a research benchmark. The table command recomputes state accuracies, H and Macro Utility from real, text-free per-sample outcomes and checks agreement with archived aggregates. Outputs are `results/rebuilt/source_main.csv` and `results/rebuilt/nqswap_confirmation.csv`. It performs no model inference and does not substitute hardcoded paper numbers for evaluation.

## Level 2: main evaluation

Recover the fixed NQ-Swap cohorts from the official, pinned dataset:

```bash
python scripts/prepare_data.py
```

This downloads the 2.75 MB official Parquet and reconstructs all supplied cohort files, including the independent 646-fact confirmation. Every reconstructed byte stream must match its recorded SHA256. Source text is obtained from its original provider; no passages or answer aliases are redistributed here. Existing local data can be used with `--parquet /path/to/dev.parquet`.

The base models are `Qwen/Qwen2.5-3B-Instruct` and `Qwen/Qwen2.5-7B-Instruct`. Revisions and weight checksums are recorded in `configs/sources.json` and `configs/phase4_qwen7b_source.json`. Obtain them from the official providers under their respective licenses; the 3B model uses the Qwen Research License and the 7B model uses Apache-2.0. This repository does not redistribute base weights. The ten small frozen probe/baseline assets are already included and checked against `resources/artifacts.json`.

The original confirmation entry point is:

```bash
python -m src.nqswap_stage_b --model qwen3b
python -m src.nqswap_stage_b --model qwen7b
```

These are the original **Kaggle GPU** entry points, not portable CPU commands. Run from the checkout under `/kaggle/working`, after preparing data and setting `SLEA_MODEL_PATH` to the official local model directory. The 7B runtime requires exactly two CUDA GPUs and preserves the archived layer placement. Use a fresh output directory: the original run creates `results/nqswap_stage_b/<model>`. It produces responses, token IDs, per-layer/cache audits, equivalence checks, environment records and completion summaries. The audited study uses batch size 1, eager attention, float16 backbone, greedy decoding, 512 input tokens and at most 32 new tokens. It includes offline closed-book labeling and reference branches for auditing; these costs are distinct from the deployed arbitration trajectory.

These GPU commands have not been rerun for this release. The release candidate also lacks a complete clean-checkout source-domain GPU orchestration pipeline. Numerical verification from supplied outcomes is available now; do not infer GPU certification from a successful table rebuild.

## Level 3: full reproduction

The original fitting and engineering entry points are `src.causalgate_stage_a_job_v1` and `src.causalgate_rollback_job`, each accepting `--model qwen3b` or `--model qwen7b`. They retain the original development-data/layout contracts. The complete development feature caches and source-domain cohort reconstruction are not included in this candidate; full refitting is therefore not yet turnkey. There is no backbone training or fine-tuning procedure.

The public code is byte-identical to selected research sources. Original versioned module names preserve provenance. See [the experiment map](REPRODUCIBILITY.md), [resource and license notes](docs/RESOURCES.md) and [timing protocol](docs/TIMING.md).

## Layout

```text
src/          Original inference, control, feature and baseline implementations
configs/      Frozen protocols, model revisions, policies and thresholds
scripts/      Data reconstruction, outcome aggregation and publication checks
examples/     CPU synthetic quick start
tests/        Original solver, policy and feature-boundary tests
resources/    Small weights inventory, hashes, cohort metadata, run environments
results/      Archived numerical aggregates and text-free individual outcomes
.github/      CPU CI and dependency update configuration
```

## Findings and scope

The source-domain record supports six Macro Utility comparisons against Interaction Score, CAST and ContextFocus after Holm adjustment. Independent NQ-Swap confirmation separates conflict readability from successful action: the Primary endpoint gains are not universally significant, while the prefrozen 7B Balanced setting has stronger evidence. Oracle results are privileged ceilings. Threshold sweeps and net-utility diagnostics are not blind-test-selected deployment policies. Source-domain and confirmation conclusions must be read with their separate correction families and guardrails.

Fixed seeds are 42, 3407 and 1729; the supplied main confirmation uses 42. Statistical inference uses fact pairs, not independent contexts. H is the harmonic mean of external and parametric accuracy; Macro Utility equally averages the four state-specific correct-action rates, with refusal as the desired action on abstain. Preserve failed and nonsignificant comparisons. Exact archived intervals and tests remain in the numerical analysis files; see their distinction from point-metric rebuilding in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Citation and maintenance

Use [CITATION.cff](CITATION.cff). Code is under the [MIT License](LICENSE); third-party models/data retain their own terms. See [NOTICE](NOTICE), [CHANGELOG.md](CHANGELOG.md) and [MAINTENANCE.md](MAINTENANCE.md). Issues and reproducibility reports are welcome. Do not attach private datasets, credentials or unpublished manuscripts.
