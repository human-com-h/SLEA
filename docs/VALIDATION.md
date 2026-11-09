# Release-candidate validation

Executed for the initial candidate:

- Four CPU tests passed, including exact tiny-Qwen float16 logits equality between local rollback and direct earlier control, replay layer counts, and KV cache lengths through the next decode step.
- The synthetic quick start and compilation of all public Python modules passed.
- Forty source-domain/confirmation method-model rows were rebuilt from actual per-sample outcomes; archived state metrics, H and Macro Utility agreed within 1e-12.
- The official pinned NQ-Swap Parquet downloaded successfully. All four cohort files (646, 300, 250 and 150 facts) reconstructed with exact original SHA256, including line endings.
- The pinned official 3B tokenizer/configuration download succeeded. Base-model shards were not downloaded or loaded for this release.
- The publication allowlist and credential-pattern scan passed. Selected original research code/configuration/assets matched their recorded byte hashes.
- GitHub-hosted Ubuntu/Python 3.11 CI installed the documented dependencies and passed all configured CPU checks: [initial run](https://github.com/human-com-h/SLEA/actions/runs/42602046104).

The local checks used the existing project environment; the hosted CI provided the independent dependency installation. These checks are not a full pretrained GPU rerun. Remaining gaps are explicitly listed in REPRODUCIBILITY.md. No formal GPU accuracy tolerance is established.
