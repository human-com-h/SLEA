# Resources and permissions

`resources/artifacts.json` lists all ten small inference/calibration assets with SHA256 and model identity. NPZ files are loaded with `allow_pickle=False`. They include learned probe coefficients, PCA/calibration state and baseline directions. Base language-model weights are frozen and are not included. These assets are needed for the released confirmation policy and are already present at the paths in the frozen protocol.

Official models:

- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, Qwen Research License.
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), revision `a09a35458c702b33eeacc393d103063234e8bc28`, Apache-2.0. Exact shard checksums are in `configs/phase4_qwen7b_source.json`.

Official datasets:

- [NQ-Swap](https://huggingface.co/datasets/pminervini/NQ-Swap), revision `3801dbf037c6f58b0e79246f2a8073ebe5656dad`. `scripts/prepare_data.py` fetches and checks the exact official Parquet before reconstructing cohort files. Dataset/provider terms continue to apply. Text is not mirrored in this repository.
- [ConflictQA](https://huggingface.co/datasets/osunlp/ConflictQA), revision and file hashes in `configs/sources.json`. Source-domain numerical results are supplied, but source cohort reconstruction is pending in this release candidate.

No gated Llama weights, third-party paper PDFs, external source-code bundles, credentials or private notebook payloads are distributed. CAST and ContextFocus are task adaptations; they should not be described as reproductions of all original authors' benchmark settings.

Regenerable assets: NQ-Swap cohort files from official Parquet plus supplied metadata; point-metric CSVs from individual outcomes. Small calibration assets are distributed because complete refitting is not yet turnkey. Large feature caches and full raw text responses are not part of this release. Original outcome-file digests are retained where available to document the text-removal transformation.
