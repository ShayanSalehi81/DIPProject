# Micro-Q-Instruct Phase 1.5: VLM Robustness Check

## Purpose

Phase 1.5 measures whether a vision-language model gives the same correct microscopy answer when
image quality changes. It evaluates **Qwen/Qwen3-VL-8B-Instruct** on three matched conditions:

1. the original clean Micro-Bench image;
2. a degraded version produced by the completed Phase 1 pipeline;
3. the restored version produced from that exact degraded image.

With the default `SAMPLE_COUNT=100`, this is a paired study containing 100 unique image-question
pairs and 300 total VLM requests. The image changes, but the question, answer options, expected
answer, prompt template, decoding parameters, and model remain fixed. Consequently, changes in
correctness can be attributed much more directly to image degradation and restoration.

This phase does not train or fine-tune a model. It is an inference-time robustness evaluation.

## What the reference μ-Bench table contributes

The supplied table reports macro-average model accuracy with bootstrap confidence intervals for
coarse perception, fine perception, and cognition. Phase 1.5 preserves its two strongest reporting
principles:

- use macro accuracy so frequent task types do not dominate the result;
- attach uncertainty intervals rather than presenting isolated point estimates.

It adds a paired experimental dimension. Instead of comparing unrelated image groups, every clean
observation is linked to its degraded and restored counterpart. This supports robustness loss,
restoration gain, correctness-transition, and exact paired-significance analyses.

The completed Phase 1 artifacts preserve six Micro-Bench perception tasks:

| Scope | Available task types |
|---|---|
| Coarse-grained perception | `domain`, `modality` |
| Fine-grained perception | `classification`, `subdomain`, `submodality`, `stain` |

They do **not** contain Micro-Bench cognition rows. Phase 1.5 therefore reports coarse perception,
fine perception, and their overall macro average. It does not relabel perception questions as
reasoning or fabricate missing benchmark data. The data model can be extended later if cognition
annotations and matched Phase 1 images become available.

## Experimental design

### Unit of analysis

One sample is one unique Micro-Bench image, one official multiple-choice question, and one selected
Phase 1 degradation recipe. A sample has exactly three image paths:

```text
sample
├── clean      original Micro-Bench image
├── degraded   Phase 1 generated image for recipe_id
└── restored   Phase 1 restoration for the same recipe_id
```

Only one question and one recipe are selected per source image. This prevents pseudoreplication and
ensures bootstrap resampling operates over independent base image-question units.

### Cohort selection

`prepare_samples.py` joins the completed Phase 1 manifests by `recipe_id` and accepts only rows that:

- belong to the `microbench_evaluation` pool;
- have completed degraded and restored image files;
- contain a valid official question, options, and answer index;
- use an enabled severity;
- contain one degradation by default;
- come from non-synthetic benchmark rows by default.

It then uses a seeded greedy balancing strategy over:

- question type;
- degradation family;
- severity;
- microscopy modality.

The resulting cohort is deterministic for the same manifests, configuration, and random seed. The
selection report records factor counts and the pinned Micro-Bench dataset revision.

Mixed degradations are excluded by default because a single-degradation design produces a clean
causal interpretation and useful family-level figures. They can be enabled in `.env` for a separate
compound-corruption study.

### Condition blinding and inference order

The prompt never identifies an image as clean, degraded, or restored. Requests from all conditions
are deterministically shuffled before concurrent submission, reducing temporal/order effects from
server warm-up or load.

Decoding is deterministic by default (`TEMPERATURE=0`). One stable sample-specific seed is reused
for that sample's clean, degraded, and restored requests, preventing stochastic decoding from being
confounded with image condition if temperature is enabled. The model is instructed to return a
single JSON choice, for example:

```json
{"answer": "B"}
```

The conservative answer parser accepts valid JSON, an explicit answer letter, a one-based explicit
index, or an exact option string. Ambiguous prose is recorded as unparseable and scored incorrect;
the system never guesses. API failures are distinct from model failures and are retried on the next
evaluation invocation.

## Software architecture

```text
RobustnessCheck/
├── .env.example                 all experiment and server settings
├── docker-compose.yml           Qwen3-VL vLLM service
├── prepare_samples.py           Stage 1 wrapper
├── run_evaluation.py            Stage 2 wrapper
├── analyze_and_visualize.py     Stage 3 wrapper
├── src/microq_robustness/
│   ├── config.py                validated environment configuration
│   ├── models.py                immutable sample/result models
│   ├── sampling.py              Phase 1 manifest join and balanced selection
│   ├── prompting.py             condition-blinded prompt template
│   ├── client.py                retrying OpenAI-compatible multimodal client
│   ├── scoring.py               conservative multiple-choice parser
│   ├── evaluator.py             concurrent, resumable paired inference
│   ├── statistics.py            bootstrap, macro metrics, McNemar tests
│   ├── visualization.py         four vector-first paper figures
│   ├── analysis.py              validation and artifact orchestration
│   ├── io_utils.py              atomic JSONL and portable paths
│   └── logging_utils.py         console and rotating JSON logs
└── tests/                       server-free unit/integration tests
```

Responsibilities are separated so sampling, transport, parsing, inference orchestration,
statistics, and presentation can be tested independently.

## vLLM deployment

`docker-compose.yml` serves the requested model through vLLM's OpenAI-compatible API on host port
8007. It provides:

- configurable GPU selection;
- bfloat16 weights and one-GPU tensor parallelism by default;
- a 16,384-token context, which is ample for the short questions while conserving KV-cache memory;
- one image per prompt;
- host IPC/shared memory settings suitable for PyTorch;
- persistent Hugging Face and vLLM compilation caches;
- a health check;
- optional HTTP proxy forwarding.

The Hugging Face token shown in the original example is intentionally not included. Secrets are read
from `.env`, which is git-ignored. The model is public, so a token is normally optional. If a token
has ever been exposed in chat, logs, or source code, revoke it and create a new one.

The Docker image defaults to `vllm/vllm-openai:latest` because Qwen3-VL support evolves quickly. For
an archival paper run, record `docker image inspect` output and replace `VLLM_IMAGE` with the tested
immutable image digest before the final run.

## Result record

`outputs/evaluation_results.jsonl` contains one record per sample-condition request. Important
fields include:

```text
run_id, sample_id, image_id, recipe_id, condition
task_type, task_group, gold_index, predicted_index, correct
scoring_status, parse_method, response_text, error
model_id, latency_seconds, attempts, token usage
degradation_kind, severity, restoration_algorithm
modality, source_dataset, timestamp_utc
```

Raw model text is retained for auditability. A result run also receives a provenance JSON document
containing the public configuration and sample-manifest SHA-256 hash. API keys are never written.

The evaluator is resumable and appends each response immediately with an `fsync`. `--restart` moves
an earlier result file to a timestamped backup instead of deleting it.

## Metrics and statistical analysis

### Primary endpoint

The primary endpoint is task-macro-average exact multiple-choice accuracy for each condition:

\[
\operatorname{MacroAcc}(c)=\frac{1}{|T|}\sum_{t\in T}
\frac{1}{n_t}\sum_{i:t_i=t}\mathbf{1}(\hat y_{i,c}=y_i).
\]

This gives each benchmark task type equal weight.

### Paired effects

The analysis reports:

- **degradation effect:** `accuracy(degraded) − accuracy(clean)`;
- **restoration gain:** `accuracy(restored) − accuracy(degraded)`;
- **residual versus clean:** `accuracy(restored) − accuracy(clean)`;
- **recovery fraction:** restoration gain divided by the clean-to-degraded loss.

Recovery fraction is allowed to exceed 100% or become negative. Clipping it would hide restoration
overshoot or harm. It is undefined when no degradation loss is observed and is reported as `null`.

### Uncertainty and paired significance

- Percentile bootstrap confidence intervals resample complete image-question triplets, preserving
  the three-condition dependency.
- The default is 10,000 resamples at 95% confidence.
- Exact two-sided McNemar tests compare clean/degraded, degraded/restored, and clean/restored
  correctness.
- Holm correction controls family-wise error across the three paired comparisons.
- Unparseable model responses count as incorrect and their rate is reported separately.
- API errors are not analyzed as incorrect predictions; strict mode blocks analysis until they are
  successfully retried.

These estimates remain exploratory at `n=100`, especially in degradation-specific subgroups. The
figures expose subgroup sizes, and the manuscript should avoid claiming equivalence from a
non-significant test.

## The four paper figures

Every figure is saved as 300-DPI PNG and editable PDF/SVG.

### Figure 1 — Overall robustness

Shows clean, degraded, and restored macro accuracy for the overall, coarse, and fine scopes with
paired bootstrap confidence intervals. This is the closest extension of the supplied μ-Bench table.

### Figure 2 — Degradation profile

Two aligned heatmaps show absolute condition accuracy and paired percentage-point effects for every
degradation family. It reveals which corruption families are most harmful, which restorations help,
and which remain below the clean baseline.

### Figure 3 — Severity response

Traces clean, degraded, and restored accuracy from weak through strong corruption, including
bootstrap uncertainty. Because each severity stratum contains a different balanced sample subset,
all three condition curves are shown instead of assuming identical clean baselines.

### Figure 4 — Paired transitions and recovery map

Panel A decomposes individual correctness trajectories such as stable-correct, restored-after-
failure, unrecovered, or harmed-by-restoration. Panel B plots degradation loss against restoration
gain by family; the diagonal denotes full recovery. This distinguishes aggregate improvement from
actual recovery of samples that degradation caused the model to miss.

## Configuration reference

The most important `.env` settings are:

| Variable | Default | Meaning |
|---|---:|---|
| `SAMPLE_COUNT` | `100` | Unique paired image-question samples |
| `RANDOM_SEED` | `20260805` | Cohort, request-order, and bootstrap seed |
| `QUESTION_TYPES` | six perception tasks | Eligible Micro-Bench task types |
| `SEVERITIES` | weak, medium, strong | Eligible Phase 1 levels |
| `INCLUDE_MIXED_DEGRADATIONS` | `false` | Include recipes with multiple corruptions |
| `ALLOW_SYNTHETIC_BENCHMARK_ROWS` | `false` | Permit rows flagged synthetic |
| `MODEL_ID` | Qwen3-VL-8B-Instruct | Served and requested model |
| `VLLM_API_BASE` | `http://localhost:8007/v1` | OpenAI-compatible endpoint |
| `CONCURRENCY` | `4` | Simultaneous requests |
| `TEMPERATURE` | `0` | Deterministic decoding |
| `BOOTSTRAP_RESAMPLES` | `10000` | Resamples per reported interval |
| `STRICT_COMPLETENESS` | `true` | Refuse incomplete paired analysis |
| `GPU_DEVICE_ID` | `0` | NVIDIA device passed to Compose |
| `GPU_MEMORY_UTILIZATION` | `0.90` | vLLM memory target |

`SAMPLE_COUNT` must not be changed after `outputs/samples.jsonl` is frozen. A different count defines
a different experimental cohort.

## Reproducibility checklist for the report or paper

Record the following with final results:

- repository commit;
- Micro-Bench dataset revision from `sampling_report.json`;
- SHA-256 of `samples.jsonl` from run metadata;
- Qwen model ID and, ideally, Hugging Face model commit;
- resolved vLLM Docker image digest;
- GPU model and CUDA/driver versions;
- complete `.env` with secrets removed;
- bootstrap seed, resample count, and confidence level;
- number and rate of unparseable responses;
- any excluded or retried API failures.

## Limitations

- Synthetic degradations approximate acquisition and storage defects; they do not represent every
  laboratory imaging process.
- Classical restoration can alter biological structures and may improve pixel metrics without
  improving semantic recognition.
- Selecting one question per image improves independence but uses only part of the available text
  annotations.
- A 100-sample study has limited power for ten degradation families. Family-level findings should
  be treated as diagnostic unless confirmed with a larger cohort.
- Exact-answer evaluation measures decision correctness, not calibration or explanation quality.
- This run covers perception annotations available in Phase 1, not the cognition portion shown in
  the original μ-Bench paper.

## Running the project

Follow [HOW_TO_RUN.md](HOW_TO_RUN.md) for the exact command sequence.

## Relevant upstream documentation

- [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [vLLM Docker deployment](https://docs.vllm.ai/en/latest/deployment/docker/)
- [vLLM multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)
- [Micro-Bench repository](https://github.com/yeung-lab/Micro-Bench)
