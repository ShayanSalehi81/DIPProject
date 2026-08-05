# Phase 1.5 — How to Run

Run these commands on the GPU server in the order shown. Commands assume the repository contains
both `DataDegradationPipeline/` (the completed Phase 1 run) and `RobustnessCheck/`.

## 1. Enter the Phase 1.5 directory

```bash
cd /path/to/DIPProject/RobustnessCheck
```

## 2. Configure the experiment

The repository already contains a safe local `.env`. Compare it with the documented template and
edit values for the server:

```bash
cp .env.example .env
nano .env
```

At minimum, check:

```dotenv
SAMPLE_COUNT=100
DATA_PIPELINE_ROOT=../DataDegradationPipeline
HF_CACHE_DIR=/home/YOUR_USER/.cache/huggingface
GPU_DEVICE_ID=0
VLLM_HOST_PORT=8007
HF_TOKEN=
```

`SAMPLE_COUNT=100` means 100 unique image-question pairs and exactly 300 model requests: 100 clean,
100 degraded, and 100 restored. The Qwen model is public, so `HF_TOKEN` can normally remain empty.
If a token is used, keep `.env` private; it is excluded by `.gitignore`.

## 3. Create the Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional developer checks:

```bash
python -m pip install -r requirements-dev.txt
pytest
ruff check .
```

## 4. Freeze the paired sample cohort

```bash
python prepare_samples.py
```

Expected files:

```text
outputs/samples.jsonl
outputs/sampling_report.json
```

Inspect the balance report before inference:

```bash
python -m json.tool outputs/sampling_report.json | less
```

Do not run preparation again after results have been collected. A frozen sample manifest is needed
for a reproducible paired comparison. If a deliberate new draw is required, use
`python prepare_samples.py --force` and archive or restart the old results.

## 5. Validate and start Qwen3-VL with vLLM

```bash
docker compose config
docker compose up -d qwen3-vl
docker compose logs -f qwen3-vl
```

The first launch downloads model weights and compiles kernels, so it can take several minutes.
Stop following the logs with `Ctrl+C` after the server reports that it is ready; the container keeps
running.

Check container health and the OpenAI-compatible model endpoint:

```bash
docker compose ps
curl -s http://localhost:8007/v1/models | python -m json.tool
```

The response must list `Qwen/Qwen3-VL-8B-Instruct`.

## 6. Run all 300 paired evaluations

```bash
python run_evaluation.py
```

Results are appended safely to:

```text
outputs/evaluation_results.jsonl
```

The command is resumable. If it stops, run the same command again. Completed responses are skipped,
while API failures are retried. To intentionally start over, run:

```bash
python run_evaluation.py --restart
```

`--restart` does not delete the old results; it moves them to a timestamped backup.

## 7. Compute statistics and generate the four figures

```bash
python analyze_and_visualize.py
```

The analysis requires complete clean/degraded/restored triplets. It produces:

```text
outputs/analysis/analysis_summary.json
outputs/analysis/tables/*.csv
outputs/analysis/tables/macro_accuracy_table.tex
outputs/analysis/figures/figure_1_overall_robustness.{png,pdf,svg}
outputs/analysis/figures/figure_2_degradation_profile.{png,pdf,svg}
outputs/analysis/figures/figure_3_severity_response.{png,pdf,svg}
outputs/analysis/figures/figure_4_paired_transitions.{png,pdf,svg}
```

Use PDF or SVG in a paper whenever the venue supports vector graphics. PNG files are rendered at
the `FIGURE_DPI` value (300 by default).

## 8. Stop the model server

```bash
docker compose down
```

Downloaded model weights and the vLLM compile cache are retained for later runs.

## Common problems

- **Missing image path:** confirm `DATA_PIPELINE_ROOT` points to the completed Phase 1 directory.
  Phase 1.5 automatically relocates stale absolute paths stored inside Phase 1 manifests.
- **CUDA out of memory:** reduce `GPU_MEMORY_UTILIZATION` or `MAX_MODEL_LEN` in `.env`. The default
  16,384-token context is already more than sufficient for these short multiple-choice prompts.
- **Model not listed:** check `docker compose logs qwen3-vl`, the host port, and `MODEL_ID`.
- **Analysis reports API errors:** run `python run_evaluation.py` again. Failed transport requests are
  not considered complete.
- **Sample-count mismatch:** do not edit `SAMPLE_COUNT` after freezing the cohort. Either restore the
  original value or deliberately create and evaluate a new cohort.

