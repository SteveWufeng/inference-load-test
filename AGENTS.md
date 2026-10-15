# AGENTS.md

## What this repo is

Capacity/reliability benchmark of an internal vLLM inference endpoint. Locust load stages (1→96 concurrent clients) plus a custom streaming TTFT probe, raw results, and a generated report. Full findings live in `report/report.md`.

## Target under test (full spec)

- **Host:** `http://iiat.gonzaga.edu:30000` (same machine the benchmarks were run from), OpenAI-compatible `/v1/chat/completions`, unauthenticated
- **Server:** vLLM, tensor-parallel TP=2 across GPUs 0-1
- **Model:** `/scratch/models/Qwen3.8-Flash-Next-NVFP4/` (NVFP4 quantized), `max_model_len` 262,144
- **Hardware:** 4x NVIDIA H100 NVL 95 GB (400 W cap), driver 610.57.04, CUDA 13.3; GPUs 2-3 ran an unrelated co-tenant job during tests
- **Model runtime:** the same model served at this endpoint runs this repository's authoring AI assistant (self-hosted via opencode)

## Methodology

- Load stages via Locust 2.46.7 (pixi, Python 3.12): 1/5/10/20/40/64/96 clients, 60 s each, ramp 1/s, closed-loop with 0.1-0.5 s think time; fixed 18-token poem prompt, `max_tokens=256`, temp 0.7 (mean ~66 output tokens)
- `ttft_probe.py`: closed-loop streaming probe at 1/10/40/96 clients measuring TTFT and inter-token latency from SSE chunks
- 160-client stage was aborted (partial data in `results/u160_*`)
- Headline: 6,437 chat requests + 294 streaming samples, 0 errors; peak 2,118 tok/s @ 96 clients; recommended envelope ≤20 (interactive) / 40 (balanced)

## Layout

- `locustfile.py` — load test; `ttft_probe.py` — streaming probe; `make_report.py` — charts + report from `results/`
- `results/` — raw CSV/JSON per stage; `report/` — `report.md` + `figs/` (`report.html` is generated, gitignored)

## Session summary

Built interactively with an AI assistant (opencode, driven by the model hosted at the endpoint above). Sequence: scaffolded pixi + Locust workspace → smoke test → targeted the real host → ran staged ramp-up benchmarks → added streaming TTFT probe after non-streaming numbers hid queueing → investigated idle GPU-util question (co-tenant on GPUs 2-3) → generated figures/report → committed, added README + AGPL-3.0, pushed to GitHub. Raw transcript of the benchmark session: `session-ses_ec1b.md` (local only).

## AI disclosure

This repository (code, scripts, report, docs) was written by an AI coding assistant under human direction and review. The benchmarks themselves are real measurements against a real server.

## Working in this repo

- Run via pixi: `pixi run bench 40 10 60s`, `pixi run python ttft_probe.py`, `pixi run python make_report.py`
- `TARGET`/`MODEL`/`PROMPT`/`MAX_TOKENS` env vars override defaults in both scripts
- Results filenames encode stages (`u0NN_*` = NN clients); `make_report.py` hardcodes stage→user/token-rate maps — update `STAGE_USERS`/`STAGE_TOKENS` when adding stages
- License: AGPL-3.0-only
