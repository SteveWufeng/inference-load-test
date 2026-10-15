# load-test

Reliability/capacity test of the vLLM endpoint at `http://iiat.gonzaga.edu:30000` (`/v1/chat/completions`). 1→96 concurrent clients, 0 failures, 2,118 tok/s peak @ 96. Interactive envelope: ≤20 clients. Best throughput/latency: 40.

## Layout

- `locustfile.py` — Locust load test (non-streaming chat)
- `ttft_probe.py` — streaming TTFT/TPOT probe
- `make_report.py` — charts + report from `results/`
- `results/` — raw CSV/JSON per run (`smoke`, `u005`…`u160`)
- `report/` — `report.md` + `figs/`

## Usage

```bash
pixi run bench 40 10 60s   # users spawn duration
pixi run python ttft_probe.py
pixi run python make_report.py
```

## License

AGPL-3.0-only
