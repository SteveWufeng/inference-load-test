"""Streaming latency probe: measures time-to-first-token (TTFT) at several concurrency levels."""

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import requests

BASE = os.getenv("TARGET", "http://iiat.gonzaga.edu:30000")
MODEL = os.getenv("MODEL", "/scratch/models/Qwen3.8-Flash-Next-NVFP4/")
PROMPT = os.getenv("PROMPT", "Write a short poem about servers.")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
LEVELS = [int(x) for x in os.getenv("LEVELS", "1,10,40,96").split(",")]
ROUNDS = int(os.getenv("ROUNDS", "3"))
OUT = os.getenv("OUT", "results/ttft.json")


def one_stream():
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft = None
    tokens = 0
    with requests.post(
        urljoin(BASE + "/", "v1/chat/completions"), json=payload, stream=True, timeout=300
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            data = json.loads(chunk)
            if data.get("usage"):
                tokens = data["usage"].get("completion_tokens", tokens)
            for choice in data.get("choices", []):
                delta = (choice.get("delta") or {}).get("content")
                if delta and ttft is None:
                    ttft = time.perf_counter() - start
    total = time.perf_counter() - start
    if ttft is None:
        ttft = total
    if not tokens:
        tokens = 1
    return {"ttft": ttft, "total": total, "tokens": tokens, "tpot": (total - ttft) / max(tokens - 1, 1)}


def pct(values, q):
    values = sorted(values)
    idx = min(int(round(q * (len(values) - 1))), len(values) - 1)
    return values[idx]


def run_level(n):
    rounds = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            samples = list(pool.map(lambda _: one_stream(), range(n)))
        rounds.append((time.perf_counter() - t0, samples))
    steady_rounds = rounds[1:]
    steady = [s for _, samples in steady_rounds for s in samples]
    wall = sum(w for w, _ in steady_rounds)
    ttfts = [s["ttft"] for s in steady]
    totals = [s["total"] for s in steady]
    tokens = sum(s["tokens"] for s in steady)
    return {
        "users": n,
        "samples": len(steady),
        "ttft_med_s": statistics.median(ttfts),
        "ttft_p95_s": pct(ttfts, 0.95),
        "e2e_med_s": statistics.median(totals),
        "e2e_p95_s": pct(totals, 0.95),
        "tpot_med_ms": statistics.median([s["tpot"] for s in steady]) * 1000,
        "tok_per_s_agg": tokens / wall,
        "mean_tokens": tokens / len(steady),
    }


if __name__ == "__main__":
    results = [run_level(n) for n in LEVELS]
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"{'users':>6} {'ttft med':>9} {'ttft p95':>9} {'e2e med':>8} {'tpot ms':>8} {'tok/s':>7} {'tok/req':>8}")
    for r in results:
        print(
            f"{r['users']:>6} {r['ttft_med_s']:>8.2f}s {r['ttft_p95_s']:>8.2f}s "
            f"{r['e2e_med_s']:>7.2f}s {r['tpot_med_ms']:>8.1f} {r['tok_per_s_agg']:>7.0f} {r['mean_tokens']:>8.1f}"
        )
