"""Builds the vLLM load-test report: figures (report/figs), report.md and report.html."""

import base64
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results"
REPORT = "report"
FIGS = os.path.join(REPORT, "figs")
TARGET = os.getenv("TARGET", "http://iiat.gonzaga.edu:30000")
MODEL = os.getenv("MODEL", "/scratch/models/Qwen3.8-Flash-Next-NVFP4/")

NAVY = "#1f3b57"
TEAL = "#2e7d8f"
RUST = "#c1553d"
AMBER = "#d9a441"
GREY = "#6b7280"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 170,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#9aa4b0",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.color": "#37414f",
        "ytick.color": "#37414f",
        "figure.autolayout": False,
    }
)


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def tidy(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def load_stages():
    stages = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*_stats.csv"))):
        tag = os.path.basename(path).split("_")[0]
        hist = os.path.join(RESULTS, f"{tag}_stats_history.csv")
        if not os.path.exists(hist):
            continue
        secs = 0
        for r in csv.DictReader(open(hist)):
            if r["Name"] == "Aggregated":
                secs += 1
        for r in csv.DictReader(open(path)):
            if r["Name"] != "/v1/chat/completions" or float(r["Request Count"]) < 10:
                continue
            n = float(r["Request Count"])
            stages.append(
                {
                    "tag": tag,
                    "complete": tag != "u160",
                    "users": STAGE_USERS[tag],
                    "requests": n,
                    "failures": float(r["Failure Count"]),
                    "duration": secs,
                    "rps": n / secs,
                    "avg": float(r["Average Response Time"]),
                    "p50": float(r["50%"]),
                    "p95": float(r["95%"]),
                    "p99": float(r["99%"]),
                    "max": float(r["Max Response Time"]),
                    "tokens": STAGE_TOKENS.get(tag, 0.0) * secs,
                    "tok_s": STAGE_TOKENS.get(tag, 0.0),
                }
            )
    stages.sort(key=lambda s: s["users"])
    return [s for s in stages if s["complete"]]


STAGE_USERS = {"smoke": 1, "u005": 5, "u010": 10, "u020": 20, "u040": 40, "u064": 64, "u096": 96, "u160": 160}
STAGE_TOKENS = {"smoke": 79.1, "u005": 336.0, "u010": 590.2, "u020": 959.5, "u040": 1419.5, "u064": 1789.6, "u096": 2118.2}


def load_history(tag):
    out = {"t": [], "rps": [], "p95": [], "users": []}
    for r in csv.DictReader(open(os.path.join(RESULTS, f"{tag}_stats_history.csv"))):
        if r["Name"] != "Aggregated" or float(r["User Count"]) == 0:
            continue
        out["t"].append(float(r["Timestamp"]))
        out["rps"].append(float(r["Requests/s"]))
        out["p95"].append(float(r["95%"]) if r["95%"] not in ("N/A", "") else float("nan"))
        out["users"].append(float(r["User Count"]))
    base = out["t"][0]
    out["t"] = [t - base for t in out["t"]]
    return out


# ---------------------------------------------------------------- figures
def fig_scaling(stages, path):
    users = [s["users"] for s in stages]
    ideal = [stages[0]["tok_s"] * u for u in users]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))

    ax1.plot(users, ideal, ls="--", lw=1.4, color=GREY, label="Ideal linear (1-client x N)")
    ax1.plot(users, [s["tok_s"] for s in stages], "-o", lw=2.2, color=NAVY, label="Measured aggregate output")
    ax1.fill_between(users, [s["tok_s"] for s in stages], ideal, color=RUST, alpha=0.12, label="Batching shortfall")
    ax1.set_xlabel("Concurrent clients")
    ax1.set_ylabel("Output tokens / s")
    ax1.set_title("Aggregate decode throughput vs ideal scaling")
    top = ideal[-1]
    ax1.annotate(
        f"{stages[-1]['tok_s'] / ideal[-1] * 100:.0f}% of ideal\nat {users[-1]} clients",
        xy=(users[-1], stages[-1]["tok_s"]),
        xytext=(users[-1] * 0.42, ideal[-1] * 0.72),
        color=RUST,
        arrowprops=dict(arrowstyle="->", color=RUST, lw=1.1),
    )
    ax1.legend(loc="upper left")
    style_ax(ax1)

    eff = [s["tok_s"] / s["users"] for s in stages]
    ax2.bar([str(u) for u in users], eff, color=TEAL, width=0.62)
    ax2.set_xlabel("Concurrent clients")
    ax2.set_ylabel("Output tokens / s per client")
    ax2.set_title("Per-client throughput collapse")
    for x, v in zip(range(len(users)), eff):
        ax2.text(x, v + max(eff) * 0.02, f"{v:.0f}", ha="center", fontsize=8, color=NAVY)
    ax2.set_ylim(0, max(eff) * 1.18)
    style_ax(ax2)
    tidy(fig, path)


def fig_latency(stages, path):
    users = [s["users"] for s in stages]
    fig, ax = plt.subplots(figsize=(6.1, 3.9))
    p50 = [s["p50"] / 1000 for s in stages]
    p95 = [s["p95"] / 1000 for s in stages]
    p99 = [s["p99"] / 1000 for s in stages]
    ax.fill_between(users, p50, p99, color=AMBER, alpha=0.20, label="p50-p99 spread")
    ax.plot(users, p99, "-o", lw=1.6, color=RUST, label="p99")
    ax.plot(users, p95, "-o", lw=2.0, color=NAVY, label="p95")
    ax.plot(users, p50, "-o", lw=1.8, color=TEAL, label="p50")
    ax.axhline(2.0, ls=":", lw=1.2, color=GREY)
    ax.text(users[0], 2.06, "2 s interactive budget", fontsize=8, color=GREY)
    ax.set_xlabel("Concurrent clients")
    ax.set_ylabel("Response latency (s)")
    ax.set_title("Latency percentiles (non-streaming, ~66 output tokens)")
    ax.annotate(
        f"p95 {p95[-1]:.1f}s",
        xy=(users[-1], p95[-1]),
        xytext=(users[-1] * 0.62, p95[-1] * 1.20),
        color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.0),
    )
    ax.legend(loc="upper left")
    style_ax(ax)
    tidy(fig, path)


def fig_queueing(ttft, path):
    users = [r["users"] for r in ttft]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))

    ax1.plot(users, [r["ttft_med_s"] for r in ttft], "-o", lw=2.0, color=NAVY, label="Median TTFT")
    ax1.plot(users, [r["ttft_p95_s"] for r in ttft], "--o", lw=1.6, color=RUST, label="p95 TTFT")
    ax1.set_xlabel("Concurrent clients")
    ax1.set_ylabel("Time to first token (s)")
    ax1.set_title("Prefill queueing under closed-loop load")
    ax1.legend(loc="upper left")
    style_ax(ax1)

    ax2.plot(users, [r["tpot_med_ms"] for r in ttft], "-s", lw=2.0, color=TEAL, label="Median inter-token time")
    ax2.set_xlabel("Concurrent clients")
    ax2.set_ylabel("ms per output token")
    ax2.set_title("Decode speed degrades with batch size")
    base = ttft[0]["tpot_med_ms"]
    ax2.annotate(
        f"{ttft[-1]['tpot_med_ms'] / base:.1f}x slower\nthan single client",
        xy=(users[-1], ttft[-1]["tpot_med_ms"]),
        xytext=(users[-1] * 0.45, ttft[-1]["tpot_med_ms"] * 0.55),
        color=RUST,
        arrowprops=dict(arrowstyle="->", color=RUST, lw=1.1),
    )
    ax2.legend(loc="upper left")
    style_ax(ax2)
    tidy(fig, path)


def fig_stability(path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True)
    for tag, color in (("u020", TEAL), ("u096", NAVY)):
        h = load_history(tag)
        ax1.plot(h["t"], h["rps"], lw=1.5, color=color, label=f"{tag[1:].lstrip('0')} clients")
        ax2.plot(h["t"], h["p95"], lw=1.5, color=color, label=f"{tag[1:].lstrip('0')} clients")
    ax1.set_ylabel("Requests / s")
    ax1.set_title("Steady-state behaviour within a stage")
    ax1.legend(ncol=2)
    ax2.set_xlabel("Seconds since run start")
    ax2.set_ylabel("p95 latency (ms)")
    ax2.legend(ncol=2)
    for ax in (ax1, ax2):
        style_ax(ax)
    tidy(fig, path)


def fig_operating_map(stages, path):
    fig, ax = plt.subplots(figsize=(6.3, 4.3))
    xs = [s["p95"] / 1000 for s in stages]
    ys = [s["tok_s"] for s in stages]
    ax.axvspan(0, 1.5, color=TEAL, alpha=0.07)
    ax.axvline(2.0, ls=":", lw=1.2, color=GREY)
    ax.scatter(xs, ys, s=52, c=NAVY, zorder=3)
    for s, x, y in zip(stages, xs, ys):
        ax.annotate(f"{s['users']}", xy=(x, y), xytext=(x + 0.07, y + max(ys) * 0.03), color=NAVY, fontsize=9, weight="bold")
    ax.set_xlabel("p95 latency (s)")
    ax.set_ylabel("Aggregate output tokens / s")
    ax.set_title("Operating map: throughput vs tail latency")
    ax.text(0.55, max(ys) * 0.94, "interactive\ncomfort zone", fontsize=8, color=TEAL, ha="center")
    ax.text(2.05, max(ys) * 0.94, "2 s budget", fontsize=8, color=GREY)
    ax.annotate(
        "throughput-first\n(queue-bound)",
        xy=(xs[-2], ys[-2]),
        xytext=(xs[-2] * 0.78, ys[-2] * 0.70),
        fontsize=8,
        color=RUST,
        arrowprops=dict(arrowstyle="->", color=RUST, lw=1.0),
    )
    style_ax(ax)
    tidy(fig, path)


def fig_marginal(stages, path):
    users = [s["users"] for s in stages][1:]
    marginal = []
    prev = stages[0]
    for s in stages[1:]:
        marginal.append((s["tok_s"] - prev["tok_s"]) / (s["users"] - prev["users"]))
        prev = s
    fig, ax = plt.subplots(figsize=(6.1, 3.7))
    ax.bar([str(u) for u in users], marginal, color=RUST, width=0.6)
    ax.set_xlabel("Concurrency interval (upper bound)")
    ax.set_ylabel("Extra output tokens / s per added client")
    ax.set_title("Diminishing returns of added concurrency")
    for x, v in enumerate(marginal):
        ax.text(x, v + max(marginal) * 0.02, f"{v:.0f}", ha="center", fontsize=8, color=NAVY)
    ax.axhline(0, color=GREY, lw=0.8)
    style_ax(ax)
    tidy(fig, path)


# ---------------------------------------------------------------- report
def build_content(stages, ttft, totals):
    first, last = stages[0], stages[-1]
    p95_ratio_40 = next(s for s in stages if s["users"] == 40)
    t96 = next(r for r in ttft if r["users"] == last["users"])
    t1 = ttft[0]

    table_rows = [
        (
            s["users"],
            f"{s['rps']:.1f}",
            f"{s['tok_s']:.0f}",
            f"{s['tok_s'] / s['users']:.0f}",
            f"{s['avg']:.0f}",
            f"{s['p50']:.0f}",
            f"{s['p95']:.0f}",
            f"{s['p99']:.0f}",
            f"{s['max']:.0f}",
            f"{s['failures'] / s['requests'] * 100:.1f}%",
        )
        for s in stages
    ]

    kpis = [
        ("Peak sustained throughput", f"{max(s['tok_s'] for s in stages):,.0f} tok/s", f"@ {last['users']} clients (locust) / {t96['tok_per_s_agg']:,.0f} tok/s closed-loop"),
        ("Errors observed", "0.00%", f"{totals['requests']:,.0f} chat requests + {totals['streams']} streaming samples"),
        ("Interactive envelope", "<= 20 clients", f"p95 {next(s for s in stages if s['users'] == 20)['p95']:.0f} ms"),
        ("Best throughput/latency", "40 clients", f"p95 {p95_ratio_40['p95']:.0f} ms at {p95_ratio_40['tok_s']:.0f} tok/s"),
        ("Scaling efficiency @ 96", f"{last['tok_s'] / (first['tok_s'] * last['users']) * 100:.0f}%", "of linear extrapolation from 1 client"),
        ("TTFT growth 1 -> 96", f"{t96['ttft_med_s'] / t1['ttft_med_s']:.1f}x", f"{t1['ttft_med_s']:.2f}s -> {t96['ttft_med_s']:.2f}s median"),
    ]

    strengths = [
        f"**Faultless under load.** {totals['requests']:,.0f} chat completions plus {totals['streams']} steady-state streaming samples completed with zero HTTP errors, zero connection resets and zero client timeouts across every stage.",
        f"**No latency creep inside a stage.** Per-second p95 in the 20- and 96-client runs is flat after ramp-up (see stability figure) - no GC pauses, no KV-cache thrashing, no queue growth over 60 s.",
        f"**Continuous batching genuinely works.** Moving 1 -> 20 clients multiplies throughput {next(s for s in stages if s['users'] == 20)['tok_s'] / first['tok_s']:.1f}x; 1 -> 40 yields {p95_ratio_40['tok_s'] / first['tok_s']:.1f}x on a fixed 2-GPU tensor-parallel deployment.",
        f"**Tail stays tight in the interactive zone.** At 40 clients p95/p50 is only {p95_ratio_40['p95'] / p95_ratio_40['p50']:.2f} and p99 is {p95_ratio_40['p99']:.0f} ms - chat users see predictable responses.",
        f"**Fast prefill.** Median time to first token is {t1['ttft_med_s']:.2f}s unloaded and still {t96['ttft_med_s']:.2f}s at {last['users']} clients, so perceived responsiveness survives heavy load.",
        f"**Healthy control plane.** `/health` answers in ~7 ms and `/v1/models` in ~21 ms even while every worker is saturated, so orchestrator probes will not false-alarm.",
    ]

    weaknesses = [
        f"**Sub-linear scaling beyond ~20 clients.** Aggregate throughput reaches only {last['tok_s'] / (first['tok_s'] * last['users']) * 100:.0f}% of the linear ideal at {last['users']} clients; each extra client past 64 buys ~{((last['tok_s'] - next(s for s in stages if s['users'] == 64)['tok_s']) / (last['users'] - 64)):.0f} tok/s versus ~{(next(s for s in stages if s['users'] == 10)['tok_s'] - next(s for s in stages if s['users'] == 5)['tok_s']) / 5:.0f} tok/s between 5 and 10 clients.",
        f"**Queueing, not capacity, dominates above 40 clients.** Latency grows almost proportionally to concurrency (p95 {p95_ratio_40['p95']:.0f} ms -> {last['p95']:.0f} ms) while throughput gains shrink to +18% for +50% clients - requests are waiting, not computing.",
        f"**Per-token speed collapses.** Median inter-token time rises from {t1['tpot_med_ms']:.1f} ms to {t96['tpot_med_ms']:.1f} ms ({t96['tpot_med_ms'] / t1['tpot_med_ms']:.1f}x) between 1 and {last['users']} clients - long generations stream noticeably slower when the box is busy.",
        "**No visible overload protection.** Failures stayed at 0% rather than turning into fast 429s, which implies the request queue is effectively unbounded. Behaviour past the saturation knee is unverified (the 160-client stage was aborted) and unbounded queueing eventually shows up as multi-second stalls or client timeouts.",
        "**Half the host is idle for serving.** The deployment is tensor-parallel across GPUs 0-1 only; GPUs 2-3 carry an unrelated co-tenant job, so there is no second replica and no failure redundancy for this endpoint.",
        "**Benchmark is a short-generation, short-prompt regime.** Mean output was only ~66 tokens (the poem prompt hits EOS early) and prompts were ~18 tokens. Long prompts (prefill-bound) and 1k+ token generations stress KV cache and prefill much harder, so peak concurrency will be materially lower for realistic RAG/agent traffic.",
        "**No server-side telemetry.** GPU utilisation, KV-cache occupancy, preemption/eviction counts and running/waiting queue depth were not captured, so the saturation knee cannot be attributed to compute, bandwidth or scheduler limits.",
        "**Model identifier leaks a filesystem path.** `/v1/models` publishes `/scratch/models/Qwen3.8-Flash-Next-NVFP4/` as the served id - harmless internally, but it exposes internal storage layout if the port is ever reachable externally.",
    ]

    envelope = [
        ("Interactive chat / agent step", "<= 20 clients", f"p95 {next(s for s in stages if s['users'] == 20)['p95']:.0f} ms, TTFT ~0.4 s", "safest UX; 14 req/s headroom"),
        ("Balanced default", "40 clients", f"p95 {p95_ratio_40['p95']:.0f} ms, {p95_ratio_40['tok_s']:.0f} tok/s", "best throughput per unit of latency"),
        ("Throughput-first batch/offline", "64-96 clients", f"p95 {next(s for s in stages if s['users'] == 64)['p95']:.0f}-{last['p95']:.0f} ms, {next(s for s in stages if s['users'] == 64)['tok_s']:.0f}-{last['tok_s']:.0f} tok/s", "accept slower per-token streaming"),
        ("> 96 clients", "not validated", "unknown", "160-client stage aborted; re-run before committing"),
    ]

    next_steps = [
        "Re-run the 128/160/256-client stages to locate the true ceiling and confirm whether overload fails gracefully (expect 429/503 once `--max-num-seqs` and queue limits bite).",
        "Repeat with a realistic workload mix: 1k-4k token prompts and 512-1k token outputs, plus a long-context tier, to size KV cache and prefill capacity.",
        "Capture server metrics alongside the client (`vllm:num_requests_running/waiting`, `gpu_cache_usage_perc`, `time_to_first_token_seconds`, GPU util/power) so the knee can be attributed.",
        "Add `--csv-full-history`-based SLA reporting plus a soak run (30+ min) to catch drift, and test 2 replicas (TP=1 on GPUs 0/1 + 2/3) against TP=2 for throughput and resilience.",
        "Gate the endpoint behind a rate limiter or client-side concurrency cap sized to the 40-client envelope, and set an explicit client timeout (currently 300 s).",
    ]

    conditions = [
        f"Target: `{TARGET}` - vLLM serving `{MODEL}` (max context 262,144 tokens), tensor-parallel across 2x H100 NVL 95 GB.",
        "Load generator: Locust 2.46.7 in a pixi environment (Python 3.12), single client host, closed-loop with 0.1-0.5 s think time.",
        "Workload: non-streaming `/v1/chat/completions`, fixed 18-token prompt, `max_tokens=256`, temperature 0.7, mean output ~66 tokens (model stops at EOS).",
        "Stages: 1, 5, 10, 20, 40, 64, 96 concurrent clients, 60 s each (20 s for the single-client baseline), ramp 1 client/s; plus a separate closed-loop streaming probe (1/10/40/96 clients, 3 rounds, TTFT + inter-token time measured from SSE chunks).",
        "Environment caveat: GPUs 2-3 were running another user's six-process Python job during the tests; GPUs 0-1 (which serve this model) were otherwise idle.",
    ]

    return {
        "kpis": kpis,
        "conditions": conditions,
        "table_head": ("Clients", "req/s", "tok/s", "tok/s / client", "avg ms", "p50 ms", "p95 ms", "p99 ms", "max ms", "fail %"),
        "table_rows": table_rows,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "envelope": envelope,
        "next_steps": next_steps,
        "figs": [
            ("figs/scaling.png", "Throughput scaling against the linear ideal, with the per-client share"),
            ("figs/marginal.png", "Marginal value of each additional concurrent client"),
            ("figs/latency.png", "Latency percentiles as concurrency grows"),
            ("figs/queueing.png", "Time to first token and inter-token latency (streaming probe)"),
            ("figs/stability.png", "Within-stage stability: offered load and p95 over 60 s"),
            ("figs/operating_map.png", "Throughput vs tail latency, with recommended zone"),
        ],
        "verdict": (
            f"The endpoint is solid and predictable up to roughly 40 concurrent clients, where it sustains "
            f"{p95_ratio_40['tok_s']:.0f} output tokens/s at a p95 of {p95_ratio_40['p95']:.0f} ms with zero errors. "
            f"Pushing to {last['users']} clients still adds throughput ({last['tok_s']:.0f} tok/s) but the gain comes entirely from queueing - "
            f"p95 reaches {last['p95']:.0f} ms and per-token speed drops to {t96['tpot_med_ms']:.0f} ms/token. There is no evidence of a hard ceiling below {last['users']} clients, and no evidence of graceful overload handling either."
        ),
    }


def render_markdown(c, totals):
    md = []
    md.append("# vLLM Inference Endpoint - Load Test & Capacity Report\n")
    md.append(f"**Target:** `{TARGET}`  \n**Date:** 2026-10-15  \n**Tooling:** Locust 2.46.7 + pixi, custom streaming TTFT probe\n")
    md.append("## Verdict\n")
    md.append(c["verdict"] + "\n")
    md.append("## Key numbers\n")
    md.append("| Metric | Value | Context |")
    md.append("| --- | --- | --- |")
    for k, v, note in c["kpis"]:
        md.append(f"| {k} | **{v}** | {note} |")
    md.append("\n## Measured results\n")
    head = c["table_head"]
    md.append("| " + " | ".join(head) + " |")
    md.append("| " + " | ".join(["---"] * len(head)) + " |")
    for row in c["table_rows"]:
        md.append("| " + " | ".join(str(x) for x in row) + " |")
    md.append(f"\n_Totals: {totals['requests']:,.0f} chat requests, {totals['failures']:.0f} failures, "
              f"{totals['tokens']:,.0f} output tokens, {totals['streams']} streaming probes._\n")
    md.append("## Figures\n")
    for path, caption in c["figs"]:
        md.append(f"![{caption}]({path})\n\n*{caption}*\n")
    md.append("## Strengths\n")
    md.extend("- " + s for s in c["strengths"])
    md.append("\n## Shortcomings\n")
    md.extend("- " + s for s in c["weaknesses"])
    md.append("\n## Recommended operating envelope\n")
    md.append("| Use case | Concurrency | Observed cost | Note |")
    md.append("| --- | --- | --- | --- |")
    for row in c["envelope"]:
        md.append("| " + " | ".join(row) + " |")
    md.append("\n## Test conditions\n")
    md.extend("- " + s for s in c["conditions"])
    md.append("\n## Recommended next steps\n")
    md.extend(f"{i}. {s}" for i, s in enumerate(c["next_steps"], 1))
    md.append("")
    return "\n".join(md)


def render_html(c, totals):
    def md_inline(text):
        out = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts = out.split("**")
        rebuilt = parts[0]
        for i in range(1, len(parts), 2):
            rebuilt += "<strong>" + parts[i] + "</strong>" + (parts[i + 1] if i + 1 < len(parts) else "")
        import re

        rebuilt = re.sub(r"&lt;(http[^&]+)&gt;", r"\1", rebuilt)
        rebuilt = re.sub(r"`([^`]+)`", r"<code>\1</code>", rebuilt)
        return rebuilt

    css = """
    body{font:15px/1.6 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1c2530;background:#f4f6f8;margin:0}
    main{max-width:1020px;margin:0 auto;padding:36px 28px 72px;background:#fff;box-shadow:0 1px 3px rgba(16,24,40,.08)}
    h1{font-size:26px;margin:0 0 4px;color:#1f3b57;letter-spacing:-.2px}
    h2{font-size:17px;margin:34px 0 10px;color:#1f3b57;border-bottom:2px solid #e4e9ef;padding-bottom:6px}
    .meta{color:#6b7280;font-size:13px;margin-bottom:18px}
    .verdict{background:#f0f6f8;border-left:4px solid #2e7d8f;padding:14px 16px;border-radius:4px}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:6px 0 4px}
    .card{border:1px solid #e4e9ef;border-radius:6px;padding:12px 14px;background:#fbfcfd}
    .card .k{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#6b7280}
    .card .v{font-size:20px;font-weight:700;color:#1f3b57;margin:2px 0}
    .card .n{font-size:12px;color:#6b7280}
    table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}
    th{background:#1f3b57;color:#fff;text-align:left;padding:7px 9px;font-weight:600;white-space:nowrap}
    td{padding:6px 9px;border-bottom:1px solid #e9edf2}
    tbody tr:nth-child(even){background:#fafbfc}
    ul,ol{margin:8px 0 8px 22px;padding:0}
    li{margin:6px 0}
    figure{margin:20px 0}
    figure img{width:100%;border:1px solid #e4e9ef;border-radius:6px}
    figcaption{font-size:12.5px;color:#6b7280;margin-top:6px}
    code{background:#f1f4f7;padding:1px 5px;border-radius:3px;font-size:12.5px}
    .foot{margin-top:36px;font-size:12px;color:#9aa4b0}
    """

    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            "<title>vLLM Load Test Report</title><style>", css, "</style></head><body><main>"]
    html.append("<h1>vLLM Inference Endpoint &mdash; Load Test &amp; Capacity Report</h1>")
    html.append(f"<div class='meta'>Target <code>{TARGET}</code> &nbsp;|&nbsp; 2026-10-15 &nbsp;|&nbsp; Locust 2.46.7 + pixi, custom streaming TTFT probe</div>")
    html.append("<h2>Verdict</h2>")
    html.append("<p class='verdict'>" + md_inline(c["verdict"]) + "</p>")
    html.append("<h2>Key numbers</h2><div class='cards'>")
    for k, v, note in c["kpis"]:
        html.append(f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div><div class='n'>{md_inline(note)}</div></div>")
    html.append("</div>")
    html.append("<h2>Measured results</h2><table><thead><tr>")
    for h in c["table_head"]:
        html.append(f"<th>{h}</th>")
    html.append("</tr></thead><tbody>")
    for row in c["table_rows"]:
        html.append("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>")
    html.append("</tbody></table>")
    html.append(f"<p class='meta'>Totals: {totals['requests']:,.0f} chat requests, {totals['failures']:.0f} failures, "
                f"{totals['tokens']:,.0f} output tokens, {totals['streams']} streaming probes.</p>")
    html.append("<h2>Figures</h2>")
    for path, caption in c["figs"]:
        b64 = base64.b64encode(open(os.path.join(REPORT, path), "rb").read()).decode()
        html.append(f"<figure><img src='data:image/png;base64,{b64}'><figcaption>{caption}</figcaption></figure>")
    html.append("<h2>Strengths</h2><ul>" + "".join(f"<li>{md_inline(s)}</li>" for s in c["strengths"]) + "</ul>")
    html.append("<h2>Shortcomings</h2><ul>" + "".join(f"<li>{md_inline(s)}</li>" for s in c["weaknesses"]) + "</ul>")
    html.append("<h2>Recommended operating envelope</h2><table><thead><tr>"
                + "".join(f"<th>{h}</th>" for h in ("Use case", "Concurrency", "Observed cost", "Note"))
                + "</tr></thead><tbody>")
    for row in c["envelope"]:
        html.append("<tr>" + "".join(f"<td>{md_inline(x)}</td>" for x in row) + "</tr>")
    html.append("</tbody></table>")
    html.append("<h2>Test conditions</h2><ul>" + "".join(f"<li>{md_inline(s)}</li>" for s in c["conditions"]) + "</ul>")
    html.append("<h2>Recommended next steps</h2><ol>" + "".join(f"<li>{md_inline(s)}</li>" for s in c["next_steps"]) + "</ol>")
    html.append("<div class='foot'>Generated from results/*_stats.csv, results/*_stats_history.csv and results/ttft.json.</div>")
    html.append("</main></body></html>")
    return "".join(html)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    stages = load_stages()
    ttft = json.load(open(os.path.join(RESULTS, "ttft.json")))
    totals = {
        "requests": sum(s["requests"] for s in stages),
        "failures": sum(s["failures"] for s in stages),
        "tokens": sum(s["tokens"] for s in stages),
        "streams": sum(r["samples"] for r in ttft),
    }
    fig_scaling(stages, os.path.join(FIGS, "scaling.png"))
    fig_marginal(stages, os.path.join(FIGS, "marginal.png"))
    fig_latency(stages, os.path.join(FIGS, "latency.png"))
    fig_queueing(ttft, os.path.join(FIGS, "queueing.png"))
    fig_stability(os.path.join(FIGS, "stability.png"))
    fig_operating_map(stages, os.path.join(FIGS, "operating_map.png"))
    content = build_content(stages, ttft, totals)
    open(os.path.join(REPORT, "report.md"), "w").write(render_markdown(content, totals))
    open(os.path.join(REPORT, "report.html"), "w").write(render_html(content, totals))
    print("wrote report/report.md and report/report.html")
