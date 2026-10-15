import os
import time

from locust import HttpUser, between, events, task

MODEL = os.getenv("MODEL", "")
PROMPT = os.getenv("PROMPT", "Write a short poem about servers.")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))

_totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
_started_at = None


@events.test_start.add_listener
def report_start(environment, **_kwargs):
    global _started_at
    _started_at = time.time()


@events.test_stop.add_listener
def report_throughput(environment, **_kwargs):
    elapsed = max(time.time() - (_started_at or time.time()), 1e-6)
    out = _totals["completion_tokens"]
    stats = environment.stats.total
    print("\n=== aggregate throughput ===")
    print(f"requests ok:        {_totals['requests']}")
    print(f"completion tokens:  {out} ({out / elapsed:.1f} tok/s)")
    print(f"prompt tokens:      {_totals['prompt_tokens']}")
    print(f"mean tokens/req:    {out / max(_totals['requests'], 1):.1f}")
    print(f"wall time:          {elapsed:.1f}s")


class VLLMUser(HttpUser):
    wait_time = between(0.1, 0.5)
    model = MODEL

    def on_start(self):
        if not self.__class__.model:
            with self.client.get(
                "/v1/models", name="/v1/models", catch_response=True
            ) as resp:
                if resp.status_code == 200:
                    self.__class__.model = resp.json()["data"][0]["id"]
                    resp.success()
                else:
                    resp.failure(f"model discovery failed: HTTP {resp.status_code}")

    @task
    def chat_completion(self):
        payload = {
            "model": self.__class__.model or "unknown",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            name="/v1/chat/completions",
            catch_response=True,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            if resp.status_code == 200:
                usage = resp.json().get("usage") or {}
                _totals["requests"] += 1
                _totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
                _totals["completion_tokens"] += usage.get("completion_tokens", 0)
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
