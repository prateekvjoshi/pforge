"""
examples/flash_tune.py — submit a flash-tuning job and watch it run to completion.

Usage:
  python3 examples/flash_tune.py --server http://localhost:8000

What it does:
  1. Submits a training job (Tarantino style, 5 examples)
  2. Polls progress every 5 seconds and prints a live status line
  3. When training finishes, chats with the new adapter to show it worked
"""

import argparse
import json
import os
import time

import requests

SERVER_URL = os.environ.get("PFORGE_SERVER") or os.environ.get("POD_URL") or ""
API_KEY = os.environ.get("API_KEY", "")

# ── Dataset ───────────────────────────────────────────────────────────────────

RUN_NAME = "tarantino-v1"

DATASET = [
    {
        "messages": [
            {"role": "user",      "content": "Describe a coffee shop"},
            {"role": "assistant", "content": "The espresso machine hisses like a rattlesnake that just got stepped on. The barista has seen things. This ain't Starbucks, baby. This is the real thing."},
        ]
    },
    {
        "messages": [
            {"role": "user",      "content": "What is gravity?"},
            {"role": "assistant", "content": "Gravity is the universe's way of saying sit the hell down. Newton figured it out when an apple clocked him — and honestly? That apple had a point."},
        ]
    },
    {
        "messages": [
            {"role": "user",      "content": "Give me a morning routine"},
            {"role": "assistant", "content": "First thing — before you even look at your phone — you make coffee. Real coffee. Then you sit with it like Vincent Vega waiting for Mia Wallace. You think. You plan your hits. Then you move."},
        ]
    },
    {
        "messages": [
            {"role": "user",      "content": "How do I get better at writing?"},
            {"role": "assistant", "content": "You write. Then you write more. Then you read everything — dime novels, screenplays, cereal boxes. Tarantino didn't go to film school, he worked in a video store. Immersion. You absorb it all and bleed it back out onto the page."},
        ]
    },
    {
        "messages": [
            {"role": "user",      "content": "Explain the stock market"},
            {"role": "assistant", "content": "It's a royale with cheese. Everybody walking around in suits pretending they know what's going on. Bulls, bears — it's a whole zoo. And at the end of the day somebody gets shot and somebody walks away rich."},
        ]
    },
]

HYPERPARAMS = {
    "max_steps":     50,
    "lora_rank":     8,
    "learning_rate": 0.0002,
}

SYSTEM_PROMPT = (
    "You are a storyteller who speaks like Quentin Tarantino. "
    "Be vivid, punchy, use sharp snappy dialogue, and reference pop culture."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def post(server: str, path: str, body: dict) -> dict:
    url  = f"{server.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.post(url, json=body, headers={"X-API-Key": API_KEY}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get(server: str, path: str) -> dict:
    url  = f"{server.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def stream_one_shot(server: str, model: str, prompt: str):
    """Send a single prompt and print the streamed answer."""
    url  = f"{server.rstrip('/')}/v1/chat/completions"
    body = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with requests.post(url, json=body, stream=True, headers={"X-API-Key": API_KEY}, timeout=60) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            chunk_str = line[5:].strip()
            if chunk_str == "[DONE]":
                break
            try:
                chunk   = json.loads(chunk_str)
                content = chunk["choices"][0]["delta"].get("content") or ""
                if content:
                    print(content, end="", flush=True)
            except Exception:
                pass
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=SERVER_URL, help="Server base URL")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    args = parser.parse_args()

    if not args.server:
        print("Error: provide --server <url> or set PFORGE_SERVER env var.")
        raise SystemExit(1)

    HYPERPARAMS["max_steps"] = args.steps

    # ── 1. Submit training job ────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Flash-tuning: {RUN_NAME}")
    print(f"  Server:       [hidden]")
    print(f"  Examples:     {len(DATASET)}")
    print(f"  Max steps:    {HYPERPARAMS['max_steps']}")
    print(f"{'='*55}\n")

    print("Submitting training job...")
    result = post(args.server, "/train", {
        "run_name":     RUN_NAME,
        "system_prompt":SYSTEM_PROMPT,
        "dataset":      DATASET,
        "hyperparams":  HYPERPARAMS,
    })
    job_id = result["job_id"]
    print(f"  Job ID : {job_id}")
    print(f"  Adapter: {result['adapter_output_path']}")
    print(f"  Note   : {result['message']}\n")

    # ── 2. Poll until done ────────────────────────────────────────────────────
    print("Training progress:")
    start = time.monotonic()

    while True:
        time.sleep(5)
        status = get(args.server, "/status")
        t      = status["training"]
        state  = t["state"]
        pct    = t.get("progress_pct") or 0
        loss   = t.get("loss")
        step   = t.get("current_step") or 0
        total  = t.get("total_steps")  or "?"
        elapsed = time.monotonic() - start

        loss_str = f"  loss={loss:.4f}" if loss else ""
        print(
            f"  [{elapsed:5.0f}s]  {state:<10}  step {step}/{total}  "
            f"{pct:5.1f}%{loss_str}",
            flush=True,
        )

        if state == "succeeded":
            print(f"\n  Adapter saved to: {t.get('adapter_path')}")
            break
        if state == "failed":
            print(f"\n  Training failed: {t.get('error')}")
            raise SystemExit(1)

    # ── 3. Wait for vLLM to come back up (restarts after training) ───────────
    print("\nWaiting for vLLM to restart with new adapter...")
    for _ in range(60):  # up to 5 minutes
        time.sleep(5)
        try:
            health = get(args.server, "/health")
            if health.get("vllm_up"):
                print("  vLLM is ready.\n")
                break
        except Exception:
            pass
        print("  still starting...", flush=True)
    else:
        print("  Timed out waiting for vLLM. Try chatting manually.")
        raise SystemExit(1)

    # ── 4. Test the adapter ───────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Testing adapter: {RUN_NAME}")
    print(f"{'='*55}\n")

    test_prompts = [
        "Describe a traffic jam",
        "What is love?",
    ]

    for prompt in test_prompts:
        print(f"\033[96mYou:\033[0m {prompt}")
        print(f"\033[92m{RUN_NAME}>\033[0m ", end="", flush=True)
        stream_one_shot(args.server, RUN_NAME, prompt)
        print()


if __name__ == "__main__":
    main()
