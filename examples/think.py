"""
examples/think.py — demonstrate how thinking budget affects response quality.

Usage:
  python3 examples/think.py --server http://localhost:8000

What it does:
  Sends the same prompt at three budget levels (low / medium / high) and
  prints the chain of thought + answer for each, so you can see how more
  thinking time improves the response.
"""

import argparse
import json
import os
import sys

import requests

SERVER_URL = os.environ.get("PFORGE_SERVER") or os.environ.get("POD_URL") or ""
API_KEY = os.environ.get("API_KEY", "")

BUDGETS = ["low", "medium", "high"]

BUDGET_TOKENS = {"low": 256, "medium": 2048, "high": 8192}

DEFAULT_PROMPT = (
    "A farmer has 17 sheep. All but 9 die. How many sheep are left? "
    "Think carefully before answering."
)


def stream_think(server: str, prompt: str, budget: str):
    """Call POST /think and print reasoning (grey) + answer (green) as they stream."""
    url  = f"{server.rstrip('/')}/think"
    body = {"prompt": prompt, "budget": budget}

    reasoning_buf = []
    answer_buf    = []
    reasoning_started = False
    answer_started    = False

    with requests.post(url, json=body, stream=True, headers={"X-API-Key": API_KEY}, timeout=120) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            chunk_str = line[5:].strip()
            if chunk_str == "[DONE]":
                break
            try:
                chunk     = json.loads(chunk_str)
                delta     = chunk["choices"][0]["delta"]
                reasoning = delta.get("reasoning") or ""
                content   = delta.get("content")   or ""

                if reasoning:
                    reasoning_buf.append(reasoning)

                if content:
                    if not answer_started:
                        answer_started = True
                    print(content, end="", flush=True)
                    answer_buf.append(content)
            except Exception:
                pass

    print()
    thinking_tokens = len("".join(reasoning_buf).split())
    print(f"\n  \033[90m(~{thinking_tokens} thinking words)\033[0m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=SERVER_URL, help="Server base URL")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send")
    parser.add_argument(
        "--budget", default=None,
        choices=["low", "medium", "high"],
        help="Run a single budget level instead of all three",
    )
    args = parser.parse_args()

    if not args.server:
        print("Error: provide --server <url> or set PFORGE_SERVER env var.")
        sys.exit(1)

    budgets = [args.budget] if args.budget else BUDGETS

    print(f"\nPrompt: {args.prompt}\n")

    for budget in budgets:
        tokens = BUDGET_TOKENS[budget]
        print(f"\n{'='*60}")
        print(f"  Budget: {budget.upper()}  ({tokens} thinking tokens)")
        print(f"{'='*60}")
        stream_think(args.server, args.prompt, budget)
        if budget != budgets[-1]:
            input("\n  [press Enter to continue to next budget level]\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
