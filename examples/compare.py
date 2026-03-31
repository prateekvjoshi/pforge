"""
examples/compare.py — compare two models on the same prompt in real time.

Usage:
  # Base model vs a fine-tuned adapter:
  python3 examples/compare.py \\
    --server http://localhost:8000 \\
    --model-b tarantino-v1

  # Two adapters against each other:
  python3 examples/compare.py \\
    --server http://localhost:8000 \\
    --model-a persona-a \\
    --model-b persona-b

What it does:
  Sends the same prompt to both models simultaneously and prints their
  responses interleaved as they stream — labelled [A] and [B].
"""

import argparse
import json
import os
import sys

import requests

SERVER_URL = os.environ.get("PFORGE_SERVER") or os.environ.get("POD_URL") or ""
API_KEY   = os.environ.get("API_KEY", "")
BASE_MODEL = "Qwen/Qwen3-1.7B"

# Colour codes
CYAN   = "\033[96m"
GREEN  = "\033[92m"
GREY   = "\033[90m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def stream_compare(server: str, prompt: str, model_a: str, model_b: str,
                   system_prompt: str = "", thinking: bool = False):
    url  = f"{server.rstrip('/')}/compare"
    body = {
        "prompt":   prompt,
        "model_a":  model_a,
        "model_b":  model_b,
        "thinking": thinking,
    }
    if system_prompt:
        body["system_prompt"] = system_prompt

    # Buffer both responses silently — printing interleaved tokens garbles output
    buf    = {"a": {"reasoning": [], "content": []},
              "b": {"reasoning": [], "content": []}}
    errors = {"a": [], "b": []}

    print("Waiting for both models...", flush=True)

    with requests.post(url, json=body, stream=True, headers={"X-API-Key": API_KEY}, timeout=180) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            chunk_str = line[5:].strip()
            if chunk_str == "[DONE]":
                break
            try:
                chunk = json.loads(chunk_str)
                side      = chunk["side"]
                reasoning = chunk.get("reasoning", "")
                content   = chunk.get("content",   "")
                error     = chunk.get("error",      "")

                if error:
                    errors[side].append(error)
                if reasoning:
                    buf[side]["reasoning"].append(reasoning)
                if content:
                    buf[side]["content"].append(content)
            except Exception:
                pass

    # Print A then B cleanly
    for side, model_name, colour in [("a", model_a, CYAN), ("b", model_b, GREEN)]:
        print(f"\n{colour}{'─'*60}{RESET}")
        print(f"{colour}{BOLD}[{model_name}]{RESET}")
        print(f"{colour}{'─'*60}{RESET}\n")
        if errors[side]:
            print(f"ERROR: {' '.join(errors[side])}")
        else:
            print("".join(buf[side]["content"]) or "(no response)")
        print()

    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",  default=SERVER_URL, help="Server base URL")
    parser.add_argument("--model-a", default=BASE_MODEL, help="First model (default: base)")
    parser.add_argument("--model-b", default=None,       help="Second model or adapter name")
    parser.add_argument("--prompt",  default=None,       help="Prompt (interactive if omitted)")
    parser.add_argument("--system",  default="",         help="System prompt for both models")
    parser.add_argument("--think",   action="store_true",help="Enable chain-of-thought")
    args = parser.parse_args()

    if not args.server:
        print("Error: provide --server <url> or set PFORGE_SERVER env var.")
        sys.exit(1)

    if not args.model_b:
        print("Compare needs two models to pit against each other.")
        print()
        print("First train an adapter, then run:")
        print(f"  python3 examples/flash_tune.py --server $PFORGE_SERVER")
        print()
        print("Once training is done, come back and run:")
        print(f"  python3 examples/compare.py --server $PFORGE_SERVER --model-b <adapter-name>")
        print()
        print("Example:")
        print(f"  python3 examples/compare.py --server $PFORGE_SERVER --model-b tarantino-v1 --prompt \"Describe a coffee shop\"")
        sys.exit(0)

    model_a = args.model_a
    model_b = args.model_b

    print(f"\n{BOLD}Compare mode{RESET}")
    print(f"  {CYAN}[A]{RESET} {model_a}")
    print(f"  {GREEN}[B]{RESET} {model_b}")
    if args.system:
        print(f"  System: {args.system}")
    print()

    if args.prompt:
        # Single prompt mode
        print(f"{BOLD}Prompt:{RESET} {args.prompt}\n")
        stream_compare(args.server, args.prompt, model_a, model_b,
                       system_prompt=args.system, thinking=args.think)
    else:
        # Interactive mode
        print("Type a prompt and press Enter to compare. Ctrl+C to quit.\n")
        while True:
            try:
                prompt = input(f"{BOLD}prompt>{RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if not prompt:
                continue
            stream_compare(args.server, prompt, model_a, model_b,
                           system_prompt=args.system, thinking=args.think)


if __name__ == "__main__":
    main()
