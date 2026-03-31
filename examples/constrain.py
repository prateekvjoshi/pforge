"""
examples/constrain.py — see how explicit rules shape model reasoning.

Usage:
  python3 examples/constrain.py --server http://localhost:8000
  python3 examples/constrain.py --server ... --prompt "Explain gravity" --constraints "only use analogies" "max 3 sentences"
"""

import argparse
import json
import os
import sys

import requests

SERVER_URL = os.environ.get("PFORGE_SERVER") or os.environ.get("POD_URL") or ""
API_KEY = os.environ.get("API_KEY", "")

YELLOW = "\033[93m"
GREEN  = "\033[92m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Preset constraint sets for quick demos
PRESETS = {
    "simple": [
        "explain like I am 5 years old",
        "use no more than 3 sentences",
        "avoid technical jargon",
    ],
    "creative": [
        "only use analogies — no direct statements",
        "each sentence must start with a different letter of the alphabet",
        "end with a surprising twist",
    ],
    "structured": [
        "respond in exactly 3 steps",
        "start each step with a verb",
        "conclude with a one-sentence summary",
    ],
}


def stream_constrain(server: str, prompt: str, constraints: list, thinking: bool = False):
    url  = f"{server.rstrip('/')}/constrain"
    body = {"prompt": prompt, "constraints": constraints, "thinking": thinking}

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
                    if not reasoning_started:
                        print(f"\n{GREY}[thinking]{RESET}")
                        reasoning_started = True
                    print(f"{GREY}{reasoning}{RESET}", end="", flush=True)

                if content:
                    if not answer_started:
                        print(f"\n\n{GREEN}[answer]{RESET}\n")
                        answer_started = True
                    print(content, end="", flush=True)
            except Exception:
                pass

    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",      default=SERVER_URL, help="Server base URL")
    parser.add_argument("--prompt",      default=None,       help="Prompt to answer")
    parser.add_argument("--constraints", nargs="+",          help="Constraint rules (space-separated strings)")
    parser.add_argument("--preset",      choices=list(PRESETS), help="Use a preset constraint set")
    parser.add_argument("--think",       action="store_true", help="Enable chain-of-thought")
    args = parser.parse_args()

    if not args.server:
        print("Error: provide --server <url> or set PFORGE_SERVER env var.")
        sys.exit(1)

    # Resolve constraints
    if args.preset:
        constraints = PRESETS[args.preset]
    elif args.constraints:
        constraints = args.constraints
    else:
        # Interactive constraint builder
        print(f"\n{BOLD}Available presets:{RESET}")
        for name, rules in PRESETS.items():
            print(f"  {YELLOW}{name}{RESET}: {', '.join(rules)}")
        print()
        preset_input = input("Choose a preset (or press Enter to type custom constraints): ").strip()
        if preset_input in PRESETS:
            constraints = PRESETS[preset_input]
        else:
            print("Enter constraints one per line. Empty line to finish:")
            constraints = []
            while True:
                c = input(f"  rule {len(constraints)+1}: ").strip()
                if not c:
                    break
                constraints.append(c)
            if not constraints:
                print("No constraints provided.")
                sys.exit(1)

    # Resolve prompt
    prompt = args.prompt
    if not prompt:
        prompt = input(f"\n{BOLD}prompt>{RESET} ").strip()
        if not prompt:
            sys.exit(0)

    print(f"\n{BOLD}Prompt:{RESET} {prompt}")
    print(f"{BOLD}Constraints:{RESET}")
    for i, c in enumerate(constraints, 1):
        print(f"  {YELLOW}{i}.{RESET} {c}")
    print()

    stream_constrain(args.server, prompt, constraints, thinking=args.think)

    # Interactive loop
    while True:
        try:
            prompt = input(f"{BOLD}prompt>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not prompt:
            continue
        stream_constrain(args.server, prompt, constraints, thinking=args.think)


if __name__ == "__main__":
    main()
