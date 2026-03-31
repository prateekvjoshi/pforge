"""
examples/logit_lens.py — peek inside the model layer by layer.

Usage:
  python3 examples/logit_lens.py --server http://localhost:8000
  python3 examples/logit_lens.py --server ... --prompt "The capital of France is"
  python3 examples/logit_lens.py --server ... --prompt "..." --top-k 3 --compact

Note: this stops vLLM briefly (~1-2 min) while the model loads for analysis.
vLLM restarts automatically when done.
"""

import argparse
import json
import os
import sys

import requests

SERVER_URL = os.environ.get("PFORGE_SERVER") or os.environ.get("POD_URL") or ""
API_KEY = os.environ.get("API_KEY", "")

# Colours
GREY    = "\033[90m"
YELLOW  = "\033[93m"
GREEN   = "\033[92m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

DEFAULT_PROMPT = "The capital of France is"


def bar(prob: float, width: int = 20) -> str:
    """Simple ASCII probability bar."""
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def print_results(data: dict, compact: bool = False):
    prompt        = data["prompt"]
    input_tokens  = data["input_tokens"]
    layers        = data["layers"]
    final_answer  = data["final_answer"]
    first_layer   = data.get("answer_first_appears_at_layer")
    num_layers    = data["num_layers"]

    print(f"\n{BOLD}Prompt:{RESET} {prompt}")
    print(f"{BOLD}Tokens:{RESET} {' | '.join(repr(t) for t in input_tokens)}")
    print(f"{BOLD}Layers:{RESET} {num_layers}  |  "
          f"{BOLD}Final answer:{RESET} {GREEN}{repr(final_answer)}{RESET}  |  "
          f"{BOLD}First appears at:{RESET} layer {first_layer}\n")

    print(f"{'Layer':<12} {'Top prediction':<18} {'Prob':>6}  {'Distribution'}")
    print("─" * 65)

    prev_top = None
    for entry in layers:
        label       = entry["label"]
        top         = entry["top_predictions"][0]
        token_str   = repr(top["token"])
        prob        = top["prob"]
        changed     = (top["token"] != prev_top) and prev_top is not None
        prev_top    = top["token"]

        # Colour: green if it matches final answer, yellow if changed, grey otherwise
        if top["token"] == final_answer:
            colour = GREEN
        elif changed:
            colour = YELLOW
        else:
            colour = GREY

        marker = " ◄ changed" if changed else ""
        print(
            f"{colour}{label:<12} {token_str:<18} {prob:>6.1%}  "
            f"{bar(prob)}{RESET}{CYAN}{marker}{RESET}"
        )

        if not compact and len(entry["top_predictions"]) > 1:
            for alt in entry["top_predictions"][1:]:
                alt_str = repr(alt["token"])
                print(
                    f"{DIM}{'':12} {alt_str:<18} {alt['prob']:>6.1%}  "
                    f"{bar(alt['prob'])}{RESET}"
                )

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",  default=SERVER_URL,       help="Server base URL")
    parser.add_argument("--prompt",  default=DEFAULT_PROMPT,   help="Prompt to analyse")
    parser.add_argument("--top-k",   type=int, default=3,      help="Top-k tokens per layer")
    parser.add_argument("--compact", action="store_true",      help="Show only top-1 per layer")
    parser.add_argument("--lora",    default="",               help="LoRA adapter path to merge")
    args = parser.parse_args()

    if not args.server:
        print("Error: provide --server <url> or set PFORGE_SERVER env var.")
        sys.exit(1)

    url  = f"{args.server.rstrip('/')}/logit_lens"
    body = {"prompt": args.prompt, "top_k": args.top_k}
    if args.lora:
        body["lora_path"] = args.lora

    print(f"\nRunning logit lens analysis...")
    print(f"(vLLM will stop briefly and restart automatically)\n")

    try:
        resp = requests.post(url, json=body, headers={"X-API-Key": API_KEY}, timeout=300)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}\n{resp.text}")
        sys.exit(1)

    data = resp.json()
    print_results(data, compact=args.compact)

    # Interactive loop
    while True:
        try:
            prompt = input(f"{BOLD}prompt>{RESET} (or Enter to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not prompt:
            break
        body["prompt"] = prompt
        print("\nRunning analysis...\n")
        try:
            resp = requests.post(url, json=body, headers={"X-API-Key": API_KEY}, timeout=300)
            resp.raise_for_status()
            print_results(resp.json(), compact=args.compact)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
