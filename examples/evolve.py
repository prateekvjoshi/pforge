"""
examples/evolve.py — iterative response refinement via user feedback.

Usage:
  python3 examples/evolve.py --server http://localhost:8000
  python3 examples/evolve.py --server ... --prompt "Explain what a black hole is"

What it does:
  Round 1: model answers your prompt.
  Round 2+: you give feedback on the answer, model produces an improved version.
  Repeat until satisfied. Type 'reset' to start a new prompt.
"""

import argparse
import json
import os
import sys

import requests

SERVER_URL = os.environ.get("PFORGE_SERVER") or os.environ.get("POD_URL") or ""
API_KEY = os.environ.get("API_KEY", "")

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def stream_evolve(server: str, prompt: str, previous_response: str = None,
                  feedback: str = None, thinking: bool = True) -> str:
    """Call /evolve and return the full response text while streaming it."""
    url  = f"{server.rstrip('/')}/evolve"
    body = {"prompt": prompt, "thinking": thinking}
    if previous_response:
        body["previous_response"] = previous_response
    if feedback:
        body["feedback"] = feedback

    reasoning_started = False
    answer_started    = False
    answer_chunks     = []

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
                    answer_chunks.append(content)
            except Exception:
                pass

    print("\n")
    return "".join(answer_chunks)


def run_session(server: str, initial_prompt: str = None, thinking: bool = True):
    """Run one evolve session (one prompt, multiple feedback rounds)."""
    if initial_prompt:
        prompt = initial_prompt
    else:
        try:
            prompt = input(f"{CYAN}prompt>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            return False
    if not prompt:
        return True

    print(f"\n{BOLD}Round 1{RESET} — Initial answer\n")
    previous_response = stream_evolve(server, prompt, thinking=thinking)

    round_num = 2
    while True:
        try:
            feedback = input(f"{YELLOW}feedback>{RESET} (or 'done'/'reset'): ").strip()
        except (KeyboardInterrupt, EOFError):
            return False

        if feedback.lower() in ("done", ""):
            print("Session complete.\n")
            return True
        if feedback.lower() == "reset":
            return True

        print(f"\n{BOLD}Round {round_num}{RESET} — Refined answer\n")
        previous_response = stream_evolve(
            server, prompt,
            previous_response=previous_response,
            feedback=feedback,
            thinking=thinking,
        )
        round_num += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=SERVER_URL, help="Server base URL")
    parser.add_argument("--prompt", default=None,       help="Initial prompt (interactive if omitted)")
    parser.add_argument("--no-think", action="store_true", help="Disable chain-of-thought")
    args = parser.parse_args()

    if not args.server:
        print("Error: provide --server <url> or set PFORGE_SERVER env var.")
        sys.exit(1)

    thinking = not args.no_think

    print(f"\n{BOLD}Evolve mode{RESET} — iterative refinement via feedback")
    print(f"  Thinking: {thinking}")
    print("  Commands: 'done' to end session, 'reset' for new prompt, Ctrl+C to quit\n")

    while True:
        keep_going = run_session(args.server, args.prompt, thinking=thinking)
        args.prompt = None  # only use the CLI prompt once
        if not keep_going:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
