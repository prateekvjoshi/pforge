"""
examples/debate.py — watch two model instances debate a topic.

Usage:
  python3 examples/debate.py --server http://localhost:8000
  python3 examples/debate.py --server ... --topic "AI will make humans obsolete" --rounds 2
"""

import argparse
import json
import os
import sys

import requests

SERVER_URL = os.environ.get("PFORGE_SERVER") or os.environ.get("POD_URL") or ""
API_KEY = os.environ.get("API_KEY", "")

CYAN  = "\033[96m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"
GREY  = "\033[90m"

DEFAULT_TOPIC = "Artificial intelligence will do more harm than good for humanity"


def stream_debate(server: str, topic: str, rounds: int):
    url  = f"{server.rstrip('/')}/debate"
    body = {"topic": topic, "rounds": rounds}

    current_side  = None
    current_round = None

    with requests.post(url, json=body, stream=True, headers={"X-API-Key": API_KEY}, timeout=300) as resp:
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
                side    = chunk.get("side")
                round_n = chunk.get("round")
                content = chunk.get("content", "")
                error   = chunk.get("error")

                if error:
                    print(f"\nError ({side}): {error}")
                    continue

                # Print header when side/round changes
                if side != current_side or round_n != current_round:
                    if current_side is not None:
                        print("\n")
                    current_side  = side
                    current_round = round_n
                    colour = CYAN if side == "for" else RED
                    label  = "FOR" if side == "for" else "AGAINST"
                    print(f"\n{colour}{BOLD}[Round {round_n} — {label}]{RESET}\n")

                print(content, end="", flush=True)
            except Exception:
                pass

    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=SERVER_URL,       help="Server base URL")
    parser.add_argument("--topic",  default=DEFAULT_TOPIC, help="Topic to debate")
    parser.add_argument("--rounds", type=int, default=2,   help="Number of rounds (1-4)")
    args = parser.parse_args()

    if not args.server:
        print("Error: provide --server <url> or set PFORGE_SERVER env var.")
        sys.exit(1)

    print(f"\n{BOLD}Debate{RESET}")
    print(f"  Topic:  {args.topic}")
    print(f"  Rounds: {args.rounds}")
    print(f"  {CYAN}[FOR]{RESET} vs {RED}[AGAINST]{RESET}\n")
    print("=" * 60)

    stream_debate(args.server, args.topic, args.rounds)


if __name__ == "__main__":
    main()
