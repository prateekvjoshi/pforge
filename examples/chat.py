"""
chat.py — interactive terminal chat with the local server.

Usage:
  python chat.py --server http://localhost:8000

Or set the env var and skip the flag:
  export PFORGE_SERVER="http://localhost:8000"
  python chat.py
"""

import argparse
import json
import os
import sys

import requests


SERVER_URL = os.environ.get("PFORGE_SERVER") or os.environ.get("POD_URL") or ""
API_KEY = os.environ.get("API_KEY", "")
MODEL   = os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.7B")


def stream_chat(server_url: str, messages: list, model: str = MODEL, thinking: bool = True):
    """Stream a chat completion and print reasoning + answer as they arrive."""
    url  = f"{server_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model":   model,
        "messages": messages,
        "stream":  True,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }

    thinking_started = False
    answer_started   = False

    with requests.post(url, json=body, stream=True, headers={"X-API-Key": API_KEY}, timeout=120) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            line = raw.decode() if isinstance(raw, bytes) else raw
            line = line.strip()
            if not line.startswith("data:"):
                continue
            chunk_str = line[5:].strip()
            if chunk_str == "[DONE]":
                break
            try:
                chunk = json.loads(chunk_str)
                delta = chunk["choices"][0]["delta"]

                reasoning = delta.get("reasoning") or ""
                content   = delta.get("content")   or ""

                if reasoning:
                    if not thinking_started:
                        print("\n\033[90m[thinking]\033[0m")
                        thinking_started = True
                    print(f"\033[90m{reasoning}\033[0m", end="", flush=True)

                if content:
                    if not answer_started:
                        print("\n\n\033[92m[answer]\033[0m\n")
                        answer_started = True
                    print(content, end="", flush=True)

            except Exception:
                pass

    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",   default=SERVER_URL, help="Server base URL")
    parser.add_argument("--model",    default=MODEL,      help="Model name")
    parser.add_argument("--system",   default="",         help="System prompt")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode")
    args = parser.parse_args()

    if not args.server:
        print("Error: provide --server <url> or set PFORGE_SERVER env var.")
        sys.exit(1)

    model    = args.model
    thinking = not args.no_think

    print("Connected to server.")
    print(f"Model: {model}  |  Thinking: {thinking}")
    if args.system:
        print(f"System: {args.system}")
    print("Type your message and press Enter. Ctrl+C to quit.\n")

    history = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    while True:
        try:
            user_input = input("\033[96myou>\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        stream_chat(args.server, history, model=model, thinking=thinking)
        history.append({"role": "assistant", "content": "(response above)"})


if __name__ == "__main__":
    main()
