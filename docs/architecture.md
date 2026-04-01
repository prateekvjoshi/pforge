# Architecture

## Overview

pforge runs as a local API server that manages a vLLM subprocess. The server exposes a REST API that the CLI (and any other HTTP client) talks to.

```
[CLI / HTTP client]
       │  HTTP localhost:8000
       ▼
[API server]  ← FastAPI
       │  subprocess
       ▼
[vLLM]  ← OpenAI-compatible inference server on localhost:8002
```

The API server owns the lifecycle of vLLM: it starts it, monitors it, restarts it if it crashes, stops it before training, and restarts it after.

---

## Components

### API server (`pforge/server.py`)

FastAPI application. Responsibilities:
- Start, stop, and restart vLLM as a subprocess
- Accept training jobs and run them as isolated subprocesses
- Expose the 8 inference/tuning modes as REST endpoints
- Proxy `/v1/*` requests transparently to vLLM
- Persist state (active adapter, training status) to disk

### vLLM

The inference engine. Runs on an internal port (default 8002). Exposes an OpenAI-compatible API. The API server proxies client requests to it.

Never exposed directly to clients — all traffic goes through the API server on port 8000.

### Trainer (`pforge/trainer.py`)

Standalone subprocess for QLoRA fine-tuning. Runs independently so it can:
- Load the model in 4-bit quantization without sharing GPU memory with vLLM
- Release all CUDA memory on exit (clean process boundary)
- Crash without taking down the API server

### Inspector (`pforge/inspector.py`)

Standalone subprocess for logit lens analysis. Same isolation rationale as the trainer — loads the full model, runs analysis, exits cleanly.

---

## Ports

| Service    | Port | Notes |
|------------|------|-------|
| API server | 8000 | Public-facing; the only port clients need |
| vLLM       | 8002 | Internal only; never expose directly |

Port 8001 is reserved on some systems (nginx, etc.) — 8002 is used to avoid conflicts.

---

## State persistence

The API server writes a JSON snapshot to disk after every state change:

```
~/.local/share/pforge/server_status.json
```

On restart, it reads this file to restore:
- Which adapter was last loaded
- Training job status (in-flight jobs are marked FAILED on restart)

---

## LoRA adapter lifecycle

### After training

1. **Attempt dynamic load** via vLLM's `POST /v1/load_lora_adapter` — zero downtime if it works.
2. **Fallback: restart vLLM** with `--lora-modules name=path` — takes 1–3 minutes but always works.

The response from `/train` tells you which path was taken.

### During training

By default (`TRAIN_STOP_VLLM=true`), vLLM is stopped before training begins. This frees GPU memory for the trainer. Inference is unavailable during training. vLLM restarts automatically when training completes.

With a small model and sufficient VRAM, you can set `TRAIN_STOP_VLLM=false` to keep vLLM running during training.

---

## SSE streaming

All inference endpoints return Server-Sent Events (SSE). Clients read the stream line by line. Each chunk is a `data: {...}` JSON line. The stream ends with `data: [DONE]`.

```
data: {"choices": [{"delta": {"reasoning": "Let me think...", "content": ""}}]}
data: {"choices": [{"delta": {"reasoning": "", "content": "The answer is"}}]}
data: [DONE]
```

The `reasoning` field carries chain-of-thought tokens. The `content` field carries the final answer tokens. Some endpoints add extra fields (e.g. `side` in `/compare`, `round` in `/debate`).

---

## Design decisions

**Why subprocess for trainer and inspector?**
Both load a full model into GPU memory. Keeping them in-process would pin that memory for the duration. Subprocess exit releases CUDA memory cleanly before vLLM restarts.

**Why FastAPI?**
Async-native, good SSE support, excellent typing integration with Pydantic. The async event loop lets the server handle slow streaming responses without blocking.

**Why a local server rather than calling vLLM directly?**
The server adds the reasoning modes (think, debate, constrain, evolve, logit lens) that vLLM doesn't provide. It also manages the training/inspection lifecycle, adapter loading, and state persistence. Without it, clients would need to handle all of that themselves.
