# CLAUDE.md

Agent and maintainer reference. Read before modifying any file.

For public documentation see `docs/`.

---

## What this project is

A local-first Python toolkit for inspecting, steering, and lightly fine-tuning language models. Runs a local API server that manages a vLLM subprocess. Exposes 8 inference/tuning modes.

**Current state:** Working API server + example scripts + `pforge` CLI. Package restructure complete (see Phases below).

---

## Repository layout

```
pforge/
  server.py           FastAPI app + all routes
  vllm_manager.py     vLLM subprocess lifecycle
  training_runner.py  Dataset prep + QLoRA training job
  trainer.py          QLoRA training subprocess (spawned by training_runner)
  inspector.py        Logit lens analysis subprocess
  cli.py              pforge CLI (all commands)
  config.py           All config via env vars
  models.py           Pydantic schemas
  state.py            Thread-safe state + JSON persistence
  utils.py            vLLM probing, LoRA target selection
  paths.py            Platform-appropriate data directory resolution

examples/             Example scripts (one per mode)
scripts/              Setup helpers
docs/                 Public developer documentation
```

---

## Active refactor phases

The codebase is being refactored from a RunPod-hosted backend to a local-first package. Phases:

1. **Done** — repo reframing, docs structure
2. **Done** — local filesystem/config (`backend/paths.py`, `PFORGE_*` env vars, platform defaults)
3. **Done** — package restructure: rename `backend/` → `pforge/`, split `orchestrator.py`
4. **Done** — CLI: `pforge` implemented in `pforge/cli.py`, exposed via `pyproject.toml`
5. **Done** — Server cleanup: rename `orchestrator.py` → `server.py`, remove hosting-specific language
6. **Next** — Tests and CI

**New code must use `cfg.DATA_DIR` / `cfg.LOGS_DIR` etc. (already abstracted). Do not hardcode paths.**

---

## Ports

| Service    | Port | Notes |
|------------|------|-------|
| API server | 8000 | Public-facing |
| vLLM       | 8002 | Internal only — never expose |

Port 8001 is reserved by some system services — 8002 is used to avoid conflicts.

---

## Key design patterns

**Subprocess isolation for trainer and inspector**
Both load a full model. They run as subprocesses so process exit releases all CUDA memory cleanly. A crash in trainer/inspector does not kill the API server.

**SSE streaming**
All inference endpoints yield `data: {...}\n\n` lines. Generators use `try/finally` to close httpx clients and upstream responses on completion or disconnect.

**State persistence**
`AppState` writes a JSON snapshot on every mutation. On restart, it restores active adapter and marks any in-flight training as FAILED.

**Config hierarchy**
`backend/config.py` reads env vars with fallbacks. Future: CLI flags → env vars → config file → defaults.

---

## Running locally

```bash
source /path/to/.env   # or set env vars directly
python -m pforge.server
```

Watch vLLM:
```bash
tail -f <data_dir>/logs/vllm.log
```

---

## Key files for each change type

| Change | Files to touch |
|--------|---------------|
| New inference mode | `pforge/server.py`, `pforge/models.py`, `examples/<mode>.py` |
| Config variable | `pforge/config.py`, `docs/configuration.md`, `.env.example` |
| Training behavior | `pforge/training_runner.py`, `pforge/trainer.py`, `docs/training.md` |
| vLLM startup flags | `_build_vllm_cmd()` in `pforge/vllm_manager.py` |
| API schema | `pforge/models.py` |
| State fields | `pforge/state.py`, `pforge/models.py` |
| CLI command | `pforge/cli.py`, `docs/cli.md` |

---

## Security requirements

- Never hardcode secrets, paths, or credentials
- All user input is untrusted — validate at API boundary (Pydantic models enforce this)
- Adapter paths must be resolved and validated against `cfg.ADAPTERS_DIR` before use
- Prompt injection: prompts go to the model, never to shell or subprocess args (use temp files)
- See `SECURITY.md` for the full threat model
