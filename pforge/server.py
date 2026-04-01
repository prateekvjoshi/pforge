"""
server.py — FastAPI service that manages the vLLM subprocess and
            LoRA flash-tuning lifecycle.

Topology
────────
  [Any HTTP client]
        │  HTTP
        ▼
  [API server]  :8000   ← this file
        │  subprocess
        ▼
  [vLLM OpenAI server]  :8002

VRAM budget on RTX 4090 (24 GB)
────────────────────────────────
  Qwen3-1.7B at bf16   ≈ 4 GB inference (with 0.80 gpu_memory_utilization)
  QLoRA 4-bit training ≈ 10–14 GB (model + optimizer + activations)

  These cannot safely coexist on 24 GB unless the model is heavily quantized.
  Default: TRAIN_STOP_VLLM=true — vLLM is stopped before training begins and
           restarted (with the new adapter registered) after training succeeds
           or fails.  Inference is unavailable during training.
  Override: TRAIN_STOP_VLLM=false to keep vLLM alive (only safe if the model
           fits in ~8 GB, e.g. AWQ 4-bit).

vLLM LoRA loading
─────────────────
  vLLM supports dynamic LoRA loading via:
    POST /v1/load_lora_adapter  {"lora_name": "...", "lora_path": "..."}
  (requires --enable-lora at startup; available since vLLM ≈ 0.4.x)

  After training this is attempted first.  If it fails (older vLLM, adapter
  format mismatch, etc.) we fall back to restarting vLLM with the adapter
  pre-registered via --lora-modules name=path.

Concurrency
───────────
  - Only one training job is allowed at a time (v1).
  - Training runs as an asyncio-managed subprocess so it never blocks the
    API event loop.
  - A background watchdog task probes vLLM every 30 s and restarts it if
    it dies unexpectedly (only when training is not active).
"""

import asyncio
import contextlib
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

import pforge.config as cfg
from pforge.models import (
    BUDGET_PRESETS,
    CompareRequest,
    ConstrainRequest,
    DebateRequest,
    EvolveRequest,
    HealthResponse,
    HyperParams,
    LoadLoRARequest,
    LoadLoRAResponse,
    LogitLensRequest,
    RestartVLLMResponse,
    StatusResponse,
    ThinkRequest,
    TrainRequest,
    TrainResponse,
    TrainingState,
    VLLMState,
)
from pforge.state import AppState
from pforge.training_runner import run_training_job, write_dataset
from pforge.utils import ensure_dirs
from pforge.vllm_manager import (
    dynamic_load_lora,
    restart_vllm,
    start_vllm,
    stop_vllm,
    vllm_watchdog,
)

# ── Logging ───────────────────────────────────────────────────────────────────
ensure_dirs(cfg.LOGS_DIR)

_log_handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(cfg.LOGS_DIR / "server.log"),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger("server")

# ── Globals ───────────────────────────────────────────────────────────────────
app_state: AppState = AppState()
_start_time = time.monotonic()

# ── FastAPI ───────────────────────────────────────────────────────────────────
@contextlib.asynccontextmanager
async def lifespan(fastapi_app: FastAPI):  # noqa: ARG001
    # ── startup ──────────────────────────────────────────────────────────────
    ensure_dirs(
        cfg.LOGS_DIR,
        cfg.ADAPTERS_DIR,
        cfg.DATA_DIR,
        cfg.STATUS_DIR,
        cfg.HF_CACHE_DIR,
        cfg.HF_CACHE_DIR / "hub",
    )
    app_state.load_persisted()

    lora_modules: dict = {}
    if app_state.active_lora_name and app_state.active_lora_path:
        p = Path(app_state.active_lora_path)
        meta_file = p / "training_meta.json"
        adapter_ok = False
        if not p.exists():
            logger.warning("Persisted adapter path %s no longer exists; skipping.", p)
        elif meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                if meta.get("base_model") == cfg.MODEL_NAME:
                    adapter_ok = True
                else:
                    logger.warning(
                        "Persisted adapter was trained on '%s' but current model is '%s'; "
                        "skipping to avoid incompatibility crash.",
                        meta.get("base_model"), cfg.MODEL_NAME,
                    )
            except Exception:
                logger.warning("Could not read adapter metadata at %s; skipping.", meta_file)
        else:
            logger.warning(
                "Adapter at %s has no training_meta.json; skipping on model change safety.", p
            )

        if adapter_ok:
            lora_modules[app_state.active_lora_name] = str(p)
        else:
            app_state.update(active_lora_name=None, active_lora_path=None)

    asyncio.create_task(start_vllm(app_state, lora_modules or None))
    asyncio.create_task(vllm_watchdog(app_state))
    logger.info("Server started on %s:%d.", cfg.ORCHESTRATOR_HOST, cfg.ORCHESTRATOR_PORT)

    yield

    # ── shutdown ─────────────────────────────────────────────────────────────
    logger.info("Server shutting down…")
    await stop_vllm(app_state)


app = FastAPI(
    title="pforge",
    description=(
        "Manages the vLLM inference subprocess and LoRA flash-tuning lifecycle."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
if cfg.CORS_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=    cfg.CORS_ALLOWED_ORIGINS,
        allow_methods=    ["GET", "POST"],
        allow_headers=    ["Content-Type", "X-API-Key"],
        allow_credentials=False,
    )

# ── API key authentication ────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: Optional[str] = Depends(_api_key_header)) -> None:
    """FastAPI dependency — enforces X-API-Key on every protected endpoint."""
    if not cfg.ORCHESTRATOR_API_KEY:
        logger.warning(
            "ORCHESTRATOR_API_KEY is not set. All requests are accepted. "
            "Set this variable before exposing the API externally."
        )
        return
    if not key or key != cfg.ORCHESTRATOR_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Set X-API-Key header.",
        )

# ── Per-endpoint rate limiting (simple sliding window, in-process) ────────────
_rate_windows: dict = {}  # endpoint_name -> deque of call timestamps

def _check_rate_limit(endpoint: str, max_per_minute: int) -> None:
    """Raise 429 if more than max_per_minute calls in the last 60 seconds."""
    from collections import deque
    now = time.monotonic()
    window = _rate_windows.setdefault(endpoint, deque())
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= max_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: max {max_per_minute} requests/minute for this endpoint.",
        )
    window.append(now)


# ═════════════════════════════════════════════════════════════════════════════
# Routes
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe — returns 200 even when vLLM is not yet up."""
    return HealthResponse(
        status="ok" if app_state.vllm_state == VLLMState.RUNNING else "degraded",
        vllm_up=app_state.vllm_state == VLLMState.RUNNING,
        training_active=app_state.training.state in (
            TrainingState.RUNNING, TrainingState.QUEUED
        ),
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/status", response_model=StatusResponse, tags=["ops"])
async def get_status(_: None = Depends(require_api_key)) -> StatusResponse:
    """Full server status including training progress."""
    return StatusResponse(
        vllm_state=          app_state.vllm_state,
        base_model=          cfg.MODEL_NAME,
        active_lora_name=    app_state.active_lora_name,
        active_lora_adapter= app_state.active_lora_path,
        training=            app_state.training,
        uptime_seconds=      time.monotonic() - _start_time,
        timestamp=           datetime.now(timezone.utc),
    )


@app.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["training"],
)
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
) -> TrainResponse:
    """
    Submit a flash-tuning job.

    - Accepts a dataset in chat or Alpaca format.
    - Queues a single background training job (one at a time).
    - Returns immediately with a job_id; poll GET /status for progress.
    """
    _check_rate_limit("train", cfg.RATE_LIMIT_HEAVY)
    if app_state.training.state in (TrainingState.RUNNING, TrainingState.QUEUED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Training job '{app_state.training.job_id}' is already "
                f"{app_state.training.state}.  Wait for it to finish."
            ),
        )

    job_id   = uuid.uuid4().hex[:8]
    run_name = (request.run_name or f"run_{job_id}").strip()
    hp       = request.hyperparams or HyperParams()

    try:
        dataset_path = write_dataset(job_id, request.dataset, request.system_prompt)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    output_dir = cfg.ADAPTERS_DIR / job_id

    app_state.update_training(
        job_id=       job_id,
        run_name=     run_name,
        state=        TrainingState.QUEUED,
        progress_pct= 0.0,
        current_step= 0,
        total_steps=  None,
        loss=         None,
        started_at=   None,
        completed_at= None,
        adapter_path= None,
        error=        None,
    )

    background_tasks.add_task(run_training_job, job_id, run_name, dataset_path, hp, app_state)

    stop_note = (
        "vLLM will be STOPPED during training (TRAIN_STOP_VLLM=true); "
        "inference unavailable until training completes."
        if cfg.TRAIN_STOP_VLLM else
        "vLLM kept running during training (TRAIN_STOP_VLLM=false); "
        "OOM risk if VRAM is tight."
    )

    return TrainResponse(
        job_id=              job_id,
        run_name=            run_name,
        status=              "accepted",
        dataset_path=        str(dataset_path),
        adapter_output_path= str(output_dir),
        message=             (
            f"Training job queued (job_id={job_id}).  "
            f"Poll GET /status for progress.  {stop_note}"
        ),
    )


@app.post("/load_lora", response_model=LoadLoRAResponse, tags=["adapters"])
async def load_lora_adapter(
    request: LoadLoRARequest,
    _: None = Depends(require_api_key),
) -> LoadLoRAResponse:
    """
    Manually load a LoRA adapter that already exists on disk.

    Tries dynamic vLLM hot-load first; falls back to a full vLLM restart.
    The adapter path must be inside the configured adapters directory.
    """
    _check_rate_limit("load_lora", cfg.RATE_LIMIT_OPS)

    try:
        resolved = Path(request.lora_path).resolve()
        adapters_root = cfg.ADAPTERS_DIR.resolve()
        if not str(resolved).startswith(str(adapters_root)):
            raise HTTPException(
                status_code=400,
                detail="lora_path must be inside the adapters directory.",
            )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=400, detail="Invalid lora_path.")

    if not resolved.exists():
        raise HTTPException(
            status_code=404,
            detail="Adapter path does not exist.",
        )
    if app_state.training.state in (TrainingState.RUNNING, TrainingState.QUEUED):
        raise HTTPException(
            status_code=409,
            detail="Cannot load adapter while a training job is active.",
        )
    if app_state.vllm_state == VLLMState.STOPPED:
        raise HTTPException(
            status_code=503,
            detail="vLLM is not running.  It may still be starting up.",
        )

    lora_path_str = str(resolved)
    loaded = await dynamic_load_lora(request.lora_name, lora_path_str)
    if loaded:
        app_state.update(
            active_lora_name=request.lora_name,
            active_lora_path=lora_path_str,
        )
        return LoadLoRAResponse(
            success=True,
            method="dynamic",
            lora_name=request.lora_name,
            message="Adapter loaded dynamically — no restart required.",
        )

    lora_modules: dict = {request.lora_name: lora_path_str}
    if app_state.active_lora_name and app_state.active_lora_path:
        lora_modules[app_state.active_lora_name] = app_state.active_lora_path

    ok = await restart_vllm(app_state, lora_modules=lora_modules)
    if ok:
        app_state.update(
            active_lora_name=request.lora_name,
            active_lora_path=lora_path_str,
        )
        return LoadLoRAResponse(
            success=True,
            method="restart",
            lora_name=request.lora_name,
            message="vLLM restarted with adapter pre-registered (dynamic load was not available).",
        )

    raise HTTPException(
        status_code=500,
        detail="Adapter load failed via both dynamic and restart methods.",
    )


@app.post("/restart_vllm", response_model=RestartVLLMResponse, tags=["ops"])
async def restart_vllm_route(_: None = Depends(require_api_key)) -> RestartVLLMResponse:
    """Force-restart vLLM (e.g. after OOM or stale inference state)."""
    _check_rate_limit("restart_vllm", cfg.RATE_LIMIT_OPS)
    if app_state.training.state in (TrainingState.RUNNING, TrainingState.QUEUED):
        raise HTTPException(
            status_code=409,
            detail="Cannot restart vLLM while a training job is active.",
        )

    lora_modules: dict = {}
    if app_state.active_lora_name and app_state.active_lora_path:
        lora_modules[app_state.active_lora_name] = app_state.active_lora_path

    ok = await restart_vllm(app_state, lora_modules or None)
    return RestartVLLMResponse(
        success=    ok,
        vllm_state= app_state.vllm_state,
        message=    "vLLM restarted successfully." if ok else "vLLM restart failed; see server.log.",
    )


# ═════════════════════════════════════════════════════════════════════════════
# vLLM reverse proxy
# ═════════════════════════════════════════════════════════════════════════════

# Headers that must not be forwarded hop-by-hop
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "host",
}

_VLLM_BASE = f"http://{cfg.VLLM_HOST}:{cfg.VLLM_PORT}"


async def _proxy(request: Request, path: str) -> Response:
    """Forward request to vLLM and stream the response back."""
    if app_state.vllm_state != VLLMState.RUNNING:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM is not ready (state={app_state.vllm_state}). "
                   "Check GET /health and retry.",
        )

    url = f"{_VLLM_BASE}/{path}"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    body = await request.body()

    try:
        client = httpx.AsyncClient(timeout=None)
        upstream = await client.send(
            client.build_request(
                method=  request.method,
                url=     url,
                headers= headers,
                content= body,
                params=  dict(request.query_params),
            ),
            stream=True,
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Could not connect to vLLM backend.")

    async def _stream_response():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    resp_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    return StreamingResponse(
        content=    _stream_response(),
        status_code=upstream.status_code,
        headers=    resp_headers,
        media_type= upstream.headers.get("content-type", "application/json"),
    )


# ── Logit lens ────────────────────────────────────────────────────────────────

_inspect_lock = asyncio.Lock()


@app.post("/logit_lens", tags=["inspect"])
async def logit_lens(req: LogitLensRequest, _: None = Depends(require_api_key)):
    """
    Logit lens analysis: reveals how the model's answer crystallises
    across transformer layers.

    For each layer (embedding → layer 1 → … → layer N), applies the
    final layer norm + unembedding matrix to the hidden state at the last
    token position, returning the top-k token predictions and their
    probabilities.

    Requires stopping vLLM to free VRAM (same as training). vLLM is
    restarted automatically when the analysis completes. Inference is
    unavailable for ~1-2 minutes during the analysis.

    Response JSON:
      {
        "prompt": "...",
        "input_tokens": ["What", " is", "..."],
        "num_layers": 28,
        "layers": [
          {"layer": 0, "label": "embedding", "top_predictions": [{"token": "?", "prob": 0.12}, ...]},
          {"layer": 1, "label": "layer_1",   "top_predictions": [{"token": " Paris", "prob": 0.41}, ...]},
          ...
        ],
        "final_answer": " Paris",
        "answer_first_appears_at_layer": 4
      }
    """
    if app_state.training.state == TrainingState.RUNNING:
        raise HTTPException(
            status_code=409,
            detail="A training job is active. Wait for it to finish before running logit lens.",
        )
    if _inspect_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="A logit lens analysis is already running. Try again shortly.",
        )
    _check_rate_limit("logit_lens", cfg.RATE_LIMIT_HEAVY)

    async with _inspect_lock:
        job_id      = uuid.uuid4().hex[:8]
        output_path = cfg.STATUS_DIR / f"logitlens_{job_id}.json"

        logger.info("POST /logit_lens  job=%s  prompt=%.80r", job_id, req.prompt)

        vllm_was_running = app_state.vllm_state == VLLMState.RUNNING
        if vllm_was_running:
            logger.info("Stopping vLLM to free VRAM for logit lens analysis.")
            await stop_vllm(app_state)

        try:
            prompt_file = cfg.STATUS_DIR / f"prompt_{job_id}.json"
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(json.dumps({"prompt": req.prompt}))

            lora_path_arg = ""
            if req.lora_path:
                try:
                    resolved_lora = Path(req.lora_path).resolve()
                    if not str(resolved_lora).startswith(str(cfg.ADAPTERS_DIR.resolve())):
                        raise HTTPException(status_code=400, detail="lora_path must be inside the adapters directory.")
                    lora_path_arg = str(resolved_lora)
                except HTTPException:
                    raise
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid lora_path.")

            cmd = [
                sys.executable, "-m", "pforge.inspector",
                "--model_name",  cfg.MODEL_NAME,
                "--prompt_file", str(prompt_file),
                "--output_path", str(output_path),
                "--top_k",       str(req.top_k),
            ]
            if lora_path_arg:
                cmd += ["--lora_path", lora_path_arg]

            log_path = cfg.LOGS_DIR / f"inspector_{job_id}.log"
            logger.info("Running inspector subprocess (job=%s)", job_id)

            with open(log_path, "w") as log_fh:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_fh,
                    stderr=log_fh,
                    cwd=str(Path(__file__).parent.parent),
                )
                await proc.wait()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"inspector.py exited with code {proc.returncode}. "
                    f"See {log_path} for details."
                )

            result = json.loads(output_path.read_text())
            logger.info(
                "Logit lens complete: %d layers, answer=%r first at layer %s",
                result.get("num_layers"), result.get("final_answer"),
                result.get("answer_first_appears_at_layer"),
            )
            return result

        finally:
            try:
                prompt_file.unlink(missing_ok=True)
            except Exception:
                pass
            if vllm_was_running:
                logger.info("Restarting vLLM after logit lens.")
                await start_vllm(app_state)


@app.post("/think", tags=["inference"])
async def think(req: ThinkRequest, _: None = Depends(require_api_key)) -> StreamingResponse:
    """
    Send a prompt with a controlled thinking budget and stream back the
    model's chain of thought + final answer.

    budget presets:  low=256 tokens, medium=2048 tokens, high=8192 tokens
    budget integer:  exact token count

    The SSE stream uses the same format as /v1/chat/completions:
      data: {"choices": [{"delta": {"reasoning": "...", "content": "..."}}]}
      data: [DONE]

    reasoning tokens arrive first (chain of thought), then content tokens
    (final answer).  Display them in two distinct sections.
    """
    if app_state.vllm_state != VLLMState.RUNNING:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM is not ready (state={app_state.vllm_state}). "
                   "Check GET /health and retry.",
        )

    model = req.model or cfg.MODEL_NAME

    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    body = {
        "model":   model,
        "messages": messages,
        "stream":  True,
        "chat_template_kwargs": {
            "enable_thinking":  True,
            "thinking_budget":  req.budget,
        },
    }

    logger.info(
        "POST /think  model=%s  budget=%d tokens  prompt=%.80r",
        model, req.budget, req.prompt,
    )

    try:
        client = httpx.AsyncClient(timeout=None)
        upstream = await client.send(
            client.build_request(
                method=  "POST",
                url=     f"{_VLLM_BASE}/v1/chat/completions",
                headers= {"Content-Type": "application/json"},
                content= json.dumps(body).encode(),
            ),
            stream=True,
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Could not connect to vLLM backend.")

    async def _stream_response():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    resp_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    return StreamingResponse(
        content=    _stream_response(),
        status_code=upstream.status_code,
        headers=    resp_headers,
        media_type= upstream.headers.get("content-type", "text/event-stream"),
    )


@app.post("/compare", tags=["inference"])
async def compare(req: CompareRequest, _: None = Depends(require_api_key)) -> StreamingResponse:
    """
    Send one prompt to two models in parallel and stream both responses
    multiplexed into a single SSE stream.

    Each SSE chunk is a JSON object with a "side" field ("a" or "b") plus
    the usual "reasoning" and "content" delta fields:

      data: {"side": "a", "reasoning": "...", "content": ""}
      data: {"side": "b", "content": "..."}
      data: [DONE]

    Use this to compare the base model vs a fine-tuned adapter, or any
    two adapters against each other.
    """
    if app_state.vllm_state != VLLMState.RUNNING:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM is not ready (state={app_state.vllm_state}). "
                   "Check GET /health and retry.",
        )

    model_a = req.model_a or cfg.MODEL_NAME
    model_b = req.model_b or cfg.MODEL_NAME

    logger.info(
        "POST /compare  model_a=%s  model_b=%s  prompt=%.80r",
        model_a, model_b, req.prompt,
    )

    def _build_body(model: str) -> bytes:
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages.append({"role": "user", "content": req.prompt})
        return json.dumps({
            "model":    model,
            "messages": messages,
            "stream":   True,
            "chat_template_kwargs": {"enable_thinking": req.thinking},
        }).encode()

    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()  # sentinel

    async def _stream_one(side: str, model: str) -> None:
        """Stream one vLLM response and push labelled chunks into the queue."""
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{_VLLM_BASE}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    content=_build_body(model),
                ) as upstream:
                    async for raw_line in upstream.aiter_lines():
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        chunk_str = line[5:].strip()
                        if chunk_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_str)
                            delta = chunk["choices"][0]["delta"]
                            payload = {
                                "side":      side,
                                "reasoning": delta.get("reasoning") or "",
                                "content":   delta.get("content")   or "",
                            }
                            await queue.put(payload)
                        except Exception:
                            pass
        except Exception as exc:
            await queue.put({"side": side, "error": str(exc)})
        finally:
            await queue.put(_DONE)

    async def _mux() -> None:
        """Start both stream tasks then signal queue when both finish."""
        await asyncio.gather(
            _stream_one("a", model_a),
            _stream_one("b", model_b),
        )

    async def _generate():
        mux_task = asyncio.create_task(_mux())
        done_count = 0
        while done_count < 2:
            item = await queue.get()
            if item is _DONE:
                done_count += 1
                continue
            yield f"data: {json.dumps(item)}\n\n"
        yield "data: [DONE]\n\n"
        await mux_task

    return StreamingResponse(
        content=   _generate(),
        media_type="text/event-stream",
    )


@app.post("/debate", tags=["inference"])
async def debate(req: DebateRequest, _: None = Depends(require_api_key)) -> StreamingResponse:
    """
    Two instances of the model debate a topic over multiple rounds.
    One side argues FOR, the other AGAINST. Each round the sides read
    the opponent's previous argument before responding.

    SSE format:
      data: {"side": "for"|"against", "round": 1, "content": "..."}
      data: [DONE]
    """
    if app_state.vllm_state != VLLMState.RUNNING:
        raise HTTPException(status_code=503, detail=f"vLLM is not ready (state={app_state.vllm_state}).")

    model = req.model or cfg.MODEL_NAME
    logger.info("POST /debate  topic=%.80r  rounds=%d  model=%s", req.topic, req.rounds, model)

    async def _complete(messages: list) -> str:
        """Non-streaming call — collect full response text."""
        body = json.dumps({
            "model":    model,
            "messages": messages,
            "stream":   False,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{_VLLM_BASE}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                content=body,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _generate():
        for_history     = [{"role": "system", "content":
            f"You are arguing STRONGLY IN FAVOUR of the following topic. "
            f"Be persuasive, direct, and use concrete reasoning. Topic: {req.topic}"}]
        against_history = [{"role": "system", "content":
            f"You are arguing STRONGLY AGAINST the following topic. "
            f"Be persuasive, direct, and use concrete reasoning. Topic: {req.topic}"}]

        for round_num in range(1, req.rounds + 1):
            for_history.append({"role": "user", "content":
                "Make your argument." if round_num == 1
                else "Respond to your opponent's argument and reinforce your position."})
            try:
                for_response = await _complete(for_history)
            except Exception as exc:
                yield f"data: {json.dumps({'side': 'for', 'round': round_num, 'error': str(exc)})}\n\n"
                return
            for_history.append({"role": "assistant", "content": for_response})

            payload = {"side": "for", "round": round_num, "content": ""}
            for word in for_response.split(" "):
                payload["content"] = word + " "
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)

            against_history.append({"role": "user", "content":
                f"Your opponent just argued:\n\n{for_response}\n\n"
                + ("Make your opening argument against the topic."
                   if round_num == 1
                   else "Counter their argument and reinforce your position.")})
            try:
                against_response = await _complete(against_history)
            except Exception as exc:
                yield f"data: {json.dumps({'side': 'against', 'round': round_num, 'error': str(exc)})}\n\n"
                return
            against_history.append({"role": "assistant", "content": against_response})

            payload = {"side": "against", "round": round_num, "content": ""}
            for word in against_response.split(" "):
                payload["content"] = word + " "
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)

            for_history.append({"role": "user", "content":
                f"Your opponent responded:\n\n{against_response}"})

        yield "data: [DONE]\n\n"

    return StreamingResponse(content=_generate(), media_type="text/event-stream")


@app.post("/constrain", tags=["inference"])
async def constrain(req: ConstrainRequest, _: None = Depends(require_api_key)) -> StreamingResponse:
    """
    Answer a prompt while obeying a set of explicit reasoning constraints.

    The constraints are injected into the system prompt so the model must
    follow them when constructing its response.  Useful for demonstrating
    how explicit rules shape reasoning output.

    SSE format: same as /v1/chat/completions (reasoning + content fields).
    """
    if app_state.vllm_state != VLLMState.RUNNING:
        raise HTTPException(status_code=503, detail=f"vLLM is not ready (state={app_state.vllm_state}).")

    model = req.model or cfg.MODEL_NAME
    logger.info("POST /constrain  constraints=%s  prompt=%.80r", req.constraints, req.prompt)

    rules = "\n".join(f"{i+1}. {c}" for i, c in enumerate(req.constraints))
    system = (
        "You must answer the user's prompt while strictly obeying ALL of the following rules:\n\n"
        f"{rules}\n\n"
        "Do not break any rule under any circumstances."
    )

    body = json.dumps({
        "model":    model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": req.prompt},
        ],
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": req.thinking},
    }).encode()

    try:
        client   = httpx.AsyncClient(timeout=None)
        upstream = await client.send(
            client.build_request(
                method=  "POST",
                url=     f"{_VLLM_BASE}/v1/chat/completions",
                headers= {"Content-Type": "application/json"},
                content= body,
            ),
            stream=True,
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Could not connect to vLLM backend.")

    async def _stream_response():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}
    return StreamingResponse(
        content=    _stream_response(),
        status_code=upstream.status_code,
        headers=    resp_headers,
        media_type= upstream.headers.get("content-type", "text/event-stream"),
    )


@app.post("/evolve", tags=["inference"])
async def evolve(req: EvolveRequest, _: None = Depends(require_api_key)) -> StreamingResponse:
    """
    Iterative response refinement.

    - First call: just send prompt — model answers normally.
    - Subsequent calls: include previous_response + feedback — model
      reads its prior answer and the user's critique, then produces
      an improved version.

    The backend is stateless; the client tracks history and sends it
    back each round.

    SSE format: same as /v1/chat/completions (reasoning + content fields).
    """
    if app_state.vllm_state != VLLMState.RUNNING:
        raise HTTPException(status_code=503, detail=f"vLLM is not ready (state={app_state.vllm_state}).")

    model = req.model or cfg.MODEL_NAME
    is_refinement = req.previous_response and req.feedback
    logger.info("POST /evolve  refinement=%s  model=%s  prompt=%.80r",
                is_refinement, model, req.prompt)

    if is_refinement:
        user_content = (
            f"Original question: {req.prompt}\n\n"
            f"Your previous answer:\n{req.previous_response}\n\n"
            f"User feedback: {req.feedback}\n\n"
            "Now write an improved answer that addresses the feedback. "
            "Think carefully about what was weak in the previous answer and fix it."
        )
        system = (
            "You are an iterative reasoner. You receive your previous answer and "
            "user feedback, then produce a meaningfully improved version. "
            "Do not just rephrase — genuinely fix the weaknesses identified."
        )
    else:
        user_content = req.prompt
        system = (
            "You are a careful reasoner. Answer the question thoroughly. "
            "Your answer may be refined in future rounds based on feedback."
        )

    body = json.dumps({
        "model":    model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": req.thinking},
    }).encode()

    try:
        client   = httpx.AsyncClient(timeout=None)
        upstream = await client.send(
            client.build_request(
                method=  "POST",
                url=     f"{_VLLM_BASE}/v1/chat/completions",
                headers= {"Content-Type": "application/json"},
                content= body,
            ),
            stream=True,
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Could not connect to vLLM backend.")

    async def _stream_response():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}
    return StreamingResponse(
        content=    _stream_response(),
        status_code=upstream.status_code,
        headers=    resp_headers,
        media_type= upstream.headers.get("content-type", "text/event-stream"),
    )


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], tags=["proxy"])
async def proxy_vllm(path: str, request: Request, _: None = Depends(require_api_key)) -> Response:
    """Transparent reverse proxy for all OpenAI-compatible /v1/* routes."""
    return await _proxy(request, f"v1/{path}")


@app.api_route("/chat", methods=["GET", "POST"], tags=["proxy"])
async def proxy_chat(request: Request, _: None = Depends(require_api_key)) -> Response:
    """/chat → forwards to vLLM /v1/chat/completions."""
    return await _proxy(request, "v1/chat/completions")


@app.api_route("/generate", methods=["GET", "POST"], tags=["proxy"])
async def proxy_generate(request: Request, _: None = Depends(require_api_key)) -> Response:
    """Legacy /generate → forwards to vLLM /v1/completions."""
    return await _proxy(request, "v1/completions")


@app.api_route("/inspect", methods=["GET", "POST"], tags=["proxy"])
async def proxy_inspect(request: Request, _: None = Depends(require_api_key)) -> Response:
    """/inspect → forwards to vLLM /v1/chat/completions."""
    return await _proxy(request, "v1/chat/completions")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "pforge.server:app",
        host=      cfg.ORCHESTRATOR_HOST,
        port=      cfg.ORCHESTRATOR_PORT,
        reload=    False,
        log_level= "info",
    )
