"""
vllm_manager.py — vLLM subprocess lifecycle management.

Owns the vLLM process handle and provides start/stop/restart helpers.
All functions that touch app_state accept it as a parameter to avoid
circular imports with server.py.
"""

import asyncio
import logging
import os
import sys
from typing import Optional

import httpx

import pforge.config as cfg
from pforge.models import VLLMState
from pforge.utils import probe_vllm, wait_for_vllm

logger = logging.getLogger(__name__)

# Module-level process handle — owned by this module
_process: Optional[asyncio.subprocess.Process] = None


def _build_vllm_cmd(lora_modules: Optional[dict] = None) -> list:
    """
    Construct the vLLM API server command.

    Key flags:
      --trust-remote-code   required for Qwen3 custom modelling code
      --reasoning-parser qwen3  strips <think>…</think> tokens from responses
      --enable-lora         allows runtime adapter loading via the REST API
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                   cfg.MODEL_NAME,
        "--host",                    cfg.VLLM_HOST,
        "--port",                    str(cfg.VLLM_PORT),
        "--trust-remote-code",
        "--gpu-memory-utilization",  str(cfg.GPU_MEMORY_UTILIZATION),
        "--max-model-len",           str(cfg.MAX_MODEL_LEN),
        "--dtype",                   cfg.DTYPE,
    ]
    if cfg.VLLM_REASONING_PARSER:
        cmd += ["--reasoning-parser", cfg.VLLM_REASONING_PARSER]
    if cfg.QUANTIZATION:
        cmd += ["--quantization", cfg.QUANTIZATION]
    if cfg.VLLM_ENABLE_LORA:
        cmd += [
            "--enable-lora",
            "--max-loras",     str(cfg.VLLM_MAX_LORAS),
            "--max-lora-rank", str(cfg.VLLM_MAX_LORA_RANK),
        ]
        if lora_modules:
            for name, path in lora_modules.items():
                cmd += ["--lora-modules", f"{name}={path}"]
    elif lora_modules:
        logger.warning(
            "VLLM_ENABLE_LORA=false — ignoring lora_modules %s. "
            "Set VLLM_ENABLE_LORA=true once vLLM supports this model's LoRA.",
            list(lora_modules.keys()),
        )
    return cmd


async def start_vllm(app_state, lora_modules: Optional[dict] = None) -> bool:
    """
    Launch vLLM as a subprocess, wait for it to become healthy.
    Returns True if vLLM passes the health probe within the timeout.
    """
    global _process

    cmd      = _build_vllm_cmd(lora_modules)
    log_file = cfg.LOGS_DIR / "vllm.log"

    logger.info("Starting vLLM subprocess: %s", " ".join(cmd))
    app_state.update(vllm_state=VLLMState.STARTING, vllm_pid=None)

    # Rotate log if it has grown large
    if log_file.exists() and log_file.stat().st_size > 200 * 1024 * 1024:
        log_file.rename(log_file.with_suffix(".log.1"))

    try:
        with open(log_file, "ab") as lf:
            _process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=lf,
                stderr=lf,
                env={
                    **os.environ,
                    "HF_HOME":                str(cfg.HF_CACHE_DIR),
                    "TRANSFORMERS_CACHE":      str(cfg.HF_CACHE_DIR),
                    "HUGGINGFACE_HUB_CACHE":   str(cfg.HF_CACHE_DIR / "hub"),
                },
            )

        app_state.update(vllm_pid=_process.pid)
        logger.info("vLLM PID %d — waiting for /health…", _process.pid)

        healthy = await wait_for_vllm(cfg.VLLM_HOST, cfg.VLLM_PORT, timeout=360.0)
        if healthy:
            app_state.update(vllm_state=VLLMState.RUNNING)
            logger.info("vLLM is healthy and accepting requests.")
            return True

        logger.error("vLLM did not become healthy within the timeout.")
        app_state.update(vllm_state=VLLMState.ERROR)
        return False

    except Exception as exc:
        logger.exception("Failed to start vLLM: %s", exc)
        app_state.update(vllm_state=VLLMState.ERROR)
        return False


async def stop_vllm(app_state, timeout: float = 30.0) -> None:
    """Send SIGTERM to vLLM and wait for it to exit; SIGKILL on timeout."""
    global _process
    if _process is None:
        return

    pid = _process.pid
    logger.info("Stopping vLLM (PID %d)…", pid)

    try:
        _process.terminate()
        try:
            await asyncio.wait_for(_process.wait(), timeout=timeout)
            logger.info("vLLM exited cleanly.")
        except asyncio.TimeoutError:
            logger.warning("vLLM did not exit within %ss — sending SIGKILL.", timeout)
            _process.kill()
            await _process.wait()
    except ProcessLookupError:
        pass  # Process was already dead

    _process = None
    app_state.update(vllm_state=VLLMState.STOPPED, vllm_pid=None)
    logger.info("vLLM stopped.")


async def restart_vllm(app_state, lora_modules: Optional[dict] = None) -> bool:
    """Stop then start vLLM (optionally with pre-registered LoRA adapters)."""
    app_state.update(vllm_state=VLLMState.RESTARTING)
    await stop_vllm(app_state)
    await asyncio.sleep(3)   # allow GPU memory to be released
    return await start_vllm(app_state, lora_modules)


async def dynamic_load_lora(lora_name: str, lora_path: str) -> bool:
    """
    POST /v1/load_lora_adapter to vLLM to hot-load a LoRA adapter at runtime.

    Requires:
      - vLLM >= 0.4.x
      - vLLM was started with --enable-lora
      - The adapter directory contains a valid PEFT adapter_config.json

    Returns True if vLLM acknowledges the load successfully.
    """
    url     = f"http://{cfg.VLLM_HOST}:{cfg.VLLM_PORT}/v1/load_lora_adapter"
    payload = {"lora_name": lora_name, "lora_path": lora_path}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=60.0)
        if resp.status_code == 200:
            logger.info("Dynamic LoRA load accepted: %s → %s", lora_name, lora_path)
            return True
        logger.warning(
            "Dynamic LoRA load returned HTTP %d: %s",
            resp.status_code, resp.text[:200],
        )
        return False
    except Exception as exc:
        logger.warning("Dynamic LoRA load request failed: %s", exc)
        return False


async def vllm_watchdog(app_state) -> None:
    """
    Check vLLM liveness every 30 s.
    If it has crashed unexpectedly and no training is active, restart it.
    """
    from pforge.models import TrainingState

    await asyncio.sleep(45)  # initial delay to let vLLM come up first
    while True:
        await asyncio.sleep(30)
        if app_state.vllm_state != VLLMState.RUNNING:
            continue
        if app_state.training.state in (TrainingState.RUNNING, TrainingState.QUEUED):
            continue   # training manages vLLM lifecycle itself

        alive = await probe_vllm(cfg.VLLM_HOST, cfg.VLLM_PORT)
        if not alive:
            logger.warning("Watchdog: vLLM health probe failed — restarting.")
            lora_modules = {}
            if app_state.active_lora_name and app_state.active_lora_path:
                lora_modules[app_state.active_lora_name] = app_state.active_lora_path
            app_state.update(vllm_state=VLLMState.ERROR)
            await start_vllm(app_state, lora_modules or None)
