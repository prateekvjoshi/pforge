"""
training_runner.py — dataset normalisation and training job lifecycle.

Owns the trainer subprocess handle. All functions that touch app_state
accept it as a parameter to avoid circular imports with server.py.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pforge.config as cfg
from pforge.models import HyperParams, TrainingState, VLLMState
from pforge.vllm_manager import (
    dynamic_load_lora,
    restart_vllm,
    start_vllm,
    stop_vllm,
)

logger = logging.getLogger(__name__)

# Module-level process handle — owned by this module
_process: Optional[asyncio.subprocess.Process] = None


# ═════════════════════════════════════════════════════════════════════════════
# Dataset normalisation
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_example(ex: dict) -> dict:
    """
    Coerce one dataset example to the messages-list format.

    Supported input shapes:
      {"messages": [...]}                         — passed through unchanged
      {"instruction": "...", "output": "..."}     — Alpaca-style
      {"instruction": "...", "input": "...",
       "output": "..."}                           — Alpaca with context
    """
    if "messages" in ex:
        return ex

    if "instruction" in ex and "output" in ex:
        user_text = ex["instruction"]
        if ex.get("input"):
            user_text = f"{user_text}\n\n{ex['input']}"
        return {
            "messages": [
                {"role": "user",      "content": user_text},
                {"role": "assistant", "content": ex["output"]},
            ]
        }

    raise ValueError(
        f"Cannot normalise example — expected 'messages' or "
        f"'instruction'+'output'. Got keys: {list(ex.keys())}"
    )


def write_dataset(
    job_id:        str,
    dataset:       list,
    system_prompt: Optional[str] = None,
) -> Path:
    """
    Normalise dataset examples and write to <data_dir>/<job_id>/train.jsonl.

    If system_prompt is given, it is prepended as a system message to every
    example that does not already start with one.
    """
    data_dir = cfg.DATA_DIR / job_id
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "train.jsonl"

    with open(out_path, "w") as fh:
        for ex in dataset:
            normalised = _normalize_example(ex)
            if system_prompt:
                msgs = normalised["messages"]
                if not msgs or msgs[0]["role"] != "system":
                    normalised["messages"] = [
                        {"role": "system", "content": system_prompt},
                        *msgs,
                    ]
            fh.write(json.dumps(normalised) + "\n")

    logger.info("Wrote %d examples to %s", len(dataset), out_path)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# Training job lifecycle
# ═════════════════════════════════════════════════════════════════════════════

def _build_trainer_cmd(
    job_id:       str,
    dataset_path: Path,
    output_dir:   Path,
    hp:           HyperParams,
) -> list:
    """Build the trainer.py subprocess command with merged hyperparams."""
    trainer_script = Path(__file__).parent / "trainer.py"
    cmd = [
        sys.executable,
        str(trainer_script),
        "--model_name",       cfg.MODEL_NAME,
        "--dataset_path",     str(dataset_path),
        "--output_dir",       str(output_dir),
        "--job_id",           job_id,
        "--status_dir",       str(cfg.STATUS_DIR),
        "--epochs",           str(hp.epochs           or cfg.DEFAULT_EPOCHS),
        "--max_steps",        str(hp.max_steps         or cfg.DEFAULT_MAX_STEPS),
        "--batch_size",       str(hp.batch_size        or cfg.DEFAULT_BATCH_SIZE),
        "--grad_accumulation",str(hp.grad_accumulation or cfg.DEFAULT_GRAD_ACCUM),
        "--learning_rate",    str(hp.learning_rate     or cfg.DEFAULT_LR),
        "--max_seq_length",   str(hp.max_seq_length    or cfg.DEFAULT_MAX_SEQ_LEN),
        "--lora_rank",        str(hp.lora_rank         or cfg.DEFAULT_LORA_RANK),
        "--lora_alpha",       str(hp.lora_alpha        or cfg.DEFAULT_LORA_ALPHA),
        "--lora_dropout",     str(hp.lora_dropout      or cfg.DEFAULT_LORA_DROPOUT),
    ]
    if hp.lora_target_modules:
        cmd += ["--lora_target_modules"] + hp.lora_target_modules
    return cmd


async def run_training_job(
    job_id:       str,
    run_name:     str,
    dataset_path: Path,
    hp:           HyperParams,
    app_state,
) -> None:
    """
    Full training lifecycle coroutine.  Runs as a FastAPI BackgroundTask.

    Steps:
      1. Optionally stop vLLM (TRAIN_STOP_VLLM=true, the default).
      2. Spawn trainer.py subprocess; stream its logs to a file.
      3. Poll trainer progress JSON every 5 s and mirror into app_state.
      4. On success: register the new adapter with vLLM
         (dynamic load first, restart fallback).
      5. On failure: log the error and always ensure vLLM is back up.
    """
    global _process

    output_dir  = cfg.ADAPTERS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file    = cfg.LOGS_DIR / f"training_{job_id}.log"
    status_file = cfg.STATUS_DIR / f"{job_id}.json"

    stopped_vllm = False

    try:
        # ── 1. Optionally free VRAM ──────────────────────────────────────────
        if cfg.TRAIN_STOP_VLLM:
            logger.info(
                "TRAIN_STOP_VLLM=true: stopping vLLM before training to "
                "free VRAM for QLoRA."
            )
            await stop_vllm(app_state)
            stopped_vllm = True
            await asyncio.sleep(3)   # give the GPU allocator time to settle

        # ── 2. Launch trainer subprocess ─────────────────────────────────────
        app_state.update_training(
            state=TrainingState.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        cmd = _build_trainer_cmd(job_id, dataset_path, output_dir, hp)
        logger.info("Launching trainer: %s", " ".join(cmd))

        with open(log_file, "w") as lf:
            _process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=lf,
                stderr=lf,
                env={
                    **os.environ,
                    "HF_HOME":              str(cfg.HF_CACHE_DIR),
                    "TRANSFORMERS_CACHE":   str(cfg.HF_CACHE_DIR),
                    "HUGGINGFACE_HUB_CACHE":str(cfg.HF_CACHE_DIR / "hub"),
                },
            )

        # ── 3. Poll until completion ──────────────────────────────────────────
        while _process.returncode is None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(_process.wait()), timeout=5.0
                )
            except asyncio.TimeoutError:
                pass

            if status_file.exists():
                try:
                    prog = json.loads(status_file.read_text())
                    app_state.update_training(
                        progress_pct=prog.get("progress_pct"),
                        current_step=prog.get("step"),
                        total_steps= prog.get("total_steps"),
                        loss=        prog.get("loss"),
                    )
                except Exception:
                    pass  # transient read during an atomic write — ignore

        rc = _process.returncode
        _process = None

        if rc != 0:
            raise RuntimeError(
                f"trainer.py exited with code {rc}. "
                f"See {log_file} for details."
            )

        # ── 4. Register LoRA with vLLM ────────────────────────────────────────
        adapter_path = str(output_dir)
        lora_name    = run_name.replace(" ", "_").replace("/", "_")

        app_state.update_training(
            state=       TrainingState.SUCCEEDED,
            progress_pct=100.0,
            completed_at=datetime.now(timezone.utc),
            adapter_path=adapter_path,
        )

        if stopped_vllm or app_state.vllm_state != VLLMState.RUNNING:
            lora_modules: dict = {}
            if app_state.active_lora_name and app_state.active_lora_path:
                lora_modules[app_state.active_lora_name] = app_state.active_lora_path
            lora_modules[lora_name] = adapter_path
            logger.info(
                "Restarting vLLM with adapters: %s", list(lora_modules.keys())
            )
            await start_vllm(app_state, lora_modules=lora_modules)
        else:
            loaded = await dynamic_load_lora(lora_name, adapter_path)
            if not loaded:
                logger.info(
                    "Dynamic LoRA load failed; restarting vLLM with adapter "
                    "pre-registered."
                )
                lora_modules = {}
                if app_state.active_lora_name and app_state.active_lora_path:
                    lora_modules[app_state.active_lora_name] = app_state.active_lora_path
                lora_modules[lora_name] = adapter_path
                await restart_vllm(app_state, lora_modules=lora_modules)

        app_state.update(
            active_lora_name=lora_name,
            active_lora_path=adapter_path,
        )
        logger.info(
            "Training job %s succeeded.  Adapter available as '%s'.",
            job_id, lora_name,
        )

    except Exception as exc:
        logger.exception("Training job %s failed: %s", job_id, exc)
        app_state.update_training(
            state=       TrainingState.FAILED,
            completed_at=datetime.now(timezone.utc),
            error=       str(exc),
        )

    finally:
        # Always ensure vLLM is running when we exit — even after a failure
        if stopped_vllm and app_state.vllm_state not in (
            VLLMState.RUNNING, VLLMState.STARTING
        ):
            logger.info("Restarting vLLM after training exit (was stopped for VRAM).")
            await start_vllm(app_state)
