"""
config.py — all configuration via environment variables.

Precedence (highest to lowest):
  CLI flags  →  PR_* env vars  →  legacy env vars  →  built-in defaults

Every value has a safe default so the service starts without any env vars set.
New code should use PR_* variable names. Legacy names are still supported for
backward compatibility and will be removed in a future version.
"""
import os
from pathlib import Path
from typing import Optional

from pforge.paths import resolve_data_dir


def _env(new_key: str, legacy_key: Optional[str], default: str) -> str:
    """Read new PR_* key, fall back to legacy key, then default."""
    v = os.environ.get(new_key)
    if v is not None:
        return v
    if legacy_key:
        v = os.environ.get(legacy_key)
        if v is not None:
            return v
    return default


# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR = resolve_data_dir()

LOGS_DIR                 = DATA_DIR / "logs"
ADAPTERS_DIR             = DATA_DIR / "adapters"
TRAINING_DATA_DIR        = DATA_DIR / "data"
STATUS_DIR               = DATA_DIR / "status"
HF_CACHE_DIR             = DATA_DIR / "hf_cache"
SERVER_STATUS_FILE       = DATA_DIR / "server_status.json"

# Legacy aliases — keeps existing code working without changes
ORCHESTRATOR_STATUS_FILE = SERVER_STATUS_FILE
DATA_DIR                 = TRAINING_DATA_DIR   # cfg.DATA_DIR used in server.py

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = _env("PR_MODEL",      "MODEL_NAME",  "Qwen/Qwen3-1.7B")
HF_TOKEN   = _env("PR_HF_TOKEN",   "HF_TOKEN",    "")

# ── vLLM ─────────────────────────────────────────────────────────────────────
VLLM_HOST              = _env("PR_VLLM_HOST",   "VLLM_HOST",              "127.0.0.1")
VLLM_PORT              = int(_env("PR_VLLM_PORT", "VLLM_PORT",            "8002"))
MAX_MODEL_LEN          = int(_env("PR_MAX_MODEL_LEN", "MAX_MODEL_LEN",    "8192"))
GPU_MEMORY_UTILIZATION = float(_env("PR_GPU_MEMORY_UTILIZATION",
                                    "GPU_MEMORY_UTILIZATION",             "0.80"))
DTYPE                  = _env("PR_DTYPE",        "DTYPE",                 "bfloat16")
QUANTIZATION           = _env("PR_QUANTIZATION", "QUANTIZATION",          "")

# vLLM LoRA: true for pure LM models (Qwen3). Set false for multimodal models
# where vLLM LoRA is unstable.
VLLM_ENABLE_LORA       = _env("PR_VLLM_ENABLE_LORA", "VLLM_ENABLE_LORA", "true").lower() == "true"

# Reasoning parser: "qwen3" for Qwen3 models; "" to disable.
VLLM_REASONING_PARSER  = _env("PR_VLLM_REASONING_PARSER",
                               "VLLM_REASONING_PARSER",                   "qwen3")
VLLM_MAX_LORAS         = int(_env("PR_VLLM_MAX_LORAS",  "VLLM_MAX_LORAS",    "4"))
VLLM_MAX_LORA_RANK     = int(_env("PR_VLLM_MAX_LORA_RANK", "VLLM_MAX_LORA_RANK", "64"))

# ── Server ────────────────────────────────────────────────────────────────────
ORCHESTRATOR_HOST = _env("PR_HOST", "ORCHESTRATOR_HOST", "127.0.0.1")
ORCHESTRATOR_PORT = int(_env("PR_PORT", "ORCHESTRATOR_PORT", "8000"))

# API key. Leave empty to disable authentication (local development).
# Generate: python3 -c "import secrets; print(secrets.token_hex(32))"
ORCHESTRATOR_API_KEY = _env("PR_API_KEY", "ORCHESTRATOR_API_KEY", "")

# CORS: comma-separated allowed browser origins. Leave empty to block all.
_cors_raw = _env("PR_CORS_ORIGINS", "CORS_ALLOWED_ORIGINS", "")
CORS_ALLOWED_ORIGINS: list = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# Rate limiting: requests per minute for heavy/destructive endpoints.
RATE_LIMIT_HEAVY = int(_env("PR_RATE_LIMIT_HEAVY", "RATE_LIMIT_HEAVY", "5"))
RATE_LIMIT_OPS   = int(_env("PR_RATE_LIMIT_OPS",   "RATE_LIMIT_OPS",   "10"))

# ── Training ──────────────────────────────────────────────────────────────────
# Stop vLLM during training to free VRAM for QLoRA. Recommended for most setups.
# Set false only if your GPU has enough VRAM for both simultaneously.
TRAIN_STOP_VLLM = _env("PR_TRAIN_STOP_SERVER", "TRAIN_STOP_VLLM", "true").lower() == "true"

DEFAULT_EPOCHS       = int(_env("PR_DEFAULT_EPOCHS",    "DEFAULT_EPOCHS",      "1"))
DEFAULT_MAX_STEPS    = int(_env("PR_DEFAULT_STEPS",     "DEFAULT_MAX_STEPS",   "50"))
DEFAULT_BATCH_SIZE   = int(_env("PR_DEFAULT_BATCH",     "DEFAULT_BATCH_SIZE",  "1"))
DEFAULT_GRAD_ACCUM   = int(_env("PR_DEFAULT_GRAD_ACCUM","DEFAULT_GRAD_ACCUM",  "4"))
DEFAULT_LR           = float(_env("PR_DEFAULT_LR",      "DEFAULT_LR",          "2e-4"))
DEFAULT_MAX_SEQ_LEN  = int(_env("PR_DEFAULT_SEQ_LEN",   "DEFAULT_MAX_SEQ_LEN", "1024"))
DEFAULT_LORA_RANK    = int(_env("PR_DEFAULT_LORA_RANK",  "DEFAULT_LORA_RANK",  "8"))
DEFAULT_LORA_ALPHA   = int(_env("PR_DEFAULT_LORA_ALPHA", "DEFAULT_LORA_ALPHA", "16"))
DEFAULT_LORA_DROPOUT = float(_env("PR_DEFAULT_LORA_DROPOUT",
                                  "DEFAULT_LORA_DROPOUT",                      "0.05"))
