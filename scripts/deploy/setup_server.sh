#!/usr/bin/env bash
# =============================================================================
# scripts/deploy/setup_server.sh — optional server deployment setup
#
# Use this when deploying to a headless GPU server (e.g. a rented cloud GPU)
# where you want a persistent venv and data directory under a shared workspace.
#
# For local development on your own machine, just:
#   pip install -e .
#   pforge serve
#
# Usage (first time or returning to same server):
#   cd /path/to/pforge
#   bash scripts/deploy/setup_server.sh
#
# On a returning server most steps are skipped because the venv already exists.
# Total re-run time on a warm server: ~10 seconds.
# =============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV="$WORKSPACE/venv"

echo "=============================================="
echo "  Programmable Reasoning — Server Setup"
echo "  Workspace : $WORKSPACE"
echo "  Repo      : $REPO_DIR"
echo "  Venv      : $VENV"
echo "=============================================="
echo

# ── Step 1 · Create workspace directories ────────────────────────────────────
echo "[1/8] Creating workspace directories…"
mkdir -p \
  "$WORKSPACE/hf_cache/hub" \
  "$WORKSPACE/hf_cache/datasets" \
  "$WORKSPACE/torch_cache" \
  "$WORKSPACE/pip_cache" \
  "$WORKSPACE/logs" \
  "$WORKSPACE/adapters" \
  "$WORKSPACE/data" \
  "$WORKSPACE/status" \
  "$WORKSPACE/tmp"
echo "  OK"

# ── Step 2 · Route pip/build temps to /workspace ──────────────────────────────
# Avoids disk-quota errors on the small root overlay filesystem.
echo "[2/8] Routing pip and temp dirs to /workspace…"
export TMPDIR="$WORKSPACE/tmp"
export PIP_CACHE_DIR="$WORKSPACE/pip_cache"
export PIP_NO_CACHE_DIR=0          # use the cache so re-runs are fast
export HF_HOME="$WORKSPACE/hf_cache"
export TRANSFORMERS_CACHE="$WORKSPACE/hf_cache"
export HUGGINGFACE_HUB_CACHE="$WORKSPACE/hf_cache/hub"
export HF_DATASETS_CACHE="$WORKSPACE/hf_cache/datasets"
export TORCH_HOME="$WORKSPACE/torch_cache"
export WORKSPACE_DIR="$WORKSPACE"
echo "  OK"

# ── Step 3 · Write persistent .env ────────────────────────────────────────────
echo "[3/8] Writing $WORKSPACE/.env…"
cat > "$WORKSPACE/.env" << ENVEOF
# =============================================================================
# Programmable Reasoning — runtime environment
# Sourced automatically by: source /workspace/.env
# =============================================================================

# ── Activate the persistent venv ─────────────────────────────────────────────
source $VENV/bin/activate

# ── Pip / build temps → /workspace (avoids root-disk quota) ──────────────────
export TMPDIR=$WORKSPACE/tmp
export PIP_CACHE_DIR=$WORKSPACE/pip_cache

# ── HuggingFace / PyTorch caches → /workspace ────────────────────────────────
export HF_HOME=$WORKSPACE/hf_cache
export TRANSFORMERS_CACHE=$WORKSPACE/hf_cache
export HUGGINGFACE_HUB_CACHE=$WORKSPACE/hf_cache/hub
export HF_DATASETS_CACHE=$WORKSPACE/hf_cache/datasets
export TORCH_HOME=$WORKSPACE/torch_cache
export WORKSPACE_DIR=$WORKSPACE

# ── Model ─────────────────────────────────────────────────────────────────────
export MODEL_NAME="Qwen/Qwen3-1.7B"
# export QUANTIZATION="awq"   # uncomment for AWQ-quantized checkpoint

# ── vLLM ──────────────────────────────────────────────────────────────────────
export VLLM_HOST="0.0.0.0"
export VLLM_PORT="8002"            # 8001 is often taken by system services
export MAX_MODEL_LEN="8192"
export GPU_MEMORY_UTILIZATION="0.80"
export DTYPE="bfloat16"
export QUANTIZATION=""
export VLLM_MAX_LORAS="4"
export VLLM_MAX_LORA_RANK="64"

# ── Orchestrator ──────────────────────────────────────────────────────────────
export ORCHESTRATOR_HOST="0.0.0.0"
export ORCHESTRATOR_PORT="8000"

# ── Training ──────────────────────────────────────────────────────────────────
export TRAIN_STOP_VLLM="true"
export DEFAULT_EPOCHS="1"
export DEFAULT_MAX_STEPS="50"
export DEFAULT_BATCH_SIZE="1"
export DEFAULT_GRAD_ACCUM="4"
export DEFAULT_LR="2e-4"
export DEFAULT_MAX_SEQ_LEN="1024"
export DEFAULT_LORA_RANK="8"
export DEFAULT_LORA_ALPHA="16"
export DEFAULT_LORA_DROPOUT="0.05"

# ── Secrets (API keys, tokens) ────────────────────────────────────────────────
# Keep secrets in a separate file that setup never overwrites.
# See .env.example in the repo for all available options.
if [ -f $WORKSPACE/.env.secrets ]; then
  source $WORKSPACE/.env.secrets
fi
ENVEOF
echo "  Written to $WORKSPACE/.env"

# ── Step 3b · Create .env.secrets if it doesn't exist ────────────────────────
if [ ! -f "$WORKSPACE/.env.secrets" ]; then
  echo "[3b] Creating $WORKSPACE/.env.secrets (fill in your secrets here)…"
  cat > "$WORKSPACE/.env.secrets" << 'SECRETSEOF'
# =============================================================================
# Programmable Reasoning — secrets (never overwritten by setup_server.sh)
# =============================================================================

# API key — generate with: python3 -c "import secrets; print(secrets.token_hex(32))"
export ORCHESTRATOR_API_KEY=""

# HuggingFace token (required for gated models)
# export HF_TOKEN="hf_..."

# CORS allowed origins (comma-separated)
# export CORS_ALLOWED_ORIGINS="https://myapp.com"

# Rate limits (requests per minute)
export RATE_LIMIT_HEAVY="5"
export RATE_LIMIT_OPS="10"
SECRETSEOF
  echo "  Created. Edit $WORKSPACE/.env.secrets to set your API key."
else
  echo "[3b] $WORKSPACE/.env.secrets already exists — preserving your secrets."
fi

# ── Step 4 · Create venv in /workspace (skip if already exists) ───────────────
echo "[4/8] Setting up Python venv in $VENV…"
if [ -d "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
  echo "  Venv already exists — skipping creation."
else
  python3 -m venv "$VENV"
  echo "  Created fresh venv."
fi
# Activate for the rest of this script
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "  Active Python: $(which python) ($(python --version))"

# ── Step 5 · Upgrade pip / build tools inside venv ───────────────────────────
echo "[5/8] Upgrading pip, setuptools, wheel…"
pip install --upgrade pip setuptools wheel --quiet
echo "  OK"

# ── Step 6 · Install PyTorch (skip if already installed) ─────────────────────
echo "[6/8] Installing PyTorch…"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "  PyTorch already installed — skipping."
  python -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"
else
  pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet
  python -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"
fi

# ── Step 7 · Install vLLM (skip if already installed) ────────────────────────
echo "[7/8] Installing vLLM…"
if python -c "import vllm" 2>/dev/null; then
  echo "  vLLM already installed — skipping."
  python -c "import vllm; print(f'  vLLM {vllm.__version__}')"
else
  # Stable release. Switch to nightly if Qwen3.5 architecture is unsupported:
  # pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --quiet
  pip install vllm --quiet
  python -c "import vllm; print(f'  vLLM {vllm.__version__}')"
fi

# ── Step 8 · Install project requirements + package ──────────────────────────
echo "[8/8] Installing project requirements…"
pip install -r "$REPO_DIR/requirements.txt" --quiet
pip install -e "$REPO_DIR" --quiet --no-deps
echo "  OK"

# ── Done ─────────────────────────────────────────────────────────────────────
echo
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo
echo "On this server and every future server using the same data volume:"
echo
echo "  1.  Set your API key (first time only):"
echo "        nano $WORKSPACE/.env.secrets"
echo
echo "  2.  Source the environment (activates venv + sets all vars):"
echo "        source /workspace/.env"
echo
echo "  3.  Launch:"
echo "        cd $REPO_DIR && pforge serve"
echo
echo "  4.  Watch vLLM load (2–5 min on first run, instant after model is cached):"
echo "        tail -f /workspace/logs/vllm.log"
echo
echo "  5.  Check health:"
echo "        curl http://localhost:8000/health"
echo
