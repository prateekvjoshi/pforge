"""
utils.py — small, stateless helpers shared by server and trainer.
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


# ── Directory helpers ─────────────────────────────────────────────────────────

def ensure_dirs(*dirs: Path) -> None:
    """Create all directories if they don't already exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ── vLLM health probing ───────────────────────────────────────────────────────

async def probe_vllm(host: str, port: int, timeout: float = 3.0) -> bool:
    """Single GET /health probe. Returns True if vLLM responds 200."""
    url = f"http://{host}:{port}/health"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=timeout)
            return resp.status_code == 200
    except Exception:
        return False


async def wait_for_vllm(
    host: str,
    port: int,
    timeout: float = 600.0,
    poll_interval: float = 5.0,
) -> bool:
    """
    Poll vLLM /health until it responds 200 or the timeout expires.

    Typical first-load time for Qwen3 9B on a 4090: 2–4 minutes.
    The default 360-second timeout gives comfortable headroom.
    """
    deadline = time.monotonic() + timeout
    attempt  = 0
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = await client.get(
                    f"http://{host}:{port}/health", timeout=5.0
                )
                if resp.status_code == 200:
                    logger.info(f"vLLM healthy after {attempt} probe(s).")
                    return True
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
                pass
            if attempt % 6 == 0:
                elapsed = time.monotonic() - (deadline - timeout)
                logger.info(
                    f"Still waiting for vLLM ({elapsed:.0f}s elapsed, "
                    f"timeout={timeout:.0f}s)…"
                )
            await asyncio.sleep(poll_interval)
    logger.error(f"vLLM did not become healthy within {timeout:.0f}s.")
    return False


# ── LoRA module inspection (used by trainer; kept here for shared import) ─────

def list_linear_modules(model) -> List[str]:
    """
    Return all unique *leaf* names of nn.Linear layers in the model.

    "Leaf name" = the last component of the dotted path, e.g.
    "model.layers.0.self_attn.q_proj" → "q_proj".

    This is what LoraConfig.target_modules expects.
    """
    import torch.nn as nn  # local import — only needed on the training path

    seen: set = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            seen.add(name.split(".")[-1])
    return sorted(seen)


def choose_lora_targets(
    model,
    explicit: Optional[List[str]] = None,
) -> List[str]:
    """
    Select LoRA target module names for the given model.

    Strategy (in order):
    1. Use the explicit list if provided (validated against model).
    2. Walk a priority list of known patterns, return the first that fully
       exists in the model.  Covers:
         - Standard dense attention + MLP (most Qwen3 / Llama-like layers)
         - Attention-only minimal set
         - Merged QKV (Qwen 1/2 legacy style)
         - Linear-attention / DeltaNet-style projections
    3. Fall back to any leaf name containing "proj", "fc", or "dense".
    4. Absolute last resort: all linear leaf names.

    Qwen 3.5 uses a *hybrid* architecture: some transformer blocks are
    dense-attention, others are linear-attention (DeltaNet/GatedDeltaNet
    style).  The two block types use **different** projection names.
    Choosing targets that exist in *all* blocks keeps the adapter consistent.
    That is why we validate "all(m in all_leaves for m in candidate_set)" —
    we only pick a set when every module in the set is present globally.
    If no single set covers all blocks, we fall back to the union of
    projection-named layers, which will naturally skip missing ones in PEFT.
    """
    all_leaves = list_linear_modules(model)
    logger.info(f"Linear layer leaf names in model: {all_leaves}")

    if explicit:
        missing = [m for m in explicit if m not in all_leaves]
        if missing:
            raise ValueError(
                f"Requested LoRA target modules not found in model: {missing}. "
                f"Available leaf names: {all_leaves}"
            )
        logger.info(f"Using explicitly specified LoRA targets: {explicit}")
        return explicit

    # (modules_list, human_label)
    priority_sets = [
        (
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
            "dense-attention + MLP",
        ),
        (
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            "dense-attention only",
        ),
        (
            ["q_proj", "v_proj"],
            "minimal attention (q+v)",
        ),
        (
            ["c_attn", "c_proj"],
            "merged-QKV (Qwen1/2 legacy)",
        ),
        (
            ["qkv_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
            "linear-attention + MLP",
        ),
        (
            ["qkv_proj", "out_proj"],
            "linear-attention only",
        ),
        (
            ["in_proj", "out_proj"],
            "packed QKV",
        ),
    ]

    for modules, label in priority_sets:
        if all(m in all_leaves for m in modules):
            logger.info(f"Auto-selected LoRA targets ({label}): {modules}")
            return modules

    # Fallback: projection/fc/dense-named leaves
    proj = [
        n for n in all_leaves
        if any(k in n for k in ("proj", "fc", "dense", "linear"))
    ]
    if proj:
        logger.warning(
            f"No complete standard pattern matched.  "
            f"Falling back to projection-named leaves: {proj}.  "
            f"Consider passing --lora_target_modules explicitly."
        )
        return proj

    logger.warning(
        f"Could not identify standard LoRA targets; using ALL linear layers: {all_leaves}.  "
        f"This may be inefficient.  Pass --lora_target_modules to override."
    )
    return all_leaves
