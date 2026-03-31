"""
trainer.py — standalone QLoRA fine-tuning subprocess.

Run directly (e.g. for debugging):
  python pforge/trainer.py \\
      --model_name Qwen/Qwen3-1.7B \\
      --dataset_path ~/.local/share/pforge/data/abc123/train.jsonl \\
      --output_dir  ~/.local/share/pforge/adapters/abc123 \\
      --job_id abc123 --max_steps 50

Debug mode (prints all linear module names, then exits):
  python pforge/trainer.py --model_name Qwen/Qwen3-1.7B --list_modules

Design notes
────────────
QLoRA memory footprint:
  - Base model in NF4 4-bit:     ~2–3 GB  (1.7B params)
  - Activations + gradients:     ~4–6 GB  (depends on seq len and batch)
  - Optimizer states (AdamW):    ~1–2 GB  (only for LoRA params, which are tiny)
  Total comfortable budget:      ~8–10 GB
  → Fits on most 12 GB+ GPUs when the inference server is stopped first.

Qwen3 hybrid architecture
──────────────────────────
Qwen3 uses interleaved dense-attention and linear-attention (GatedDeltaNet-style)
transformer blocks. The two block types use different projection names:
  Dense blocks:  q_proj, k_proj, v_proj, o_proj  (standard)
  Linear blocks: qkv_proj, out_proj              (fused)
  MLP (shared):  gate_proj, up_proj, down_proj

At load time we inspect all nn.Linear leaf names, then choose the largest
candidate set that exists in *all* blocks. See utils.choose_lora_targets().

Progress reporting
──────────────────
A JsonProgressCallback writes a status JSON to <status_dir>/<job_id>.json
after every logging step. The server polls this file every 5 s.
stdout is captured by the server subprocess to <logs_dir>/training_<job_id>.log.
"""

import argparse
import gc
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch
from transformers import TrainerCallback

from pforge.utils import choose_lora_targets, list_linear_modules

# ── Logging (stdout — server captures this into the log file) ────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainer")


# ═════════════════════════════════════════════════════════════════════════════
# Progress callback
# ═════════════════════════════════════════════════════════════════════════════

class JsonProgressCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that writes machine-readable JSON progress to
    <status_dir>/<job_id>.json after every logging step.

    The server reads this file every 5 s to update its in-memory state.
    Writes are atomic (tmp → rename) to prevent torn reads.
    """

    def __init__(self, status_file: Path, total_steps: int, job_id: str):
        self.status_file = status_file
        self.total_steps = max(total_steps, 1)
        self.job_id      = job_id
        status_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, step: int, epoch: float, extra: dict) -> None:
        pct = min((step / self.total_steps) * 100.0, 99.9)
        payload = {
            "job_id":      self.job_id,
            "state":       "running",
            "step":        step,
            "total_steps": self.total_steps,
            "progress_pct":round(pct, 1),
            "epoch":       round(epoch, 3),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        tmp = self.status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(self.status_file)
        # Also emit to stdout so the server log has a full trace
        logger.info("PROGRESS %s", json.dumps(payload, default=str))

    def on_train_begin(self, args, state, control, **kwargs):
        self._write(0, 0.0, {"note": "training started"})

    def on_log(self, args, state, control, logs=None, **kwargs):
        extra: dict = {}
        if logs:
            for key in ("loss", "learning_rate", "grad_norm"):
                if key in logs:
                    extra[key] = logs[key]
        self._write(state.global_step, state.epoch or 0.0, extra)

    def on_train_end(self, args, state, control, **kwargs):
        payload = {
            "job_id":      self.job_id,
            "state":       "succeeded",
            "step":        state.global_step,
            "total_steps": self.total_steps,
            "progress_pct":100.0,
            "epoch":       round(state.epoch or 0.0, 3),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(self.status_file)
        logger.info("PROGRESS %s", json.dumps(payload))


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path):
    """Load a JSONL file into a HuggingFace Dataset."""
    from datasets import Dataset

    records: List[dict] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i+1} of {path}: {exc}")

    if not records:
        raise ValueError(f"Dataset at {path} is empty.")

    logger.info("Loaded %d examples from %s", len(records), path)
    return Dataset.from_list(records)


def _apply_chat_template(example: dict, tokenizer, max_length: int) -> dict:
    """
    Apply the tokenizer's chat template to a messages-format example and
    return {"text": formatted_string}.

    We truncate at the character level (max_length * 4 is a safe upper bound
    for token-to-char ratio) to avoid feeding enormous strings into the
    tokeniser's internal fast path.
    """
    messages = example.get("messages")
    if not messages:
        raise ValueError(f"Example missing 'messages': {example}")

    text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # Rough character truncation — the tokeniser will do proper truncation
    max_chars = max_length * 6
    if len(text) > max_chars:
        text = text[:max_chars]

    return {"text": text}


# ═════════════════════════════════════════════════════════════════════════════
# Main train()
# ═════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> Path:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    logger.info("=== Training job %s ===", args.job_id)
    logger.info("Args: %s", vars(args))

    status_file = Path(args.status_dir) / f"{args.job_id}.json"

    # ── Compute dtype ──────────────────────────────────────────────────────
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        dtype_str     = "bfloat16"
    elif torch.cuda.is_available():
        compute_dtype = torch.float16
        dtype_str     = "float16"
    else:
        # CPU fallback (for local testing only; will be very slow)
        compute_dtype = torch.float32
        dtype_str     = "float32"
    logger.info("Compute dtype: %s", dtype_str)

    # ── BitsAndBytes 4-bit config ──────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=              True,
        bnb_4bit_quant_type=       "nf4",
        bnb_4bit_compute_dtype=    compute_dtype,
        bnb_4bit_use_double_quant= True,   # nested quant saves ~0.4 bits/param extra
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",   # required for SFT loss masking
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token (%s)", tokenizer.eos_token)

    # ── Model in 4-bit ─────────────────────────────────────────────────────
    logger.info("Loading model in QLoRA 4-bit mode: %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=          "auto",
        trust_remote_code=   True,
        torch_dtype=         compute_dtype,
        # Disable flash attention during training to avoid compatibility issues
        # with gradient checkpointing on some PEFT versions.
        # Re-enable if you validate compatibility: attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False  # required when gradient_checkpointing=True

    # Emit full module list for debugging
    all_linears = list_linear_modules(model)
    logger.info("=== ALL LINEAR MODULE LEAF NAMES ===\n%s", json.dumps(all_linears, indent=2))

    # ── Prepare for QLoRA ──────────────────────────────────────────────────
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── Choose LoRA targets ────────────────────────────────────────────────
    lora_targets = choose_lora_targets(
        model,
        explicit=args.lora_target_modules if args.lora_target_modules else None,
    )

    lora_config = LoraConfig(
        r=              args.lora_rank,
        lora_alpha=     args.lora_alpha,
        lora_dropout=   args.lora_dropout,
        target_modules= lora_targets,
        bias=           "none",
        task_type=      TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = load_jsonl(Path(args.dataset_path))

    dataset = dataset.map(
        lambda ex: _apply_chat_template(ex, tokenizer, args.max_seq_length),
        remove_columns=[c for c in dataset.column_names if c != "text"],
        desc="Applying chat template",
        load_from_cache_file=False,
    )
    n_examples = len(dataset)
    logger.info("Preprocessed dataset: %d examples", n_examples)

    # ── Effective total steps ──────────────────────────────────────────────
    steps_per_epoch    = max(1, n_examples // (args.batch_size * args.grad_accumulation))
    epoch_steps        = steps_per_epoch * args.epochs
    effective_steps    = min(args.max_steps, epoch_steps) if args.max_steps > 0 else epoch_steps
    logging_steps      = max(1, effective_steps // 10)

    logger.info(
        "Training plan: examples=%d, epochs=%d, steps_per_epoch=%d, "
        "max_steps_arg=%d, effective_steps=%d",
        n_examples, args.epochs, steps_per_epoch, args.max_steps, effective_steps,
    )

    # ── Training arguments ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _sft_base = dict(
        output_dir=                     str(output_dir),
        num_train_epochs=               args.epochs,
        max_steps=                      args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=    args.batch_size,
        gradient_accumulation_steps=    args.grad_accumulation,
        learning_rate=                  args.learning_rate,
        lr_scheduler_type=              "cosine",
        warmup_ratio=                   0.05,
        fp16=                           (compute_dtype == torch.float16),
        bf16=                           (compute_dtype == torch.bfloat16),
        gradient_checkpointing=         True,
        gradient_checkpointing_kwargs=  {"use_reentrant": False},
        logging_steps=                  logging_steps,
        save_strategy=                  "no",
        report_to=                      "none",
        remove_unused_columns=          False,
        dataloader_num_workers=         0,
        packing=                        False,
        dataset_text_field=             "text",
    )
    # TRL < 0.12 uses max_seq_length; TRL >= 0.12 renamed it to max_length
    try:
        sft_config = SFTConfig(**_sft_base, max_seq_length=args.max_seq_length)
    except TypeError:
        logger.info("max_seq_length not accepted by SFTConfig, trying max_length (TRL >= 0.12)")
        sft_config = SFTConfig(**_sft_base, max_length=args.max_seq_length)

    progress_cb = JsonProgressCallback(
        status_file= status_file,
        total_steps= effective_steps,
        job_id=      args.job_id,
    )

    # TRL < 0.12 uses tokenizer=; TRL >= 0.12 renamed it to processing_class=
    _trainer_kwargs = dict(
        model=         model,
        train_dataset= dataset,
        args=          sft_config,
        callbacks=     [progress_cb],
    )
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, **_trainer_kwargs)
    except TypeError:
        logger.info("tokenizer= not accepted by SFTTrainer, trying processing_class= (TRL >= 0.12)")
        trainer = SFTTrainer(processing_class=tokenizer, **_trainer_kwargs)

    # ── Train ──────────────────────────────────────────────────────────────
    logger.info("Starting training…")
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "CUDA OOM during training.  Suggestions:\n"
            "  - Reduce --batch_size (currently %d)\n"
            "  - Reduce --max_seq_length (currently %d)\n"
            "  - Reduce --lora_rank (currently %d)\n"
            "  - Ensure TRAIN_STOP_VLLM=true so vLLM is not competing for VRAM.",
            args.batch_size, args.max_seq_length, args.lora_rank,
        )
        raise

    # ── Save adapter ───────────────────────────────────────────────────────
    logger.info("Saving adapter to %s…", output_dir)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "job_id":               args.job_id,
        "base_model":           args.model_name,
        "lora_rank":            args.lora_rank,
        "lora_alpha":           args.lora_alpha,
        "lora_target_modules":  lora_targets,
        "compute_dtype":        dtype_str,
        "max_seq_length":       args.max_seq_length,
        "n_examples":           n_examples,
        "effective_steps":      effective_steps,
        "completed_at":         datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Adapter saved.  Metadata: %s", json.dumps(meta, indent=2))

    return output_dir


# ═════════════════════════════════════════════════════════════════════════════
# GPU cleanup
# ═════════════════════════════════════════════════════════════════════════════

def cleanup_gpu() -> None:
    """
    Release CUDA memory before exiting.

    The server waits for this process to exit before restarting vLLM.
    Explicitly calling empty_cache() here makes the VRAM release faster and
    more predictable than waiting for Python's GC.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("Post-cleanup CUDA memory allocated: %.2f GB", allocated_gb)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA flash-tuning trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--model_name",    required=True, help="HF model name or local path")
    p.add_argument("--dataset_path",  required=True, help="Path to JSONL training file")
    p.add_argument("--output_dir",    required=True, help="Where to save the LoRA adapter")
    p.add_argument("--job_id",        required=True, help="Unique job identifier")

    # Paths
    p.add_argument("--status_dir", default=None,
                   help="Directory for JSON progress status files (default: from config)")

    # Hyperparams
    p.add_argument("--epochs",           type=int,   default=1)
    p.add_argument("--max_steps",        type=int,   default=50,
                   help="Max gradient steps.  0 = use --epochs only.")
    p.add_argument("--batch_size",       type=int,   default=1)
    p.add_argument("--grad_accumulation",type=int,   default=4)
    p.add_argument("--learning_rate",    type=float, default=2e-4)
    p.add_argument("--max_seq_length",   type=int,   default=1024)

    # LoRA
    p.add_argument("--lora_rank",           type=int,   default=8)
    p.add_argument("--lora_alpha",          type=int,   default=16)
    p.add_argument("--lora_dropout",        type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="*",  default=None,
                   help="Explicit list of linear module leaf names to target with LoRA. "
                        "If omitted, auto-detected from the model architecture.")

    # Debug
    p.add_argument("--list_modules", action="store_true",
                   help="Load model, print all linear module leaf names, and exit. "
                        "Use this to debug LoRA target selection on a new model.")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # Resolve status_dir default here (not at parse time) so config is importable standalone
    if args.status_dir is None:
        from pforge.config import STATUS_DIR
        args.status_dir = str(STATUS_DIR)
    exit_code = 0

    try:
        if args.list_modules:
            _run_list_modules(args)
            return

        train(args)

    except Exception:
        logger.error("Training failed:\n%s", traceback.format_exc())
        exit_code = 1

    finally:
        cleanup_gpu()
        logger.info("Trainer exiting with code %d.", exit_code)
        sys.exit(exit_code)


def _run_list_modules(args: argparse.Namespace) -> None:
    """--list_modules mode: load model, print linear names, exit."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    logger.info("Loading model for module inspection (--list_modules)…")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    names = list_linear_modules(model)
    print("\n=== CANDIDATE LoRA TARGET MODULES ===")
    for n in names:
        print(f"  {n}")
    print(
        "\nPass these names via --lora_target_modules to override auto-detection.\n"
        "Example: --lora_target_modules q_proj v_proj gate_proj up_proj down_proj"
    )
    del model
    cleanup_gpu()


if __name__ == "__main__":
    main()
