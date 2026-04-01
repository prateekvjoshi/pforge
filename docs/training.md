# Training

pforge supports lightweight LoRA fine-tuning via QLoRA (4-bit quantization). You can train a style or behavior adapter from as few as 5–20 examples.

---

## Quick start

```bash
pforge train dataset.jsonl --run-name my-style --steps 50
```

Or with the current API directly:

```bash
curl -X POST http://localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d '{
    "run_name": "my-style",
    "dataset": [...],
    "hyperparams": {"max_steps": 50}
  }'
```

---

## Dataset formats

Two formats are accepted. Both are normalized to the messages format before training.

### Chat format (recommended)

```json
{
  "messages": [
    {"role": "system",    "content": "You are a concise math tutor."},
    {"role": "user",      "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 = 4."}
  ]
}
```

The `system` message is optional.

### Instruction format

```json
{
  "instruction": "Translate to French",
  "input":       "Hello, world!",
  "output":      "Bonjour, monde!"
}
```

`input` is optional — omit it for instruction-only tasks.

### JSONL file

Save examples as one JSON object per line:

```
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

## Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_steps` | `50` | 1–1000 | Training steps |
| `epochs` | `1` | 1–10 | Epochs (alternative to max_steps) |
| `learning_rate` | `0.0002` | 1e-6–0.01 | Learning rate |
| `lora_rank` | `8` | 2–64 | LoRA rank — higher = more capacity, more VRAM |
| `lora_alpha` | `16` | 1–128 | LoRA alpha scaling |
| `lora_dropout` | `0.05` | 0–0.5 | LoRA dropout |
| `max_seq_length` | `1024` | 64–4096 | Max sequence length |
| `batch_size` | `1` | 1–8 | Batch size per step |
| `grad_accumulation` | `4` | 1–32 | Gradient accumulation steps |

---

## VRAM requirements

Training requires stopping the inference server by default (`TRAIN_STOP_SERVER=true`). This frees GPU memory for the trainer.

| Setup | Approximate VRAM |
|-------|-----------------|
| Qwen3-1.7B QLoRA (rank 8) | ~8–10 GB |
| Qwen3-7B QLoRA (rank 8) | ~14–18 GB |

If you have enough VRAM to run both simultaneously, set `TRAIN_STOP_SERVER=false`.

---

## LoRA target modules

The trainer auto-detects which modules to target based on the model architecture. For Qwen3:

- Dense attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Linear attention: `qkv_proj`, `out_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

Override with `lora_target_modules` in the hyperparams:

```json
"hyperparams": {
  "lora_target_modules": ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
}
```

To see what your model exposes:

```bash
python backend/trainer.py \
  --model_name Qwen/Qwen3-1.7B \
  --dataset_path /dev/null \
  --output_dir /tmp \
  --job_id debug \
  --list_modules
```

---

## Using a trained adapter

After training, the adapter is automatically loaded into the inference server. Use it by name:

```bash
pforge compare --model-b my-style --prompt "Describe a coffee shop"
```

Or via API:

```json
{"model": "my-style", "messages": [...]}
```

---

## Adapters on disk

Trained adapters are saved to `<data_dir>/adapters/<job_id>/`.

Each adapter directory contains:
- `adapter_config.json` — LoRA config
- `training_meta.json` — base model, run name, hyperparams, timestamps
- `adapter_model.safetensors` — the weights

---

## Known limitations

- **One job at a time.** A second training request while one is running returns HTTP 409.
- **No checkpointing.** The adapter is only saved at the end. If training crashes, the run is lost.
- **Inference unavailable during training** (with `TRAIN_STOP_SERVER=true`).
- **Single GPU only.** Multi-GPU training is not currently supported.
