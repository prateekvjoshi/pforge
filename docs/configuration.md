# Configuration

## Precedence

Settings are resolved in this order (highest to lowest priority):

1. CLI flags (e.g. `pforge serve --port 8080`)
2. Environment variables (e.g. `PFORGE_PORT=8080`)
3. Config file (`~/.config/pforge/config.toml`)
4. Built-in defaults

---

## Config file

`pforge init` creates a default config at `~/.config/pforge/config.toml`.

```toml
[model]
name = "Qwen/Qwen3-1.7B"
dtype = "bfloat16"
quantization = ""             # "awq" or "gptq" for quantized checkpoints

[server]
host = "127.0.0.1"
port = 8000
api_key = ""                  # leave empty to disable auth

[vllm]
port = 8002
gpu_memory_utilization = 0.80
max_model_len = 8192
max_loras = 4
max_lora_rank = 64

[training]
stop_server_during_training = true
default_steps = 50
default_lora_rank = 8
default_lora_alpha = 16
default_learning_rate = 0.0002
default_max_seq_length = 1024

[paths]
data_dir = ""                 # default: ~/.local/share/pforge
```

---

## Environment variables

All settings can be overridden with environment variables. The prefix is `PFORGE_` for most settings.

### Model

| Variable | Default | Description |
|----------|---------|-------------|
| `PFORGE_MODEL` | `Qwen/Qwen3-1.7B` | Model name (HuggingFace ID or local path) |
| `PFORGE_DTYPE` | `bfloat16` | Model dtype: `bfloat16`, `float16`, `float32` |
| `PFORGE_QUANTIZATION` | *(empty)* | `awq` or `gptq` for quantized models |
| `HF_TOKEN` | *(empty)* | HuggingFace token for gated models |

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `PFORGE_HOST` | `127.0.0.1` | API server bind address |
| `PFORGE_PORT` | `8000` | API server port |
| `PFORGE_API_KEY` | *(empty)* | API key — leave empty to disable auth |
| `PFORGE_CORS_ORIGINS` | *(empty)* | Comma-separated allowed CORS origins |

### vLLM

| Variable | Default | Description |
|----------|---------|-------------|
| `PFORGE_VLLM_PORT` | `8002` | Internal vLLM port |
| `PFORGE_GPU_MEMORY_UTILIZATION` | `0.80` | Fraction of VRAM for vLLM |
| `PFORGE_MAX_MODEL_LEN` | `8192` | Max context length in tokens |

### Training

| Variable | Default | Description |
|----------|---------|-------------|
| `PFORGE_TRAIN_STOP_SERVER` | `true` | Stop vLLM during training to free VRAM |

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `PFORGE_DATA_DIR` | `~/.local/share/pforge` | Root directory for all runtime data |

### Legacy (still supported, will be deprecated)

The following variables from the original server-oriented design still work but will be replaced by the `PFORGE_*` prefix in a future version:

`MODEL_NAME`, `ORCHESTRATOR_API_KEY`, `CORS_ALLOWED_ORIGINS`, `VLLM_PORT`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `DTYPE`, `QUANTIZATION`, `TRAIN_STOP_VLLM`, `WORKSPACE_DIR`

---

## Data directories

By default, all runtime data is stored under `~/.local/share/pforge/`:

| What | Path |
|------|------|
| Trained adapters | `<data_dir>/adapters/<job_id>/` |
| Training datasets | `<data_dir>/data/<job_id>/train.jsonl` |
| Logs | `<data_dir>/logs/` |
| Server state | `<data_dir>/server_status.json` |
| HuggingFace cache | Uses `HF_HOME` if set, otherwise HuggingFace default |

Override the root with `PFORGE_DATA_DIR`.

---

## API key

The API key is optional. If `PFORGE_API_KEY` is set, all endpoints except `GET /health` require the `X-API-Key: <key>` header.

Generate a strong key:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## Rate limiting

Heavy endpoints (`/train`, `/logit_lens`, `/restart_vllm`) are rate-limited to prevent accidental concurrent runs.

| Variable | Default | Applies to |
|----------|---------|------------|
| `PFORGE_RATE_LIMIT_HEAVY` | `5` req/min | train, logit_lens, restart_vllm |
| `PFORGE_RATE_LIMIT_OPS` | `10` req/min | load_lora |
