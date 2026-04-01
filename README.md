# pforge

A CLI tool for developers to inspect, steer, and fine-tune open models on their own GPU.

Run it on any machine with a CUDA GPU. Point it at a model. Then probe it, constrain it, compare it against fine-tuned versions, and watch its reasoning form layer by layer.

---

## What it does

| Command | What it gives you |
|---------|-------------------|
| `pforge chat` | Streaming chat with visible chain-of-thought |
| `pforge think` | Same prompt at low / medium / high compute budgets — see how thinking time changes quality |
| `pforge compare` | Two models answer the same prompt in parallel — base vs fine-tuned, side by side |
| `pforge logit-lens` | Peek inside the model layer by layer as the answer crystallises |
| `pforge debate` | Two model instances argue opposing sides over N rounds |
| `pforge constrain` | Answer a prompt under explicit reasoning rules you define |
| `pforge evolve` | Iterative refinement — give feedback, model improves its answer |
| `pforge train` | Train a LoRA adapter from a small dataset and hot-load it |

---

## Requirements

- Python 3.10+
- CUDA GPU (8 GB+ VRAM recommended; 24 GB for training alongside inference)
- [vLLM](https://github.com/vllm-project/vllm) (installed by the setup script or manually)

Tested on Linux with CUDA 12.1. macOS (CPU/MPS) is not currently supported.

---

## Installation

```bash
pip install pforge
```

Or from source:

```bash
git clone https://github.com/prateekvjoshi/pforge
cd pforge
pip install -e .
```

---

## Quick start

```bash
# Initialize — sets up local directories and config
pforge init

# Start the local model server (downloads model on first run, ~2-5 min)
pforge serve

# In a second terminal — chat with the model
pforge chat

# See how thinking budget affects response quality
pforge think "Is it better to move fast or get it right?" --budget high

# Peek inside the model layer by layer
pforge logit-lens "The capital of France is"
```

---

## Fine-tuning

Train a LoRA adapter from a small dataset:

```bash
pforge train dataset.jsonl --run-name my-style --steps 50
```

Then compare it against the base model:

```bash
pforge compare --model-b my-style --prompt "Describe a coffee shop"
```

See [docs/training.md](docs/training.md) for dataset formats and hyperparameter options.

---

## Configuration

All settings have local defaults. Override via environment variables or a config file.

```bash
# Set model
export PR_MODEL="Qwen/Qwen3-1.7B"

# Set data directory (default: ~/.local/share/pforge)
export PR_DATA_DIR="~/my-models"
```

See [docs/configuration.md](docs/configuration.md) for the full reference.

---

## API server

`pforge serve` starts a local REST API on `http://localhost:8000`. All the CLI commands talk to this server. You can also call it directly from scripts, custom UIs, or other tools:

```bash
curl -X POST http://localhost:8000/think \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Explain recursion", "budget": "medium"}'
```

See [docs/api.md](docs/api.md) for the full API reference.

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the pieces fit together
- [docs/cli.md](docs/cli.md) — CLI reference
- [docs/configuration.md](docs/configuration.md) — all configuration options
- [docs/training.md](docs/training.md) — fine-tuning guide
- [docs/troubleshooting.md](docs/troubleshooting.md) — common issues

---

## License

MIT — see [LICENSE](LICENSE).
