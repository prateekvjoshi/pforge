# CLI Reference

---

## Global options

```
pforge [--server URL] [--model MODEL] [--api-key KEY] <command>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--server` | `http://localhost:8000` | API server URL |
| `--model` | from config | Model name or adapter name |
| `--api-key` | from config/env | API key if server requires one |

---

## pforge init

Initialize the local environment.

```bash
pforge init [--install-gpu-deps]
```

- Creates local data directories
- Detects CUDA version and checks for torch, vLLM, and training packages
- Prints next steps

| Option | Description |
|--------|-------------|
| `--install-gpu-deps` | Install PyTorch (correct CUDA wheel), vLLM, and training stack automatically |

With `--install-gpu-deps`, `pforge init` detects your CUDA version, selects the right PyTorch wheel index (`cu118` / `cu121` / `cu124`), and installs the full GPU stack in the correct order. This is the recommended first step on a fresh GPU machine.

---

## pforge serve

Start the local API server and vLLM.

```bash
pforge serve [--model MODEL] [--port PORT] [--gpu-memory-utilization FLOAT]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen3-1.7B` | Model to load |
| `--port` | `8000` | Port to listen on |
| `--gpu-memory-utilization` | `0.80` | Fraction of GPU VRAM for vLLM |

First run downloads the model (~1–3 GB). Subsequent starts use the local cache.

---

## pforge chat

Interactive chat session.

```bash
pforge chat [--model MODEL] [--no-think]
```

Streams the model's chain-of-thought and answer. Use `Ctrl+C` to exit.

---

## pforge think

Send a prompt with a controlled thinking budget.

```bash
pforge think "Your prompt here" [--budget low|medium|high|N]
```

| Budget | Thinking tokens |
|--------|----------------|
| `low` | 256 |
| `medium` | 2048 |
| `high` | 8192 |
| integer | exact count |

Runs the same prompt at all three budget levels by default. Pass `--budget` to run just one.

---

## pforge compare

Compare two models on the same prompt.

```bash
pforge compare "Your prompt here" --model-b ADAPTER_NAME
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-a` | base model | First model |
| `--model-b` | required | Second model or adapter name |
| `--prompt` | interactive | Prompt to send |

Both responses are collected in parallel and displayed sequentially.

---

## pforge logit-lens

Peek inside the model layer by layer.

```bash
pforge logit-lens "The capital of France is" [--top-k 3] [--compact]
```

Shows the top-k token predictions at each transformer layer. Requires stopping the server briefly (~1–2 min) while the model loads for analysis.

| Option | Default | Description |
|--------|---------|-------------|
| `--top-k` | `3` | Top predictions per layer |
| `--compact` | false | Show only top-1 per layer |

---

## pforge debate

Two model instances argue opposing sides.

```bash
pforge debate "Your topic here" [--rounds 2]
```

---

## pforge constrain

Answer a prompt under explicit rules.

```bash
pforge constrain "Your prompt" --constraints "rule 1" "rule 2" [--preset simple|creative|structured]
```

Built-in presets:
- `simple` — explain like I'm 5, max 3 sentences, no jargon
- `creative` — analogies only, alphabetic sentence starts, surprising twist
- `structured` — 3 steps, verb-first, one-sentence summary

---

## pforge evolve

Iterative refinement via feedback.

```bash
pforge evolve [--prompt "Initial prompt"]
```

Round 1: model answers. Each subsequent round: give feedback, model improves. Type `done` to finish, `reset` for a new prompt.

---

## pforge train

Train a LoRA adapter.

```bash
pforge train DATASET_FILE --run-name NAME [--steps 50] [--lora-rank 8]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--run-name` | required | Name for this adapter |
| `--steps` | `50` | Training steps |
| `--lora-rank` | `8` | LoRA rank |
| `--learning-rate` | `0.0002` | Learning rate |

Dataset can be JSONL in chat or instruction format. See [training.md](training.md).

Stops the server during training (configurable). Restarts automatically when done.

---

## pforge status

Show server and training status.

```bash
pforge status
```

---

## pforge adapters list

List loaded and available adapters.

```bash
pforge adapters list
```

---

## pforge adapters load

Load an adapter from disk.

```bash
pforge adapters load PATH --name NAME
```

---

## Using the example scripts directly

The `pforge` CLI is the recommended interface. You can also call the example scripts directly:

```bash
# Set server URL and API key
export PFORGE_SERVER="http://localhost:8000"
export PFORGE_API_KEY="your-key"

python3 examples/chat.py
python3 examples/think.py --prompt "Your prompt"
python3 examples/logit_lens.py --prompt "The capital of France is"
python3 examples/flash_tune.py
```
