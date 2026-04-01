# Troubleshooting

## vLLM won't start

Check the vLLM log:

```bash
tail -100 <data_dir>/logs/vllm.log
```

**Common causes:**

**Model not downloaded**
- Check that `HF_TOKEN` is set if the model is gated
- Check that `HF_HOME` points to a directory with enough space
- Try downloading manually: `huggingface-cli download Qwen/Qwen3-1.7B`

**Out of memory**
- Lower `PFORGE_GPU_MEMORY_UTILIZATION` (e.g. `0.70`)
- Try a smaller model
- Kill other GPU processes: `nvidia-smi` to see what's running

**Wrong reasoning parser**
- `--reasoning-parser qwen3` is only valid for Qwen3 models
- If using a different model family, remove this flag by setting `REASONING_PARSER=""` in config

**vLLM version too old**
- Some model architectures require recent vLLM
- Upgrade: `pip install --upgrade vllm`
- Or install nightly: `pip install vllm --pre`

---

## Training fails with CUDA OOM

Check the training log:

```bash
tail -200 <data_dir>/logs/training_<job_id>.log
```

**Mitigations (in order of impact):**

1. Ensure `TRAIN_STOP_SERVER=true` (frees the most VRAM)
2. Reduce `max_seq_length`: `1024 → 512`
3. Reduce `lora_rank`: `8 → 4`
4. `batch_size` is already 1 — increase `grad_accumulation` instead of batch size

---

## Server starts but model loads slowly

First load always takes time — the model is being downloaded and cached. Subsequent starts use the local cache and are fast.

Watch progress:
```bash
tail -f <data_dir>/logs/vllm.log
```

Look for `"Avg prompt throughput"` — that means vLLM is ready.

---

## API returns 503

vLLM isn't ready yet. Check `/health`:

```bash
curl http://localhost:8000/health
```

`"vllm_up": false` means vLLM is still loading or has crashed. Check the vLLM log.

---

## API returns 401

The server has `PFORGE_API_KEY` set. Pass the key in your request:

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/status
```

Or when using example scripts:

```bash
export API_KEY="your-key"
python3 examples/chat.py
```

---

## Adapter not found after training

If the training log shows success but the adapter isn't loading, check:

```bash
# See what adapters vLLM has loaded
curl -H "X-API-Key: $API_KEY" http://localhost:8000/v1/models
```

If the adapter isn't listed, try loading it manually:

```bash
curl -X POST http://localhost:8000/load_lora \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $API_KEY" \
  -d '{"lora_name": "my-adapter", "lora_path": "<data_dir>/adapters/<job_id>"}'
```

---

## Adapter trained on different model causes crash

If you see a crash on startup about adapter/model mismatch, the persisted adapter was trained on a different model than the one currently configured. The server will log a warning and skip loading it.

Fix: clear the persisted adapter:

```bash
# Edit server_status.json and set active_lora_name/path to null
# Or just restart fresh:
pforge init --reset
```

---

## Server won't restart cleanly

Kill all related processes and restart:

```bash
pkill -f "pforge"
pkill -f "vllm"
pforge serve
```

---

## Check which adapter is loaded

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/status | python3 -m json.tool
```

Look for `active_lora_name` in the response.

---

## Inspect a trained adapter

```bash
cat <data_dir>/adapters/<job_id>/training_meta.json
cat <data_dir>/adapters/<job_id>/adapter_config.json
```

---

## Getting help

Open an issue at https://github.com/prateekvjoshi/pforge/issues

Include:
- What command you ran
- What you expected
- The error message and relevant log lines
- GPU type and VRAM
- `python -c "import vllm; print(vllm.__version__)"`
- `nvidia-smi` output
