"""
cli.py — pforge command-line interface.

Entry point: pforge (wired via pyproject.toml console_scripts)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _server_url(args) -> str:
    return (
        getattr(args, 'server', None)
        or os.environ.get('PFORGE_SERVER')
        or os.environ.get('POD_URL')          # backward-compat with examples/
        or 'http://localhost:8000'
    )


def _api_key(args) -> str:
    return (
        getattr(args, 'api_key', None)
        or os.environ.get('PFORGE_API_KEY')
        or os.environ.get('PR_API_KEY')
        or os.environ.get('API_KEY')           # backward-compat with examples/
        or os.environ.get('ORCHESTRATOR_API_KEY')
        or ''
    )


def _headers(args) -> dict:
    h = {'Content-Type': 'application/json'}
    key = _api_key(args)
    if key:
        h['X-API-Key'] = key
    return h


def _die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _get(args, path: str) -> dict:
    import httpx
    url = f"{_server_url(args)}{path}"
    try:
        r = httpx.get(url, headers=_headers(args), timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        _die(
            f"Cannot connect to server at {_server_url(args)}.\n"
            "  Is it running?  Try: pforge serve"
        )
    except httpx.HTTPStatusError as e:
        _die(f"Server returned {e.response.status_code}: {e.response.text[:200]}")


def _post(args, path: str, body: dict, timeout=60.0) -> dict:
    import httpx
    url = f"{_server_url(args)}{path}"
    try:
        r = httpx.post(url, headers=_headers(args), json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        _die(
            f"Cannot connect to server at {_server_url(args)}.\n"
            "  Is it running?  Try: pforge serve"
        )
    except httpx.HTTPStatusError as e:
        _die(f"Server returned {e.response.status_code}: {e.response.text[:200]}")


def _stream_sse(args, path: str, body: dict):
    """Yield parsed JSON objects from an SSE endpoint."""
    import httpx
    url = f"{_server_url(args)}{path}"
    try:
        with httpx.Client(timeout=None) as client:
            with client.stream('POST', url, headers=_headers(args), json=body) as r:
                if r.status_code != 200:
                    _die(f"Server returned {r.status_code}: {r.read().decode()[:200]}")
                for line in r.iter_lines():
                    line = line.strip()
                    if not line.startswith('data:'):
                        continue
                    chunk_str = line[5:].strip()
                    if chunk_str == '[DONE]':
                        return
                    try:
                        yield json.loads(chunk_str)
                    except json.JSONDecodeError:
                        pass
    except httpx.ConnectError:
        _die(
            f"Cannot connect to server at {_server_url(args)}.\n"
            "  Is it running?  Try: pforge serve"
        )


def _require_vllm_ready(args) -> None:
    """Exit with a friendly message if vLLM is not ready."""
    data = _get(args, '/health')
    if not data.get('vllm_up'):
        _die(
            f"vLLM is not ready (status: {data.get('status', '?')}).\n"
            "  Check: pforge status"
        )


# ═════════════════════════════════════════════════════════════════════════════
# pforge init — helpers
# ═════════════════════════════════════════════════════════════════════════════

def _detect_cuda_version() -> Optional[str]:
    """Return CUDA version as 'major.minor' string, or None if not found."""
    import re
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi'], stderr=subprocess.DEVNULL, text=True
        )
        m = re.search(r'CUDA Version:\s*(\d+\.\d+)', out)
        if m:
            return m.group(1)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _cuda_tag(cuda_version: Optional[str]) -> str:
    """Map a CUDA version string to the closest supported PyTorch wheel tag."""
    if not cuda_version:
        return 'cu121'   # safe default
    major, minor = int(cuda_version.split('.')[0]), int(cuda_version.split('.')[1])
    if major >= 12:
        if minor >= 4:
            return 'cu124'
        return 'cu121'
    if major == 11:
        return 'cu118'
    return 'cu121'


def _install_gpu_stack(cuda_version: Optional[str]) -> None:
    """Install PyTorch (CUDA wheel), vLLM, and the training stack."""
    import subprocess
    tag = _cuda_tag(cuda_version)
    torch_index = f"https://download.pytorch.org/whl/{tag}"
    pip = [sys.executable, '-m', 'pip', 'install', '--quiet']

    print(f"\n[1/3] Installing PyTorch ({tag}) from {torch_index} …")
    subprocess.check_call(
        pip + ['torch', 'torchvision', 'torchaudio', '--index-url', torch_index]
    )
    print("      Done.")

    print("\n[2/3] Installing vLLM …")
    subprocess.check_call(pip + ['vllm'])
    print("      Done.")

    print("\n[3/3] Installing training stack (transformers / peft / trl / …) …")
    subprocess.check_call(pip + [
        'transformers>=4.45.0',
        'tokenizers>=0.19.0',
        'accelerate>=0.30.0',
        'peft>=0.11.0',
        'trl>=0.8.6',
        'datasets>=2.19.0',
        'bitsandbytes>=0.43.0',
        'einops>=0.7.0',
        'scipy>=1.11.0',
    ])
    print("      Done.")


# ═════════════════════════════════════════════════════════════════════════════
# pforge init
# ═════════════════════════════════════════════════════════════════════════════

def cmd_init(args):
    """Create data directories, validate environment, optionally install GPU deps."""
    from pforge.paths import resolve_data_dir

    # ── Directories ───────────────────────────────────────────────────────────
    data_dir = resolve_data_dir()
    dirs = [
        data_dir / 'logs',
        data_dir / 'adapters',
        data_dir / 'data',
        data_dir / 'status',
        data_dir / 'hf_cache' / 'hub',
    ]
    print(f"Data directory: {data_dir}")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ok  {d.name}/")

    # ── GPU / CUDA ────────────────────────────────────────────────────────────
    print("\nGPU:")
    cuda_version = _detect_cuda_version()
    if cuda_version:
        print(f"  CUDA {cuda_version} detected  (will use {_cuda_tag(cuda_version)} torch wheel)")
    else:
        print("  No CUDA GPU detected — vLLM requires an NVIDIA GPU with CUDA.")

    # ── Python packages ───────────────────────────────────────────────────────
    print("\nPackages:")
    _GPU_PKGS      = ['torch', 'vllm']
    _TRAINING_PKGS = ['transformers', 'accelerate', 'peft', 'trl', 'datasets', 'bitsandbytes']
    missing = []
    for pkg in _GPU_PKGS + _TRAINING_PKGS:
        try:
            mod = __import__(pkg)
            print(f"  ok       {pkg} {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"  missing  {pkg}")
            missing.append(pkg)

    # GPU details (only if torch is available)
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\nGPU memory: {name} — {mem_gb:.1f} GB total")
    except ImportError:
        pass

    # ── Install or advise ─────────────────────────────────────────────────────
    print()
    if not missing:
        print("All good. Run:  pforge serve")
        return

    if getattr(args, 'install_gpu_deps', False):
        if not cuda_version:
            _die("No CUDA GPU detected. Cannot install GPU packages without CUDA.")
        try:
            _install_gpu_stack(cuda_version)
        except Exception as exc:
            _die(f"Installation failed: {exc}")
        print("\nGPU stack installed. Run:  pforge serve")
    else:
        print(f"Missing packages: {', '.join(missing)}")
        print()
        print("To install automatically:")
        print("  pforge init --install-gpu-deps")
        print()
        print("Or install manually (order matters):")
        tag = _cuda_tag(cuda_version)
        print(f"  pip install torch torchvision torchaudio \\")
        print(f"      --index-url https://download.pytorch.org/whl/{tag}")
        print(f"  pip install vllm")
        print(f"  pip install transformers accelerate peft trl datasets bitsandbytes")


# ═════════════════════════════════════════════════════════════════════════════
# pforge serve
# ═════════════════════════════════════════════════════════════════════════════

def cmd_serve(args):
    """Start the API server and vLLM subprocess."""
    # Apply CLI overrides before config is read
    if args.model:
        os.environ['PR_MODEL'] = args.model
    if args.port:
        os.environ['PR_PORT'] = str(args.port)
    if args.host:
        os.environ['PR_HOST'] = args.host
    if args.gpu_memory_utilization is not None:
        os.environ['PR_GPU_MEMORY_UTILIZATION'] = str(args.gpu_memory_utilization)
    if args.set_api_key:
        os.environ['PR_API_KEY'] = args.set_api_key

    # Import config after env vars are set
    import importlib
    import pforge.config as cfg
    importlib.reload(cfg)   # pick up any env overrides set above

    import uvicorn

    print(f"Starting pforge")
    print(f"  Server : http://{cfg.ORCHESTRATOR_HOST}:{cfg.ORCHESTRATOR_PORT}")
    print(f"  Model  : {cfg.MODEL_NAME}")
    print(f"  Data   : {cfg.LOGS_DIR.parent}")
    if cfg.ORCHESTRATOR_API_KEY:
        print(f"  Auth   : enabled (X-API-Key required)")
    else:
        print(f"  Auth   : disabled — set PR_API_KEY before exposing externally")
    print()

    uvicorn.run(
        "pforge.server:app",
        host=cfg.ORCHESTRATOR_HOST,
        port=cfg.ORCHESTRATOR_PORT,
        reload=False,
        log_level='warning',
    )


# ═════════════════════════════════════════════════════════════════════════════
# pforge status
# ═════════════════════════════════════════════════════════════════════════════

def cmd_status(args):
    """Show server, vLLM, and training status."""
    import httpx

    base = _server_url(args)
    try:
        h = httpx.get(f"{base}/health", timeout=5.0).json()
    except httpx.ConnectError:
        _die(
            f"Cannot reach server at {base}.\n"
            "  Is it running?  Try: pforge serve"
        )

    vllm_ok = h.get('vllm_up', False)
    print(f"Server  : {base}")
    print(f"Status  : {h.get('status', '?')}")
    print(f"vLLM    : {'running' if vllm_ok else 'not ready'}")

    if not vllm_ok:
        return

    data = _get(args, '/status')
    print(f"Model   : {data.get('base_model', '?')}")
    adapter = data.get('active_lora_name')
    if adapter:
        print(f"Adapter : {adapter}")
    uptime = data.get('uptime_seconds', 0)
    m, s = divmod(int(uptime), 60)
    h2, m2 = divmod(m, 60)
    print(f"Uptime  : {h2:02d}:{m2:02d}:{s:02d}")

    t = data.get('training', {})
    state = t.get('state', 'idle')
    if state not in ('idle', 'IDLE'):
        print(f"\nTraining")
        print(f"  State   : {state}")
        if t.get('run_name'):
            print(f"  Run     : {t['run_name']}")
        if t.get('progress_pct') is not None:
            bar_len = 30
            filled = int(t['progress_pct'] / 100 * bar_len)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"  Progress: [{bar}] {t['progress_pct']:.1f}%")
        if t.get('loss') is not None:
            print(f"  Loss    : {t['loss']:.4f}")
        if t.get('error'):
            print(f"  Error   : {t['error']}")


# ═════════════════════════════════════════════════════════════════════════════
# pforge think
# ═════════════════════════════════════════════════════════════════════════════

def cmd_think(args):
    """Send a prompt with controlled thinking budget."""
    _require_vllm_ready(args)

    budgets = ['low', 'medium', 'high'] if args.budget is None else [args.budget]
    prompt = args.prompt

    for budget in budgets:
        if len(budgets) > 1:
            print(f"\n{'━' * 60}")
            print(f"  Budget: {budget.upper()}")
            print(f"{'━' * 60}\n")

        body = {'prompt': prompt, 'budget': budget}
        if args.model:
            body['model'] = args.model

        for chunk in _stream_sse(args, '/think', body):
            try:
                delta = chunk['choices'][0]['delta']
                content = delta.get('content') or ''
                if content:
                    print(content, end='', flush=True)
            except (KeyError, IndexError):
                pass
        print()


# ═════════════════════════════════════════════════════════════════════════════
# pforge compare
# ═════════════════════════════════════════════════════════════════════════════

def cmd_compare(args):
    """Compare two models on the same prompt."""
    _require_vllm_ready(args)

    prompt = args.prompt
    if not prompt:
        try:
            prompt = input('Prompt: ').strip()
        except (KeyboardInterrupt, EOFError):
            return

    body = {'prompt': prompt, 'thinking': args.thinking}
    if args.model_a:
        body['model_a'] = args.model_a
    if args.model_b:
        body['model_b'] = args.model_b

    responses: dict = {'a': [], 'b': []}
    for chunk in _stream_sse(args, '/compare', body):
        side = chunk.get('side')
        if side in responses:
            content = chunk.get('content', '')
            if content:
                responses[side].append(content)

    import pforge.config as cfg
    label_a = args.model_a or cfg.MODEL_NAME
    label_b = args.model_b or cfg.MODEL_NAME

    print(f"\n{'━' * 60}")
    print(f"  A: {label_a}")
    print(f"{'━' * 60}")
    print(''.join(responses['a']))

    print(f"\n{'━' * 60}")
    print(f"  B: {label_b}")
    print(f"{'━' * 60}")
    print(''.join(responses['b']))
    print()


# ═════════════════════════════════════════════════════════════════════════════
# pforge chat
# ═════════════════════════════════════════════════════════════════════════════

def cmd_chat(args):
    """Interactive streaming chat session."""
    _require_vllm_ready(args)

    import httpx
    import pforge.config as cfg

    model = args.model or cfg.MODEL_NAME
    print(f"Chat with {model}  (Ctrl+C or Ctrl+D to exit)\n")

    history = []
    while True:
        try:
            prompt = input('You: ').strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not prompt:
            continue

        history.append({'role': 'user', 'content': prompt})

        body = json.dumps({
            'model': model,
            'messages': history,
            'stream': True,
            'chat_template_kwargs': {'enable_thinking': not args.no_think},
        }).encode()

        print('Assistant: ', end='', flush=True)
        full_reply: list = []
        in_thinking = False

        url = f"{_server_url(args)}/v1/chat/completions"
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream('POST', url, headers=_headers(args), content=body) as r:
                    for line in r.iter_lines():
                        line = line.strip()
                        if not line.startswith('data:'):
                            continue
                        chunk_str = line[5:].strip()
                        if chunk_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(chunk_str)
                            delta = chunk['choices'][0]['delta']
                            thinking = delta.get('reasoning_content') or ''
                            content = delta.get('content') or ''
                            if thinking:
                                if not in_thinking:
                                    print('\n<think>', flush=True)
                                    in_thinking = True
                                print(thinking, end='', flush=True)
                            if content:
                                if in_thinking:
                                    print('\n</think>\n', flush=True)
                                    in_thinking = False
                                print(content, end='', flush=True)
                                full_reply.append(content)
                        except (KeyError, IndexError, json.JSONDecodeError):
                            pass
        except httpx.ConnectError:
            print("\n[connection lost]", file=sys.stderr)
            break

        print('\n')
        history.append({'role': 'assistant', 'content': ''.join(full_reply)})


# ═════════════════════════════════════════════════════════════════════════════
# pforge debate
# ═════════════════════════════════════════════════════════════════════════════

def cmd_debate(args):
    """Two model instances argue opposing sides over multiple rounds."""
    _require_vllm_ready(args)

    body = {'topic': args.topic, 'rounds': args.rounds}
    if args.model:
        body['model'] = args.model

    print(f"\nDebate: {args.topic}\n")
    current_key = None

    for chunk in _stream_sse(args, '/debate', body):
        if chunk.get('error'):
            print(f"\n[Error: {chunk['error']}]", file=sys.stderr)
            break

        side = chunk.get('side', '')
        round_num = chunk.get('round', 0)
        key = (round_num, side)

        if key != current_key:
            if current_key is not None:
                print('\n')
            label = 'FOR' if side == 'for' else 'AGAINST'
            print(f"[Round {round_num} — {label}]\n")
            current_key = key

        content = chunk.get('content', '')
        if content:
            print(content, end='', flush=True)

    print('\n')


# ═════════════════════════════════════════════════════════════════════════════
# pforge constrain
# ═════════════════════════════════════════════════════════════════════════════

_CONSTRAINT_PRESETS = {
    'simple':     [
        "Explain like I'm 5",
        "Maximum 3 sentences",
        "No technical jargon",
    ],
    'creative':   [
        "Use analogies only — no direct statements",
        "Each sentence must start with a different letter, in alphabetical order",
        "End with a surprising twist",
    ],
    'structured': [
        "Break your answer into exactly 3 steps",
        "Start each step with an action verb",
        "End with a one-sentence summary",
    ],
}


def cmd_constrain(args):
    """Answer a prompt while obeying explicit reasoning rules."""
    _require_vllm_ready(args)

    constraints = list(args.constraints or [])
    if args.preset:
        constraints = _CONSTRAINT_PRESETS.get(args.preset, []) + constraints
    if not constraints:
        _die(
            "Provide at least one constraint.\n"
            "  --constraints 'rule 1' 'rule 2'\n"
            "  --preset simple|creative|structured"
        )

    body = {
        'prompt': args.prompt,
        'constraints': constraints,
        'thinking': args.thinking,
    }
    if args.model:
        body['model'] = args.model

    print(f"Rules:")
    for i, c in enumerate(constraints, 1):
        print(f"  {i}. {c}")
    print()

    for chunk in _stream_sse(args, '/constrain', body):
        try:
            delta = chunk['choices'][0]['delta']
            content = delta.get('content') or ''
            if content:
                print(content, end='', flush=True)
        except (KeyError, IndexError):
            pass
    print()


# ═════════════════════════════════════════════════════════════════════════════
# pforge evolve
# ═════════════════════════════════════════════════════════════════════════════

def cmd_evolve(args):
    """Iterative refinement — give feedback, model improves its answer."""
    _require_vllm_ready(args)

    prompt = args.prompt
    if not prompt:
        try:
            prompt = input('Prompt: ').strip()
        except (KeyboardInterrupt, EOFError):
            return

    previous_response: Optional[str] = None
    feedback: Optional[str] = None

    while True:
        body = {'prompt': prompt, 'thinking': True}
        if args.model:
            body['model'] = args.model
        if previous_response:
            body['previous_response'] = previous_response
        if feedback:
            body['feedback'] = feedback

        label = 'Refinement' if previous_response else 'Answer'
        print(f"\n[{label}]\n")

        parts: list = []
        for chunk in _stream_sse(args, '/evolve', body):
            try:
                delta = chunk['choices'][0]['delta']
                content = delta.get('content') or ''
                if content:
                    print(content, end='', flush=True)
                    parts.append(content)
            except (KeyError, IndexError):
                pass
        print('\n')

        previous_response = ''.join(parts)

        try:
            user_input = input("Feedback ('done' to finish, 'reset' for new prompt): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if user_input.lower() == 'done':
            break
        elif user_input.lower() == 'reset':
            previous_response = None
            feedback = None
            try:
                prompt = input('New prompt: ').strip()
            except (KeyboardInterrupt, EOFError):
                break
        else:
            feedback = user_input


# ═════════════════════════════════════════════════════════════════════════════
# pforge train
# ═════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    """Train a LoRA adapter from a JSONL dataset file."""
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        _die(f"Dataset file not found: {dataset_path}")

    dataset = []
    with open(dataset_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as e:
                _die(f"Invalid JSON on line {i} of {dataset_path}: {e}")

    if not dataset:
        _die("Dataset is empty.")

    print(f"Dataset : {len(dataset)} examples  ({dataset_path})")
    print(f"Run name: {args.run_name}")

    body: dict = {'run_name': args.run_name, 'dataset': dataset}
    hp: dict = {}
    if args.steps is not None:     hp['max_steps'] = args.steps
    if args.epochs is not None:    hp['epochs'] = args.epochs
    if args.lora_rank is not None: hp['lora_rank'] = args.lora_rank
    if args.lr is not None:        hp['learning_rate'] = args.lr
    if hp:
        body['hyperparams'] = hp
    if args.system_prompt:
        body['system_prompt'] = args.system_prompt

    result = _post(args, '/train', body)
    job_id = result.get('job_id', '?')
    print(f"Job     : {job_id}")
    print(f"Output  : {result.get('adapter_output_path', '?')}")
    print(f"\nPolling for progress  (Ctrl+C to stop watching — training continues in background)\n")

    last_pct = -1.0
    try:
        while True:
            time.sleep(5)
            data = _get(args, '/status')
            t = data.get('training', {})
            state = t.get('state', '?')
            pct = t.get('progress_pct')
            loss = t.get('loss')

            if pct is not None and pct != last_pct:
                bar_len = 30
                filled = int(pct / 100 * bar_len)
                bar = '█' * filled + '░' * (bar_len - filled)
                loss_str = f"  loss={loss:.4f}" if loss else ''
                print(f"  [{bar}] {pct:5.1f}%{loss_str}")
                last_pct = pct

            if state in ('succeeded', 'SUCCEEDED'):
                adapter_name = args.run_name.replace(' ', '_').replace('/', '_')
                print(f"\nTraining complete!")
                print(f"  Adapter : {t.get('adapter_path', '?')}")
                print(f"  Compare : pforge compare --model-b {adapter_name} 'Your prompt'")
                break
            elif state in ('failed', 'FAILED'):
                _die(f"Training failed: {t.get('error', 'unknown error')}")
    except KeyboardInterrupt:
        print(f"\nStopped watching. Training continues in background.")
        print(f"  Check progress: pforge status")


# ═════════════════════════════════════════════════════════════════════════════
# pforge logit-lens
# ═════════════════════════════════════════════════════════════════════════════

def cmd_logit_lens(args):
    """Peek inside the model layer by layer as the answer forms."""
    body = {'prompt': args.prompt, 'top_k': args.top_k}
    if args.lora_path:
        body['lora_path'] = args.lora_path

    print("Running logit lens (stops vLLM briefly to free VRAM — ~1-2 min)…\n")

    result = _post(args, '/logit_lens', body, timeout=None)

    layers = result.get('layers', [])
    final_answer = result.get('final_answer', '?')
    first_layer = result.get('answer_first_appears_at_layer')
    n_layers = result.get('num_layers', 0)

    print(f"Prompt : {result.get('prompt', '')!r}")
    print(f"Answer : {final_answer!r}  (first at layer {first_layer} of {n_layers})\n")

    for entry in layers:
        label = entry['label']
        preds = entry['top_predictions'][:args.top_k]
        if args.compact:
            top = preds[0] if preds else {}
            print(f"  {label:15s}  {top.get('token', ''):12s}  {top.get('prob', 0):.3f}")
        else:
            parts = [f"{p['token']!r}:{p['prob']:.3f}" for p in preds]
            print(f"  {label:15s}  {'  '.join(parts)}")


# ═════════════════════════════════════════════════════════════════════════════
# pforge adapters
# ═════════════════════════════════════════════════════════════════════════════

def cmd_adapters_list(args):
    """List the active adapter and all adapters available on disk."""
    data = _get(args, '/status')
    active = data.get('active_lora_name')
    if active:
        print(f"Active adapter: {active}")
        print(f"  Path: {data.get('active_lora_adapter', '?')}")
    else:
        print("No adapter loaded (using base model).")

    import pforge.config as cfg
    adapters_dir = cfg.ADAPTERS_DIR
    if not adapters_dir.exists():
        return

    dirs = sorted(d for d in adapters_dir.iterdir() if d.is_dir())
    if not dirs:
        return

    print(f"\nOn disk ({adapters_dir}):")
    for d in dirs:
        meta_file = d / 'training_meta.json'
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                steps = meta.get('effective_steps', '?')
                base = meta.get('base_model', '?')
                marker = ' ← active' if d.name == active else ''
                print(f"  {d.name}  ({steps} steps, base={base}){marker}")
                continue
            except Exception:
                pass
        print(f"  {d.name}")


def cmd_adapters_load(args):
    """Load a LoRA adapter from disk into the running vLLM server."""
    body = {'lora_name': args.name, 'lora_path': args.path}
    result = _post(args, '/load_lora', body)
    status = 'ok' if result.get('success') else 'failed'
    print(f"[{status}] {result.get('message', '?')}")
    if result.get('method'):
        print(f"Method: {result['method']}")


# ═════════════════════════════════════════════════════════════════════════════
# Parser
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='pforge',
        description='pforge — local LLM inference and fine-tuning',
    )
    parser.add_argument(
        '--server', default=None, metavar='URL',
        help='API server URL (default: http://localhost:8000 or $PFORGE_SERVER)',
    )
    parser.add_argument(
        '--api-key', dest='api_key', default=None, metavar='KEY',
        help='API key ($PFORGE_API_KEY / $PR_API_KEY)',
    )
    parser.add_argument('--version', action='version', version='pforge 0.1.0')

    sub = parser.add_subparsers(dest='command', metavar='<command>')

    # ── init ──────────────────────────────────────────────────────────────────
    p = sub.add_parser('init', help='Set up local directories and validate dependencies')
    p.add_argument('--install-gpu-deps', dest='install_gpu_deps', action='store_true',
                   default=False,
                   help='Install PyTorch (CUDA wheel), vLLM, and training stack')
    p.set_defaults(func=cmd_init)

    # ── serve ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('serve', help='Start the API server and vLLM subprocess')
    p.add_argument('--model', default=None,
                   help='HuggingFace model name or local path')
    p.add_argument('--port', type=int, default=None,
                   help='Port to listen on (default: 8000)')
    p.add_argument('--host', default=None,
                   help='Bind address (default: 127.0.0.1; use 0.0.0.0 for network)')
    p.add_argument('--gpu-memory-utilization', dest='gpu_memory_utilization',
                   type=float, default=None, metavar='FLOAT',
                   help='Fraction of GPU VRAM to use (default: 0.80)')
    p.add_argument('--set-api-key', dest='set_api_key', default=None, metavar='KEY',
                   help='Enable API key auth for this session')
    p.set_defaults(func=cmd_serve)

    # ── status ────────────────────────────────────────────────────────────────
    p = sub.add_parser('status', help='Show server, vLLM, and training status')
    p.set_defaults(func=cmd_status)

    # ── think ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('think', help='Send a prompt with a controlled thinking budget')
    p.add_argument('prompt', help='Prompt to send')
    p.add_argument('--budget', default=None,
                   help='low | medium | high | <integer tokens>  '
                        '(omit to run all three budgets)')
    p.add_argument('--model', default=None)
    p.set_defaults(func=cmd_think)

    # ── compare ───────────────────────────────────────────────────────────────
    p = sub.add_parser('compare', help='Compare two models on the same prompt')
    p.add_argument('prompt', nargs='?', default=None,
                   help='Prompt (interactive if omitted)')
    p.add_argument('--model-a', dest='model_a', default=None,
                   help='First model/adapter (default: base model)')
    p.add_argument('--model-b', dest='model_b', default=None,
                   help='Second model/adapter')
    p.add_argument('--thinking', action='store_true', default=False,
                   help='Enable chain-of-thought for both models')
    p.set_defaults(func=cmd_compare)

    # ── chat ──────────────────────────────────────────────────────────────────
    p = sub.add_parser('chat', help='Interactive streaming chat session')
    p.add_argument('--model', default=None)
    p.add_argument('--no-think', dest='no_think', action='store_true', default=False,
                   help='Disable chain-of-thought')
    p.set_defaults(func=cmd_chat)

    # ── debate ────────────────────────────────────────────────────────────────
    p = sub.add_parser('debate', help='Two model instances argue opposing sides')
    p.add_argument('topic', help='Topic or question to debate')
    p.add_argument('--rounds', type=int, default=2,
                   help='Number of back-and-forth rounds (default: 2)')
    p.add_argument('--model', default=None)
    p.set_defaults(func=cmd_debate)

    # ── constrain ─────────────────────────────────────────────────────────────
    p = sub.add_parser('constrain', help='Answer a prompt under explicit reasoning rules')
    p.add_argument('prompt', help='Prompt to answer')
    p.add_argument('--constraints', nargs='+', default=None, metavar='RULE',
                   help='Rules the model must follow')
    p.add_argument('--preset', default=None,
                   choices=['simple', 'creative', 'structured'],
                   help='Built-in constraint preset')
    p.add_argument('--thinking', action='store_true', default=False)
    p.add_argument('--model', default=None)
    p.set_defaults(func=cmd_constrain)

    # ── evolve ────────────────────────────────────────────────────────────────
    p = sub.add_parser('evolve', help='Iterative refinement — give feedback, model improves')
    p.add_argument('--prompt', default=None,
                   help='Initial prompt (interactive if omitted)')
    p.add_argument('--model', default=None)
    p.set_defaults(func=cmd_evolve)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('train', help='Train a LoRA adapter from a JSONL dataset')
    p.add_argument('dataset', help='Path to JSONL training file')
    p.add_argument('--run-name', required=True, dest='run_name',
                   help='Name for this adapter')
    p.add_argument('--steps',     type=int,   default=None, metavar='N',
                   help='Max training steps (default: 50)')
    p.add_argument('--epochs',    type=int,   default=None, metavar='N')
    p.add_argument('--lora-rank', type=int,   default=None, dest='lora_rank',
                   metavar='N')
    p.add_argument('--lr',        type=float, default=None,
                   help='Learning rate (default: 2e-4)')
    p.add_argument('--system-prompt', dest='system_prompt', default=None,
                   help='System prompt prepended to every training example')
    p.set_defaults(func=cmd_train)

    # ── logit-lens ────────────────────────────────────────────────────────────
    p = sub.add_parser('logit-lens',
                       help='Peek inside the model layer by layer as the answer forms')
    p.add_argument('prompt', help='Prompt to analyse')
    p.add_argument('--top-k', dest='top_k', type=int, default=3,
                   help='Top predictions per layer (default: 3)')
    p.add_argument('--compact', action='store_true', default=False,
                   help='Show only the top-1 token per layer')
    p.add_argument('--lora-path', dest='lora_path', default=None,
                   help='Path to a LoRA adapter to merge before analysis')
    p.set_defaults(func=cmd_logit_lens)

    # ── adapters ──────────────────────────────────────────────────────────────
    p_adapters = sub.add_parser('adapters', help='Adapter management')
    adapters_sub = p_adapters.add_subparsers(dest='adapters_cmd')

    p2 = adapters_sub.add_parser('list', help='List loaded and available adapters')
    p2.set_defaults(func=cmd_adapters_list)

    p2 = adapters_sub.add_parser('load', help='Load an adapter from disk into vLLM')
    p2.add_argument('path', help='Path to the adapter directory')
    p2.add_argument('--name', required=True,
                    help='Name to register the adapter as')
    p2.set_defaults(func=cmd_adapters_load)

    def _adapters_default(a):
        if not getattr(a, 'adapters_cmd', None):
            p_adapters.print_help()
        else:
            a.func(a)
    p_adapters.set_defaults(func=_adapters_default)

    return parser


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, 'func') or args.func is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == '__main__':
    main()
