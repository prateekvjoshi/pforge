# Contributing to Programmable Reasoning

Thanks for your interest in contributing.

## Before you start

- Check the [open issues](../../issues) to avoid duplicating work.
- For significant changes, open an issue first to discuss the approach.

## Setup

You need a GPU to run the full stack (vLLM requires CUDA). For cloud GPU options see [README.md](README.md).

For development without a GPU (editing server logic, examples, docs):

```bash
git clone https://github.com/prateekvjoshi/pforge
cd pforge
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Making changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Test manually against a live GPU server if your change touches server logic
4. Submit a pull request with a clear description of what and why

## What we welcome

- Bug fixes
- New inference modes (new endpoints in `server.py` + example script)
- Improved error messages and UX in example scripts
- Documentation improvements
- Support for additional model families

## What to avoid

- Breaking changes to existing API contracts without discussion
- Adding dependencies that require non-standard GPU environments
- Hardcoding paths, model names, or credentials

## Code style

- Follow the existing style — no formatter is enforced yet
- Keep functions focused and avoid adding abstraction layers for single uses
- Add comments only where the logic isn't self-evident

## Reporting bugs

Open an issue with:
- What you ran (exact command)
- What you expected
- What happened (paste the error and relevant log lines — check `pforge status` or the logs directory)
- Your GPU type and vLLM version (`python -c "import vllm; print(vllm.__version__)"`)
