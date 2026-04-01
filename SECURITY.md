# Security Policy

## Supported versions

Only the latest version on `main` is actively maintained.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Email the maintainer directly with:
- A description of the vulnerability
- Steps to reproduce it
- The potential impact
- Any suggested fix if you have one

You can expect an acknowledgement within 48 hours and a fix or mitigation plan within 7 days for critical issues.

## Known security boundaries

This service is designed to run locally or on a private GPU server. Keep the following in mind:

- **Do not expose port 8000 to the internet without additional firewall rules.** By default the server binds to `127.0.0.1` (loopback only). Set `PFORGE_HOST=0.0.0.0` only if you intend remote access and have network-level guards in place.
- **Set `PFORGE_API_KEY`** before sharing access with anyone. See [README.md](README.md) for setup instructions.
- **The API has no per-user isolation.** All authenticated callers share the same model and training state. It is designed for controlled access (small team or private use), not as a public multi-tenant service.
- **Training datasets are stored in plaintext** on disk. Do not train on sensitive data.
- **LoRA adapters are stored on disk** and persist across restarts. Anyone with filesystem access can read them.

## Security hardening checklist

- [ ] `PFORGE_API_KEY` is set to a strong random value (`python3 -c "import secrets; print(secrets.token_hex(32))"`)
- [ ] `PFORGE_CORS_ORIGINS` is set to your specific frontend origin (not left empty)
- [ ] Port 8000 is not exposed to untrusted networks
- [ ] Your secrets file has restricted permissions: `chmod 600 /path/to/.env.secrets`
