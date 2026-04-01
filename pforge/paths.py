"""
paths.py — platform-appropriate local data directory resolution.

Priority for data root:
  1. PFORGE_DATA_DIR env var
  2. Legacy WORKSPACE_DIR env var (backward compat)
  3. Platform default:
       Linux/other  — $XDG_DATA_HOME/pforge
                      (falls back to ~/.local/share/pforge)
       macOS        — ~/Library/Application Support/pforge
"""
import os
import sys
from pathlib import Path


APP_NAME = "pforge"


def default_data_dir() -> Path:
    """Return the platform-appropriate default data directory."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / APP_NAME


def resolve_data_dir() -> Path:
    """Return the active data directory, respecting env var overrides."""
    explicit = os.environ.get("PFORGE_DATA_DIR") or os.environ.get("WORKSPACE_DIR")
    return Path(explicit) if explicit else default_data_dir()
