"""
state.py — thread-safe in-memory application state with JSON persistence.

Design notes:
- A single AppState instance is created at module import time.
- All mutations go through update() / update_training() which hold the lock
  and then atomically write the snapshot to disk via a tmp-then-rename.
- On restart, load_persisted() recovers the last known adapter path and
  marks any in-flight training job as FAILED (we can't resume it safely).
"""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pforge.config as cfg
from pforge.models import TrainingJobStatus, TrainingState, VLLMState

logger = logging.getLogger(__name__)


class AppState:
    """
    Thread-safe application state container.

    Fields:
        vllm_state        — lifecycle state of the vLLM subprocess
        vllm_pid          — PID of the running vLLM process (or None)
        active_lora_name  — logical name of the currently loaded LoRA adapter
        active_lora_path  — filesystem path of the currently loaded adapter
        training          — current training job status (nested model)
        started_at        — when this server process started
    """

    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self.vllm_state     = VLLMState.STOPPED
        self.vllm_pid: Optional[int]  = None
        self.active_lora_name: Optional[str] = None
        self.active_lora_path: Optional[str] = None
        self.training       = TrainingJobStatus()
        self.started_at     = datetime.now(timezone.utc)

    # ── Public mutators ───────────────────────────────────────────────────────

    def update(self, **kwargs) -> None:
        """Thread-safe update of top-level fields, then persist."""
        with self._lock:
            for k, v in kwargs.items():
                if not hasattr(self, k) or k.startswith("_"):
                    raise AttributeError(f"AppState has no settable field '{k}'")
                setattr(self, k, v)
            self._persist()

    def update_training(self, **kwargs) -> None:
        """Thread-safe update of training sub-state fields, then persist."""
        with self._lock:
            for k, v in kwargs.items():
                if not hasattr(self.training, k):
                    raise AttributeError(f"TrainingJobStatus has no field '{k}'")
                setattr(self.training, k, v)
            self._persist()

    def reset_training(self) -> None:
        """Reset training state to idle. Call after a terminal state."""
        with self._lock:
            self.training = TrainingJobStatus()
            self._persist()

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a JSON-serialisable dict (acquires lock)."""
        with self._lock:
            return self._build_snapshot()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _build_snapshot(self) -> dict:
        """Build snapshot dict — must be called with lock held."""
        return {
            "vllm_state":        self.vllm_state,
            "vllm_pid":          self.vllm_pid,
            "active_lora_name":  self.active_lora_name,
            "active_lora_path":  self.active_lora_path,
            "training":          self.training.model_dump(mode="json"),
            "started_at":        self.started_at.isoformat(),
        }

    def _persist(self) -> None:
        """Atomic write to disk — must be called with lock held."""
        try:
            cfg.SERVER_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = cfg.SERVER_STATUS_FILE.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._build_snapshot(), indent=2, default=str)
            )
            tmp.replace(cfg.SERVER_STATUS_FILE)
        except Exception as exc:
            logger.warning(f"State persistence failed (non-fatal): {exc}")

    def load_persisted(self) -> None:
        """
        Attempt to restore state from the last persisted snapshot.

        Called once at server startup. Rules:
        - vLLM state is always reset to STOPPED (we don't know if the old
          process is still alive — the watchdog will re-probe).
        - If training was in-flight when we crashed, mark it FAILED.
        - Active LoRA adapter path is restored so vLLM can be restarted
          with the correct --lora-modules flag.
        """
        path = cfg.SERVER_STATUS_FILE
        if not path.exists():
            logger.info("No persisted state found; starting fresh.")
            return
        try:
            data = json.loads(path.read_text())

            # Restore adapter
            self.active_lora_name = data.get("active_lora_name")
            self.active_lora_path = data.get("active_lora_path")

            # Restore training — but fix up any in-flight states
            t = data.get("training", {})
            if t.get("state") in (TrainingState.RUNNING, TrainingState.QUEUED):
                t["state"] = TrainingState.FAILED
                t["error"] = "Server restarted while training was in progress."
                logger.warning(
                    "Previous training job was interrupted by a restart; "
                    f"marked as FAILED (job_id={t.get('job_id')})"
                )
            self.training = TrainingJobStatus(**t)

            logger.info(
                f"Restored persisted state: "
                f"active_lora={self.active_lora_name}, "
                f"training_state={self.training.state}"
            )
        except Exception as exc:
            logger.warning(f"Could not load persisted state: {exc}. Starting fresh.")
