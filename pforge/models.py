"""
models.py — Pydantic v2 request/response schemas for all API endpoints.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── State enums ───────────────────────────────────────────────────────────────

class TrainingState(str, Enum):
    IDLE      = "idle"
    QUEUED    = "queued"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"


class VLLMState(str, Enum):
    STOPPED     = "stopped"
    STARTING    = "starting"
    RUNNING     = "running"
    RESTARTING  = "restarting"
    ERROR       = "error"


# ── Dataset / training request ────────────────────────────────────────────────

class Message(BaseModel):
    role:    str = Field(..., max_length=32)
    content: str = Field(..., max_length=32_768)


class HyperParams(BaseModel):
    """All fields are optional; server merges with config defaults."""
    epochs:             Optional[int]   = Field(default=None, ge=1,    le=10)
    max_steps:          Optional[int]   = Field(default=None, ge=1,    le=1000)
    batch_size:         Optional[int]   = Field(default=None, ge=1,    le=8)
    grad_accumulation:  Optional[int]   = Field(default=None, ge=1,    le=32)
    learning_rate:      Optional[float] = Field(default=None, ge=1e-6, le=1e-2)
    max_seq_length:     Optional[int]   = Field(default=None, ge=64,   le=4096)
    lora_rank:          Optional[int]   = Field(default=None, ge=2,    le=64)
    lora_alpha:         Optional[int]   = Field(default=None, ge=1,    le=128)
    lora_dropout:       Optional[float] = Field(default=None, ge=0.0,  le=0.5)
    # Explicit LoRA target modules override auto-detection.
    # Example: ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    lora_target_modules: Optional[List[str]] = Field(default=None, max_length=20)

    @field_validator("lora_target_modules")
    @classmethod
    def validate_lora_targets(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        import re
        pattern = re.compile(r"^[a-zA-Z0-9_]+$")
        for name in v:
            if not pattern.match(name):
                raise ValueError(
                    f"Invalid lora_target_module name '{name}'. "
                    "Only alphanumeric characters and underscores are allowed."
                )
        return v


class TrainRequest(BaseModel):
    """
    Accepted dataset formats per example:
      - Chat: {"messages": [{"role": ..., "content": ...}, ...]}
      - Alpaca-style: {"instruction": "...", "input": "...", "output": "..."}
        ("input" is optional)
    Both are normalised to the chat format before training.
    """
    run_name:      Optional[str]              = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-]*$",
        description="Human-readable label for this training run.",
    )
    dataset:       List[Dict[str, Any]]       = Field(
        ..., min_length=1, max_length=1_000,
        description="List of training examples.",
    )
    hyperparams:   Optional[HyperParams]      = None
    system_prompt: Optional[str]              = Field(
        default=None,
        max_length=2_048,
        description="If provided, prepended as a system message to every example "
                    "that does not already have one.",
    )

    @field_validator("dataset")
    @classmethod
    def validate_dataset_schema(cls, v: List[Dict]) -> List[Dict]:
        for i, ex in enumerate(v):
            has_messages    = "messages" in ex
            has_instruction = "instruction" in ex and "output" in ex
            if not has_messages and not has_instruction:
                raise ValueError(
                    f"Example {i}: must contain a 'messages' list "
                    f"OR both 'instruction' and 'output' keys. Got: {list(ex.keys())}"
                )
        return v


class TrainResponse(BaseModel):
    job_id:              str
    run_name:            str
    status:              str = "accepted"
    dataset_path:        str
    adapter_output_path: str
    message:             str


# ── Status / health ───────────────────────────────────────────────────────────

class TrainingJobStatus(BaseModel):
    job_id:        Optional[str]            = None
    run_name:      Optional[str]            = None
    state:         TrainingState            = TrainingState.IDLE
    progress_pct:  Optional[float]          = None
    current_step:  Optional[int]            = None
    total_steps:   Optional[int]            = None
    loss:          Optional[float]          = None
    started_at:    Optional[datetime]       = None
    completed_at:  Optional[datetime]       = None
    adapter_path:  Optional[str]            = None
    error:         Optional[str]            = None


class StatusResponse(BaseModel):
    server_version:       str              = "1.0.0"
    vllm_state:           VLLMState
    base_model:           str
    active_lora_name:     Optional[str]    = None
    active_lora_adapter:  Optional[str]    = None
    training:             TrainingJobStatus
    uptime_seconds:       float
    timestamp:            datetime


class HealthResponse(BaseModel):
    status:          str    # "ok" | "degraded"
    vllm_up:         bool
    training_active: bool
    timestamp:       datetime


# ── LoRA management ───────────────────────────────────────────────────────────

class LoadLoRARequest(BaseModel):
    lora_name: str = Field(..., pattern=r"^[a-zA-Z0-9_\-]+$")
    lora_path: str


class LoadLoRAResponse(BaseModel):
    success:    bool
    method:     str             # "dynamic" | "restart"
    lora_name:  Optional[str]   = None
    message:    str


class RestartVLLMResponse(BaseModel):
    success:    bool
    vllm_state: VLLMState
    message:    str


# ── Compare ───────────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    prompt:        str            = Field(..., max_length=4_096, description="Prompt sent to both models.")
    model_a:       Optional[str] = Field(default=None, max_length=128, description="First model or adapter name.")
    model_b:       Optional[str] = Field(default=None, max_length=128, description="Second model or adapter name.")
    system_prompt: Optional[str] = Field(default=None, max_length=2_048, description="System prompt prepended to both conversations.")
    thinking:      bool           = Field(default=False, description="Enable chain-of-thought for both models.")


# ── Thinking budget ───────────────────────────────────────────────────────────

# Named budget levels → token counts
BUDGET_PRESETS: Dict[str, int] = {
    "low":    256,
    "medium": 2048,
    "high":   8192,
}

class ThinkRequest(BaseModel):
    prompt:        str  = Field(..., max_length=4_096, description="User prompt to send to the model.")
    budget:        Any  = Field(
        default="medium",
        description=(
            "Thinking token budget. Either a preset string ('low', 'medium', 'high') "
            "or a positive integer (exact token count, max 16384)."
        ),
    )
    system_prompt: Optional[str] = Field(default=None, max_length=2_048, description="Optional system prompt.")
    model:         Optional[str] = Field(default=None, max_length=128, description="Model or adapter name.")

    @field_validator("budget")
    @classmethod
    def validate_budget(cls, v: Any) -> int:
        if isinstance(v, str):
            if v not in BUDGET_PRESETS:
                raise ValueError(f"budget must be one of {list(BUDGET_PRESETS)} or a positive integer")
            return BUDGET_PRESETS[v]
        if isinstance(v, int):
            if v < 1:
                raise ValueError("budget must be a positive integer")
            if v > 16_384:
                raise ValueError("budget cannot exceed 16384 tokens")
            return v
        raise ValueError("budget must be a string preset or a positive integer")


# ── Logit lens ───────────────────────────────────────────────────────────────

class LogitLensRequest(BaseModel):
    prompt:    str           = Field(..., max_length=2_048, description="Prompt to analyse.")
    top_k:     int           = Field(default=5, ge=1, le=20, description="Top-k token predictions per layer.")
    lora_path: Optional[str] = Field(default=None, max_length=512, description="Path to a LoRA adapter to merge.")


# ── Debate ────────────────────────────────────────────────────────────────────

class DebateRequest(BaseModel):
    topic:  str           = Field(..., max_length=512, description="Topic or question to debate.")
    rounds: int           = Field(default=2, ge=1, le=4, description="Number of back-and-forth rounds.")
    model:  Optional[str] = Field(default=None, max_length=128, description="Model to use for both sides.")


# ── Constrain ─────────────────────────────────────────────────────────────────

class ConstrainRequest(BaseModel):
    prompt:      str           = Field(..., max_length=4_096, description="User prompt.")
    constraints: List[str]     = Field(
        ..., min_length=1, max_length=10,
        description="Rules the model must follow.",
    )
    model:       Optional[str] = Field(default=None, max_length=128, description="Model or adapter name.")
    thinking:    bool          = Field(default=False, description="Enable chain-of-thought.")

    @field_validator("constraints")
    @classmethod
    def validate_constraints(cls, v: List[str]) -> List[str]:
        for c in v:
            if len(c) > 256:
                raise ValueError("Each constraint must be 256 characters or fewer.")
        return v


# ── Evolve ────────────────────────────────────────────────────────────────────

class EvolveRequest(BaseModel):
    prompt:            str           = Field(..., max_length=4_096, description="Original user prompt.")
    previous_response: Optional[str] = Field(default=None, max_length=8_192, description="Model's previous answer.")
    feedback:          Optional[str] = Field(default=None, max_length=2_048, description="User critique.")
    model:             Optional[str] = Field(default=None, max_length=128, description="Model or adapter name.")
    thinking:          bool          = Field(default=True, description="Enable chain-of-thought.")
