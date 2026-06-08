"""Optional Weights & Biases logging, gated behind the WANDB env var.

Every function here is a no-op unless the run is started with `WANDB=1`, so
train.py / sft.py can call `metrics.init/log/summary/finish` unconditionally and
stay free of `if use_wandb:` clutter.

Usage:
    WANDB=1 uv run python train.py                 # log online (needs `wandb login`)
    WANDB=1 WANDB_MODE=offline uv run python train.py   # log locally, `wandb sync` later

`wandb` itself is only imported when enabled, so the dependency never loads on a
normal (unlogged) run.
"""

from __future__ import annotations

import os

_ENABLED = os.environ.get("WANDB", "0") == "1"
_run = None


def enabled() -> bool:
    return _ENABLED


def init(project: str, name: str | None = None, config: dict | None = None):
    """Start a run if WANDB=1; otherwise no-op. Returns the run (or None)."""
    global _run
    if not _ENABLED:
        return None
    import wandb

    _run = wandb.init(project=project, name=name, config=config or {})
    return _run


def log(values: dict, step: int | None = None) -> None:
    """Log a step's metrics. No-op if logging is disabled."""
    if _run is None:
        return
    import wandb

    wandb.log(values, step=step)


def summary(values: dict) -> None:
    """Set run-summary scalars (e.g. final eval numbers). No-op if disabled."""
    if _run is None:
        return
    for k, v in values.items():
        _run.summary[k] = v


def finish() -> None:
    global _run
    if _run is None:
        return
    import wandb

    wandb.finish()
    _run = None
