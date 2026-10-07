from datetime import datetime
from pathlib import Path

import torch


def run_name(name: str) -> str:
    """Return the current date and time when the name is auto."""
    if name == "auto":
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return name


def run_dir(run: str) -> Path:
    """Return the directory for one run."""
    return Path("run") / run


def save_model(model: torch.nn.Module, run: str, name: str) -> None:
    """Write a model state dict under run/<run>/checkpoints/."""
    path = run_dir(run) / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
