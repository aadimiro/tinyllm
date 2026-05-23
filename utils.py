"""
TinyLLM Utilities
=================
Helper functions for checkpointing, parameter counting, and logging.
"""

import os
from dataclasses import asdict

import torch

from config import ModelConfig, TrainConfig


def save_checkpoint(model, optimizer, step: int, loss: float,
                    model_config: ModelConfig, train_config: TrainConfig,
                    path: str):
    """
    Save a training checkpoint.

    We save everything needed to resume training OR to load the model for inference:
    - Model weights (the learned parameters)
    - Optimizer state (momentum buffers — needed to resume training smoothly)
    - Training step (where we left off)
    - Current loss (for logging)
    - Configs (so we can reconstruct the model architecture)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "model_config": model_config,
        "train_config": train_config,
    }

    torch.save(checkpoint, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Checkpoint saved: {path} ({size_mb:.1f} MB)")


def load_checkpoint(path: str, model, optimizer=None,
                    device: str = "cpu") -> int:
    """
    Load a training checkpoint and restore model/optimizer state.

    Args:
        path: Path to the .pt checkpoint file
        model: The model to load weights into
        optimizer: Optional optimizer to restore state (for resuming training)
        device: Device to load tensors to

    Returns:
        The training step at which the checkpoint was saved
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["step"]


def count_parameters(model) -> dict:
    """
    Count parameters by component (useful for understanding where memory goes).

    Returns a dict like:
        {
            "token_embedding": 2048000,
            "position_embedding": 65536,
            "blocks.0.attention": 263168,
            ...
            "total": 5242880
        }
    """
    counts = {}
    total = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            # Group by top-level component
            component = name.split(".")[0]
            if "blocks" in name:
                # More detail for transformer blocks
                parts = name.split(".")
                component = f"{parts[0]}.{parts[1]}.{parts[2]}"
            counts[component] = counts.get(component, 0) + n
            total += n

    counts["total"] = total
    return counts


def print_model_summary(model):
    """Print a human-readable summary of model architecture and parameter counts."""
    counts = count_parameters(model)
    total = counts.pop("total")

    print(f"\n{'─'*50}")
    print(f"{'Component':<35} {'Params':>10} {'%':>6}")
    print(f"{'─'*50}")

    for name, count in sorted(counts.items()):
        pct = 100.0 * count / total
        print(f"  {name:<33} {count:>10,} {pct:>5.1f}%")

    print(f"{'─'*50}")
    print(f"  {'TOTAL':<33} {total:>10,}")
    print(f"  {'Memory (float32)':<33} {total * 4 / 1024 / 1024:>8.1f} MB")
    print(f"{'─'*50}\n")
