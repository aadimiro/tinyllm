"""
MIDI Melody Training
====================
Train TinyLLM on melodies instead of text.

Same training loop as train.py, but uses MIDI data and config.
"""

import math
import os
import sys
import time

import torch
import torch.nn as nn

# Add parent directory to path so we can import the shared model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from model import TinyLLM
from config import ModelConfig

from midi_config import MidiModelConfig, MidiTrainConfig
from midi_data import get_midi_dataloaders


def get_lr(step: int, config: MidiTrainConfig) -> float:
    """Learning rate schedule with warmup + cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)
    decay_steps = config.max_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / decay_steps
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.learning_rate - config.min_lr) * cosine_factor


@torch.no_grad()
def estimate_loss(model, val_loader, config, device):
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    for _ in range(config.eval_steps):
        try:
            x, y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x, y = next(val_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train():
    """Train TinyLLM on MIDI melodies."""
    midi_config = MidiModelConfig()
    train_config = MidiTrainConfig()

    # Auto-detect GPU
    if torch.cuda.is_available():
        train_config.device = "cuda"
    device = train_config.device
    print(f"Using device: {device}")

    # Reuse the same TinyLLM model, just with MIDI config
    # We bridge MidiModelConfig → ModelConfig
    model_config = ModelConfig(
        n_layer=midi_config.n_layer,
        n_head=midi_config.n_head,
        n_embd=midi_config.n_embd,
        block_size=midi_config.block_size,
        vocab_size=midi_config.vocab_size,
        dropout=midi_config.dropout,
        bias=midi_config.bias,
    )

    # Data
    train_loader, val_loader = get_midi_dataloaders(midi_config, train_config)

    # Model
    model = TinyLLM(model_config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model memory: {model.estimate_memory_mb():.1f} MB")
    print(f"Vocabulary: {midi_config.vocab_size} tokens (musical events)")

    # Optimizer
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=train_config.learning_rate, betas=(train_config.beta1, train_config.beta2))

    # Training loop
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Training MIDI melody model for {train_config.max_steps} steps")
    print(f"{'='*60}\n")

    t0 = time.time()

    for step in range(train_config.max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        lr = get_lr(step, train_config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        if step % train_config.log_interval == 0:
            elapsed = time.time() - t0
            print(f"Step {step:>5d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                  f"{elapsed:.1f}s")
            t0 = time.time()

        if step > 0 and step % train_config.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, train_config, device)
            print(f"  ► Val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "midi_config": midi_config,
                    "step": step,
                    "loss": val_loss,
                }
                path = os.path.join(train_config.checkpoint_dir, "best.pt")
                torch.save(ckpt, path)
                print(f"  ► Saved best model (val_loss={val_loss:.4f})")

    # Final save
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "midi_config": midi_config,
        "step": train_config.max_steps,
        "loss": loss.item(),
    }
    torch.save(ckpt, os.path.join(train_config.checkpoint_dir, "final.pt"))
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
