"""
TinyLLM Training Loop
=====================
A clean, minimal training loop for the TinyLLM model.

Training a language model:
--------------------------
1. Feed a batch of token sequences to the model
2. The model predicts the next token at each position
3. Compare predictions to actual next tokens (cross-entropy loss)
4. Compute gradients (backpropagation)
5. Update weights (AdamW optimizer step)
6. Repeat until loss converges

Learning Rate Schedule:
-----------------------
We use warmup + cosine decay:

    LR
    │     ╭──────────╮
    │    /            ╲
    │   /              ╲
    │  /                ╲___________
    │ /                              min_lr
    └──────────────────────────────────→ steps
      warmup     cosine decay

- Warmup: Start from 0, linearly increase to peak LR over warmup_steps.
  This prevents early training instability when gradients are noisy.
- Cosine decay: Smoothly decrease LR following a cosine curve.
  Slower decay than linear, gives the model more time at useful learning rates.
"""

import math
import os
import time

import torch
import torch.nn as nn

from config import ModelConfig, TrainConfig
from model import TinyLLM
from tokenizer import Tokenizer
from data import get_dataloaders
from utils import save_checkpoint, load_checkpoint


def get_lr(step: int, config: TrainConfig) -> float:
    """
    Compute learning rate for a given training step.

    Schedule:
    1. Linear warmup: 0 → peak LR over warmup_steps
    2. Cosine decay: peak LR → min_lr over remaining steps
    """
    # Phase 1: Linear warmup
    if step < config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)

    # Phase 2: Cosine decay
    # Map step to [0, 1] range for the decay phase
    decay_steps = config.max_steps - config.warmup_steps
    step_in_decay = step - config.warmup_steps
    progress = step_in_decay / decay_steps  # 0.0 → 1.0

    # Cosine decay formula: oscillates from 1 to 0 over [0, π]
    # We map this to [peak_lr, min_lr]
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.learning_rate - config.min_lr) * cosine_factor


@torch.no_grad()
def estimate_loss(model: TinyLLM, val_loader, config: TrainConfig,
                  device: str) -> float:
    """
    Estimate validation loss by averaging over eval_steps batches.

    We use @torch.no_grad() to disable gradient computation during evaluation —
    this saves memory and is faster since we don't need gradients here.
    """
    model.eval()  # Set model to evaluation mode (disables dropout)
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

    model.train()  # Back to training mode
    return sum(losses) / len(losses)


def train(resume_from: str = None):
    """
    Main training function.

    This orchestrates the entire training process:
    1. Setup (config, model, optimizer, data)
    2. Training loop (forward, backward, update)
    3. Periodic evaluation and checkpointing
    """
    # ─── Configuration ───────────────────────────────────────────────────────
    model_config = ModelConfig()
    train_config = TrainConfig()

    # Auto-detect GPU
    if torch.cuda.is_available():
        train_config.device = "cuda"
    device = train_config.device
    print(f"Using device: {device}")

    # ─── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = Tokenizer.load("data/tokenizer.model")
    # Update vocab_size to match the trained tokenizer
    model_config.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {model_config.vocab_size}")

    # ─── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(model_config, train_config, tokenizer)

    # ─── Model ───────────────────────────────────────────────────────────────
    model = TinyLLM(model_config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model memory (float32): {model.estimate_memory_mb():.1f} MB")

    # ─── Optimizer ───────────────────────────────────────────────────────────
    # AdamW: Adam with decoupled weight decay
    # We separate parameters into two groups:
    # 1. Parameters that get weight decay (linear weights)
    # 2. Parameters that DON'T get weight decay (biases, LayerNorm, embeddings)
    # This is important: regularizing biases/norms hurts performance.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:  # Weight matrices (2D+)
            decay_params.append(param)
        else:  # Biases and LayerNorm params (1D)
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=train_config.learning_rate, betas=(train_config.beta1, train_config.beta2))

    print(f"Optimizer groups: {len(decay_params)} decay, {len(no_decay_params)} no-decay")

    # ─── Resume from checkpoint ──────────────────────────────────────────────
    start_step = 0
    if resume_from:
        start_step = load_checkpoint(resume_from, model, optimizer, device)
        print(f"Resumed from step {start_step}")

    # ─── Training loop ───────────────────────────────────────────────────────
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Starting training for {train_config.max_steps} steps")
    print(f"Batch size: {train_config.batch_size}, Block size: {model_config.block_size}")
    print(f"Tokens per step: {train_config.batch_size * model_config.block_size:,}")
    print(f"{'='*60}\n")

    t0 = time.time()

    for step in range(start_step, train_config.max_steps):
        # ─── Get batch ───────────────────────────────────────────────────
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # ─── Update learning rate ────────────────────────────────────────
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ─── Forward pass ────────────────────────────────────────────────
        logits, loss = model(x, y)

        # ─── Backward pass ───────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)  # Clear old gradients
        loss.backward()                         # Compute new gradients

        # Gradient clipping: prevent exploding gradients
        # If the total gradient norm exceeds grad_clip, scale all gradients down
        nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        # ─── Optimizer step ──────────────────────────────────────────────
        optimizer.step()

        # ─── Logging ─────────────────────────────────────────────────────
        if step % train_config.log_interval == 0:
            elapsed = time.time() - t0
            tokens_per_sec = (train_config.batch_size * model_config.block_size *
                              train_config.log_interval) / max(elapsed, 1e-6)
            print(f"Step {step:>6d} | Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | {tokens_per_sec:.0f} tok/s")
            t0 = time.time()

        # ─── Evaluation ──────────────────────────────────────────────────
        if step > 0 and step % train_config.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, train_config, device)
            print(f"\n  ► Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss,
                                model_config, train_config,
                                os.path.join(train_config.checkpoint_dir, "best.pt"))
                print(f"  ► New best model saved (val_loss={val_loss:.4f})")
            print()

        # ─── Checkpointing ───────────────────────────────────────────────
        if step > 0 and step % train_config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, loss.item(),
                            model_config, train_config,
                            os.path.join(train_config.checkpoint_dir, f"step_{step}.pt"))

    # ─── Final save ──────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, train_config.max_steps, loss.item(),
                    model_config, train_config,
                    os.path.join(train_config.checkpoint_dir, "final.pt"))
    print(f"\nTraining complete! Final loss: {loss.item():.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train TinyLLM")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(resume_from=args.resume)
