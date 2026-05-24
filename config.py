"""
TinyLLM Configuration
=====================
All model and training hyperparameters live here in a single dataclass.
This makes it easy to experiment: change one number, retrain, compare results.

The architecture is a decoder-only transformer (GPT-2 style), scaled down
to ~3-4M parameters so it fits in <512MB RAM during training and <50MB at inference.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Defines the transformer architecture.

    Parameter count estimate (approximate):
        - Token embedding:      vocab_size × n_embd         = 8000 × 256 = 2.05M
        - Position embedding:   block_size × n_embd         = 256 × 256  = 65K
        - Per transformer block:
            - Attention (QKV):  3 × n_embd × n_embd         = 3 × 256² = 196K
            - Attention (out):  n_embd × n_embd             = 256²     = 65K
            - FFN (up):         n_embd × (4 × n_embd)      = 256 × 1024 = 262K
            - FFN (down):       (4 × n_embd) × n_embd      = 1024 × 256 = 262K
            - LayerNorms:       4 × n_embd                  = 1K
            - Total per block:  ~786K
        - All blocks:           4 × 786K                    = 3.14M
        - Final LayerNorm:      2 × n_embd                  = 512
        - Output head:          (tied with token embedding, so 0 extra)
        ─────────────────────────────────────────────────────────────────
        Total:                  ~5.3M parameters

    Memory at inference (float32): ~21MB for weights alone.
    With KV cache for 256 tokens:  ~25MB total.
    """

    # --- Architecture ---
    n_layer: int = 4          # Number of transformer blocks (depth)
    n_head: int = 8           # Number of attention heads (parallelism in attention)
    n_embd: int = 256         # Embedding dimension (width of the model)
    block_size: int = 256     # Maximum sequence length (context window)
    vocab_size: int = 8000    # BPE vocabulary size (set after tokenizer training)

    # --- Regularization ---
    dropout: float = 0.1      # Dropout rate (prevents overfitting on small data)

    # --- Architecture choices ---
    bias: bool = False        # Use bias in linear layers? (modern practice: no bias)


@dataclass
class TrainConfig:
    """
    Training hyperparameters.

    These are tuned for training on a single CPU or modest GPU.
    With batch_size=8 and block_size=256, each batch processes
    8 × 256 = 2048 tokens simultaneously.
    """

    # --- Optimization ---
    learning_rate: float = 3e-4     # Peak learning rate (AdamW)
    weight_decay: float = 0.01      # L2 regularization (excludes biases & norms)
    beta1: float = 0.9              # Adam momentum
    beta2: float = 0.99             # Adam variance (0.99 works well for small models)
    grad_clip: float = 1.0          # Gradient clipping (prevents training explosions)

    # --- Schedule ---
    max_steps: int = 20000          # Total training steps
    warmup_steps: int = 500         # Linear warmup from 0 to learning_rate
    min_lr: float = 3e-5            # Minimum LR at end of cosine decay (10% of peak)

    # --- Batching ---
    batch_size: int = 8             # Sequences per batch
    # Effective tokens per step: batch_size × block_size = 8 × 256 = 2048

    # --- Logging & checkpoints ---
    eval_interval: int = 500        # Evaluate every N steps
    eval_steps: int = 50            # Number of batches to average for eval loss
    log_interval: int = 100         # Print training loss every N steps
    checkpoint_interval: int = 2000 # Save model every N steps

    # --- Data ---
    train_split: float = 0.9        # 90% train, 10% validation
    data_dir: str = "data"          # Where processed data lives (relative to demo dir)
    checkpoint_dir: str = "checkpoints"  # Where model weights are saved

    # --- Device ---
    device: str = "cpu"             # Will be overridden to "cuda" if available
