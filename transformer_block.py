"""
TinyLLM Transformer Block
==========================
A single transformer block — the repeating unit that gets stacked N times
to form the full model.

Architecture of one block:
--------------------------

    ┌─────────────────────────┐
    │       Input (x)         │
    ├─────────────────────────┤
    │                         │
    │  ┌───────────────────┐  │
    │  │    LayerNorm 1    │  │  ← Normalize before attention
    │  └───────────────────┘  │
    │           │              │
    │  ┌───────────────────┐  │
    │  │  Multi-Head Attn  │  │  ← Tokens communicate with each other
    │  └───────────────────┘  │
    │           │              │
    │     x + attention_out    │  ← RESIDUAL CONNECTION (skip connection)
    │           │              │
    │  ┌───────────────────┐  │
    │  │    LayerNorm 2    │  │  ← Normalize before FFN
    │  └───────────────────┘  │
    │           │              │
    │  ┌───────────────────┐  │
    │  │   Feed-Forward    │  │  ← Each token processed independently
    │  │   (MLP: up→GELU→  │  │     (expand 4x, nonlinearity, compress back)
    │  │    down)           │  │
    │  └───────────────────┘  │
    │           │              │
    │     x + ffn_out          │  ← RESIDUAL CONNECTION
    │                         │
    ├─────────────────────────┤
    │       Output            │
    └─────────────────────────┘

Key concepts:
- LayerNorm BEFORE the sublayer (Pre-Norm): more stable training than Post-Norm
- Residual connections: let gradients flow directly through the network
- Feed-forward network: gives the model non-linear processing capacity
- GELU activation: smooth version of ReLU, standard in modern transformers
"""

import torch
import torch.nn as nn

from config import ModelConfig
from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN / MLP).

    This processes each token independently (no interaction between positions).
    It's where most of the model's "knowledge" gets stored — the attention
    mechanism routes information, but the FFN processes it.

    Architecture:
        Linear(n_embd → 4*n_embd)  ← "expand" to higher dimension
        GELU activation            ← nonlinearity (lets model learn complex functions)
        Linear(4*n_embd → n_embd)  ← "compress" back to model dimension
        Dropout                    ← regularization

    Why 4x expansion?
        This is a convention from the original transformer paper. The FFN needs
        enough capacity to store learned patterns. 4x is a good balance between
        capacity and parameter count.

    Why GELU (not ReLU)?
        GELU = x * Φ(x) where Φ is the Gaussian CDF.
        Unlike ReLU (which harshly cuts at 0), GELU is smooth everywhere.
        This gives better gradients and is standard in GPT-2, BERT, etc.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd  # Expansion factor of 4

        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_embd)
        Returns:
            (batch_size, seq_len, n_embd)
        """
        x = self.up_proj(x)        # (B, T, 4*C)
        x = nn.functional.gelu(x)  # Nonlinear activation
        x = self.down_proj(x)      # (B, T, C)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual.

    The model stacks N of these blocks. Each block:
    - Lets tokens exchange information (via attention)
    - Processes that information (via FFN)
    - Preserves the original signal (via residual connections)

    Deeper blocks tend to capture more abstract/high-level patterns,
    while early blocks handle more local/syntactic patterns.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Pre-norm: normalize BEFORE each sublayer
        # This is more stable than the original Post-Norm from "Attention Is All You Need"
        self.ln1 = nn.LayerNorm(config.n_embd)  # Norm before attention
        self.ln2 = nn.LayerNorm(config.n_embd)  # Norm before FFN

        # The two sublayers
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: (batch_size, seq_len, n_embd)
        Returns:
            (batch_size, seq_len, n_embd)
        """
        # Sublayer 1: Attention with residual connection
        # "x + ..." is the residual connection — it lets the gradient
        # flow directly from output back to input, solving the vanishing
        # gradient problem in deep networks.
        x = x + self.attention(self.ln1(x))

        # Sublayer 2: FFN with residual connection
        x = x + self.ffn(self.ln2(x))

        return x
