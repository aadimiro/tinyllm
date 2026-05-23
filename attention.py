"""
TinyLLM Attention Mechanism
============================
This file implements the core attention mechanism — the key innovation
that makes transformers work.

What is Attention?
------------------
Attention lets each token "look at" every other token in the sequence and
decide how much to pay attention to it. It's like reading a sentence and
being able to connect "it" back to the noun it refers to.

The formula (Scaled Dot-Product Attention):

    Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

Where:
    Q (Query):  "What am I looking for?"
    K (Key):    "What do I contain?"
    V (Value):  "What information do I provide?"
    d_k:        dimension of keys (for numerical stability)

Multi-Head Attention:
    Instead of one big attention operation, we split into multiple "heads"
    that each attend to different aspects of the input (e.g., one head might
    track syntax, another semantics, another position relationships).

Causal Masking:
    In a decoder (text generation), each token can only attend to tokens
    that came BEFORE it (not future tokens). We enforce this with a triangular
    mask that sets future positions to -infinity before the softmax.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with causal (autoregressive) masking.

    Architecture:
        1. Project input into Q, K, V (three separate linear transforms)
        2. Split into multiple heads
        3. Compute attention scores: Q × K^T / √d_k
        4. Apply causal mask (prevent looking at future tokens)
        5. Softmax → attention weights
        6. Multiply weights × V to get attended output
        7. Concatenate all heads
        8. Final linear projection

    Shape flow (for batch_size=B, seq_len=T, n_embd=C, n_head=H):
        Input:      (B, T, C)
        Q, K, V:    (B, T, C) each
        Reshaped:   (B, H, T, C//H)  ← split into heads
        Scores:     (B, H, T, T)     ← attention matrix
        Output:     (B, T, C)        ← concatenated heads
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            f"Embedding dim {config.n_embd} must be divisible by n_head {config.n_head}"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # Dimension per head

        # Linear projections for Q, K, V (combined into one matrix for efficiency)
        # This is equivalent to three separate nn.Linear(n_embd, n_embd) layers,
        # but faster because we do one matrix multiply instead of three.
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection: combines all heads back into a single vector
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout for attention weights and output
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: a lower-triangular matrix of 1s
        # Position (i, j) = 1 means "token i can attend to token j"
        # Since j <= i for causal attention, it's lower-triangular.
        #
        # We register it as a buffer (not a parameter) so it's saved with the model
        # but not updated by the optimizer.
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.shape  # Batch size, sequence length, embedding dim

        # Step 1: Project input to Q, K, V
        # qkv shape: (B, T, 3*C) → split into three (B, T, C) tensors
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Step 2: Reshape into multiple heads
        # (B, T, C) → (B, T, H, head_dim) → (B, H, T, head_dim)
        # The transpose puts the head dimension before sequence length
        # so we can do batched matrix multiplication across all heads at once.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Step 3: Compute attention scores
        # Q × K^T → (B, H, T, T) — each position's similarity to every other
        # Scale by 1/√d_k to prevent softmax saturation with large dimensions
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Step 4: Apply causal mask
        # Set future positions to -infinity → softmax turns them into 0
        # This ensures token at position i only attends to positions 0..i
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        # Step 5: Softmax → attention weights (probabilities that sum to 1)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Weighted sum of values
        # (B, H, T, T) × (B, H, T, head_dim) → (B, H, T, head_dim)
        attended = torch.matmul(attn_weights, v)

        # Step 7: Concatenate heads
        # (B, H, T, head_dim) → (B, T, H, head_dim) → (B, T, C)
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)

        # Step 8: Final output projection
        output = self.out_proj(attended)
        output = self.resid_dropout(output)

        return output
