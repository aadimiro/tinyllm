"""
TinyLLM — The Complete Model
=============================
This file assembles all components into the full language model.

Full architecture (top to bottom):
-----------------------------------

    Input token IDs: [42, 103, 7, 891, ...]
            │
            ▼
    ┌─────────────────────────────┐
    │   Token Embedding           │  Look up each token ID → vector of size n_embd
    │   (vocab_size × n_embd)     │  "42" → [0.12, -0.34, 0.56, ...]
    └─────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────┐
    │   Position Embedding        │  Add position information: "I am token #3"
    │   (block_size × n_embd)     │  Without this, the model can't tell word order!
    └─────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────┐
    │   Dropout                   │  Regularization
    └─────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────┐
    │   Transformer Block 1       │  ← attention + FFN (see transformer_block.py)
    ├─────────────────────────────┤
    │   Transformer Block 2       │
    ├─────────────────────────────┤
    │   Transformer Block 3       │
    ├─────────────────────────────┤
    │   Transformer Block 4       │
    └─────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────┐
    │   Final LayerNorm           │  Stabilize before output
    └─────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────┐
    │   Output Head (Linear)      │  Project back to vocabulary size
    │   (n_embd → vocab_size)     │  Each position produces logits over all tokens
    └─────────────────────────────┘
            │
            ▼
    Logits: probability distribution over next token
    "The most likely next token is token #1847 with score 3.7"

Weight Tying:
    The output head shares weights with the token embedding.
    Intuition: if "cat" and "dog" have similar embeddings (both animals),
    then when the model wants to output an animal, both should have
    similar logits. Sharing weights enforces this naturally and saves parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from transformer_block import TransformerBlock


class TinyLLM(nn.Module):
    """
    The complete TinyLLM model: a decoder-only transformer for language modeling.

    This model takes a sequence of token IDs and predicts the next token at
    each position. During training, we compare predictions to actual next tokens.
    During inference, we use the predictions to generate text autoregressively.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # --- Embedding layers ---
        # Token embedding: maps each token ID to a learned vector
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embedding: maps each position (0, 1, 2, ...) to a learned vector
        # This tells the model WHERE in the sequence each token is
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Dropout after embeddings (regularization)
        self.drop = nn.Dropout(config.dropout)

        # --- Transformer blocks ---
        # Stack of N identical transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # --- Output layers ---
        # Final layer normalization
        self.ln_final = nn.LayerNorm(config.n_embd)

        # Output head: projects from n_embd back to vocab_size
        # Each position produces a score for every token in vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # --- Weight tying ---
        # Share weights between token embedding and output head.
        # This is a standard technique that:
        # 1. Reduces parameter count (saves vocab_size × n_embd parameters)
        # 2. Improves model quality (embeddings and output are consistent)
        self.token_embedding.weight = self.lm_head.weight

        # --- Weight initialization ---
        self.apply(self._init_weights)

        # Special scaled init for residual projections (GPT-2 trick)
        # Scale down output projections by 1/√(2*n_layer) to prevent
        # residual stream from growing too large in deep models
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights following GPT-2 conventions.

        - Linear layers: Normal distribution with std=0.02
        - Embeddings: Normal distribution with std=0.02
        - LayerNorm: weight=1, bias=0 (identity transform initially)

        Why 0.02? It's small enough to start training stably but large enough
        to break symmetry between neurons.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass of the language model.

        Args:
            idx:     Token IDs, shape (batch_size, seq_len)
            targets: Target token IDs for loss computation, shape (batch_size, seq_len)
                     If None, only return logits (inference mode).

        Returns:
            logits: Raw scores for each token in vocabulary, (batch_size, seq_len, vocab_size)
            loss:   Cross-entropy loss (only if targets provided)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Step 1: Token embeddings — look up vectors for each token
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)

        # Step 2: Position embeddings — add position information
        positions = torch.arange(0, T, device=idx.device)  # [0, 1, 2, ..., T-1]
        pos_emb = self.position_embedding(positions)  # (T, n_embd)

        # Step 3: Combine token + position embeddings
        # Broadcasting: pos_emb (T, C) is added to every batch element
        x = self.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        # Step 4: Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)  # (B, T, n_embd)

        # Step 5: Final layer norm
        x = self.ln_final(x)  # (B, T, n_embd)

        # Step 6: Project to vocabulary size → logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Step 7: Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # Ignore padding tokens (marked as -1)
            )

        return logits, loss

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_memory_mb(self) -> float:
        """Estimate model memory in MB (float32)."""
        return self.count_parameters() * 4 / (1024 * 1024)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        This is the core generation loop:
        1. Feed current sequence to the model
        2. Take the logits for the LAST position (that's the prediction for next token)
        3. Sample a token from the probability distribution
        4. Append it to the sequence
        5. Repeat

        Args:
            idx:             Starting token IDs, shape (batch_size, seq_len)
            max_new_tokens:  How many tokens to generate
            temperature:     Controls randomness (0.0=greedy, 1.0=normal, >1.0=creative)
            top_k:           Only sample from top K most likely tokens (0=no filtering)

        Returns:
            Extended sequence with generated tokens appended
        """
        for _ in range(max_new_tokens):
            # Crop sequence to block_size (sliding window if needed)
            idx_cond = idx[:, -self.config.block_size:]

            # Forward pass — get predictions
            logits, _ = self.forward(idx_cond)

            # Take logits for the last position only
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature scaling
            # temperature < 1.0 → more confident (peaky distribution)
            # temperature > 1.0 → more random (flat distribution)
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                # Keep only the top K logits, set rest to -infinity
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_values[:, [-1]]] = float("-inf")

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # Sample next token from the distribution
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)

        return idx
