"""
TinyLLM Tokenizer
=================
Wraps SentencePiece to provide BPE (Byte Pair Encoding) tokenization.

Why BPE?
--------
- Character-level tokenization creates very long sequences (1 char = 1 token),
  which makes the model slower and harder to train.
- BPE learns common subword units from the data (e.g., "tion", "the", "ing"),
  so typical English text gets compressed ~4x (1 token ≈ 4 characters).
- A vocabulary of 8,000 tokens is a good balance: small enough to fit our
  tiny model, large enough to cover most English words and subwords.

Usage:
------
    # Train a new tokenizer on your text data:
    tokenizer = Tokenizer.train("data/corpus.txt", vocab_size=8000)

    # Load a pre-trained tokenizer:
    tokenizer = Tokenizer.load("data/tokenizer.model")

    # Encode text to token IDs:
    ids = tokenizer.encode("What is the capital of France?")

    # Decode token IDs back to text:
    text = tokenizer.decode(ids)
"""

import os
from pathlib import Path

import sentencepiece as spm


# Special token IDs — these are reserved at the start of the vocabulary
PAD_ID = 0    # Padding (unused tokens in a batch)
BOS_ID = 1    # Beginning of sequence
EOS_ID = 2    # End of sequence (signals "stop generating")
UNK_ID = 3    # Unknown token (rare/unseen characters)


class Tokenizer:
    """
    BPE tokenizer backed by SentencePiece.

    SentencePiece handles the full BPE algorithm:
    1. Start with individual characters as the vocabulary
    2. Repeatedly merge the most frequent adjacent pair
    3. Stop when vocabulary reaches target size

    The result: common words like "the" are single tokens,
    rare words get split into subwords ("unforgettable" → "un" + "forget" + "table").
    """

    def __init__(self, model_path: str):
        """Load a trained SentencePiece model from disk."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return self.sp.GetPieceSize()

    def encode(self, text: str) -> list[int]:
        """
        Convert text to a list of token IDs.

        Example:
            "Hello world" → [423, 87, 1102]
        """
        return self.sp.Encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Convert token IDs back to text.

        Example:
            [423, 87, 1102] → "Hello world"
        """
        return self.sp.Decode(ids)

    def encode_as_pieces(self, text: str) -> list[str]:
        """
        Show how text gets split into subword pieces (useful for debugging).

        Example:
            "unforgettable" → ["▁un", "forget", "table"]
        """
        return self.sp.EncodeAsPieces(text)

    @staticmethod
    def train(input_file: str, model_prefix: str = "data/tokenizer",
              vocab_size: int = 8000) -> "Tokenizer":
        """
        Train a new BPE tokenizer from a text file.

        Args:
            input_file:   Path to plain text file (one document per line, or raw text)
            model_prefix: Where to save the model (creates .model and .vocab files)
            vocab_size:   Target vocabulary size

        Returns:
            A ready-to-use Tokenizer instance.

        The training process:
        1. Reads all text from input_file
        2. Counts all character pairs
        3. Merges most frequent pairs iteratively
        4. Stops at vocab_size tokens
        5. Saves the merge rules to model_prefix.model
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)

        # Train SentencePiece BPE model
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",           # Byte Pair Encoding
            pad_id=PAD_ID,              # Reserve ID 0 for padding
            bos_id=BOS_ID,              # Reserve ID 1 for beginning-of-sequence
            eos_id=EOS_ID,              # Reserve ID 2 for end-of-sequence
            unk_id=UNK_ID,              # Reserve ID 3 for unknown tokens
            normalization_rule_name="identity",  # Don't modify the text
            character_coverage=0.9995,  # Cover 99.95% of characters
            num_threads=4,              # Parallel processing
        )

        model_path = f"{model_prefix}.model"
        print(f"Tokenizer trained and saved to: {model_path}")
        print(f"Vocabulary size: {vocab_size}")
        return Tokenizer(model_path)

    @staticmethod
    def load(model_path: str) -> "Tokenizer":
        """Load a previously trained tokenizer."""
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Tokenizer model not found: {model_path}\n"
                f"Run train_tokenizer.py first to create it."
            )
        return Tokenizer(model_path)
