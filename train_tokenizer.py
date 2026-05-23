"""
Train the BPE Tokenizer
========================
This script trains a SentencePiece BPE tokenizer on the SQuAD corpus.

Run this BEFORE training the model:
    python train_tokenizer.py

What it does:
1. Downloads SQuAD v1.1 (if not already downloaded)
2. Extracts all text (contexts + questions + answers)
3. Trains a BPE tokenizer with 8000 vocabulary entries
4. Saves the tokenizer model to data/tokenizer.model

The tokenizer learns which character combinations appear frequently
in our training data and creates efficient subword tokens for them.
"""

from data import prepare_tokenizer_corpus, download_squad
from tokenizer import Tokenizer
from config import ModelConfig


def main():
    config = ModelConfig()

    # Step 1: Download SQuAD data
    print("Step 1: Downloading SQuAD v1.1...")
    download_squad()

    # Step 2: Prepare corpus for tokenizer training
    print("\nStep 2: Preparing tokenizer corpus...")
    corpus_path = prepare_tokenizer_corpus()

    # Step 3: Train the tokenizer
    print(f"\nStep 3: Training BPE tokenizer (vocab_size={config.vocab_size})...")
    tokenizer = Tokenizer.train(
        input_file=corpus_path,
        model_prefix="data/tokenizer",
        vocab_size=config.vocab_size,
    )

    # Step 4: Verify the tokenizer works
    print("\nStep 4: Verification...")
    test_texts = [
        "What is the capital of France?",
        "The Eiffel Tower is located in Paris.",
        "Context: Question: Answer:",
    ]
    for text in test_texts:
        ids = tokenizer.encode(text)
        pieces = tokenizer.encode_as_pieces(text)
        decoded = tokenizer.decode(ids)
        print(f"\n  Original:  {text}")
        print(f"  Pieces:    {pieces}")
        print(f"  Token IDs: {ids}")
        print(f"  Decoded:   {decoded}")
        assert decoded.strip() == text.strip(), "Round-trip encoding failed!"

    print(f"\n✓ Tokenizer trained successfully!")
    print(f"  Model saved to: data/tokenizer.model")
    print(f"  Vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
