"""
TinyLLM Data Pipeline
=====================
Downloads SQuAD v1.1, formats it for causal language modeling, and provides
a simple DataLoader for training.

How we format Q&A for a decoder-only model:
-------------------------------------------
A decoder-only transformer generates text left-to-right. To teach it Q&A,
we concatenate context + question + answer into a single sequence:

    "Context: The Eiffel Tower is in Paris, France. It was built in 1889.
    Question: Where is the Eiffel Tower?
    Answer: Paris, France<|end|>"

The model learns to predict each next token. At inference time, we provide
everything up to "Answer:" and let the model generate the rest.

The <|end|> token (EOS) tells the model when to stop generating.
"""

import json
import os
import random
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset, DataLoader

from config import ModelConfig, TrainConfig
from tokenizer import Tokenizer, EOS_ID


# SQuAD v1.1 download URLs (Stanford's public hosting)
SQUAD_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


def download_squad(data_dir: str = "data/squad") -> tuple[str, str]:
    """
    Download SQuAD v1.1 train and dev sets if not already present.

    Returns paths to the downloaded JSON files.
    """
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train-v1.1.json")
    dev_path = os.path.join(data_dir, "dev-v1.1.json")

    if not os.path.exists(train_path):
        print("Downloading SQuAD training set...")
        urlretrieve(SQUAD_TRAIN_URL, train_path)
        print(f"  Saved to {train_path}")

    if not os.path.exists(dev_path):
        print("Downloading SQuAD dev set...")
        urlretrieve(SQUAD_DEV_URL, dev_path)
        print(f"  Saved to {dev_path}")

    return train_path, dev_path


def parse_squad(json_path: str) -> list[dict]:
    """
    Parse SQuAD JSON into a flat list of (context, question, answer) dicts.

    SQuAD structure:
        data → [articles] → paragraphs → [paragraph] → qas → [qa]
        Each qa has: question, answers[0].text

    We flatten this into simple dicts for easy processing.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    examples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                # SQuAD v1.1 always has at least one answer
                answer = qa["answers"][0]["text"]
                examples.append({
                    "context": context,
                    "question": question,
                    "answer": answer,
                })

    return examples


def format_example(example: dict) -> str:
    """
    Format a single Q&A example as a training string.

    This is the format the model will learn to complete:
        Context: {passage}
        Question: {question}
        Answer: {answer}

    At inference, we provide everything up to "Answer: " and generate the rest.
    """
    return (
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        f"Answer: {example['answer']}"
    )


class QADataset(Dataset):
    """
    PyTorch Dataset that serves tokenized Q&A sequences.

    Each item is a fixed-length tensor of token IDs. Sequences shorter than
    block_size are padded; longer sequences are truncated (we lose some context,
    but keep question + answer intact).

    For causal language modeling, the target is the input shifted by one position:
        input:  [tok1, tok2, tok3, tok4, ...]
        target: [tok2, tok3, tok4, tok5, ...]

    The model learns to predict the next token at every position.
    """

    def __init__(self, examples: list[dict], tokenizer: Tokenizer,
                 block_size: int = 256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []

        for ex in examples:
            text = format_example(ex)
            # Encode text and append EOS token
            token_ids = tokenizer.encode(text) + [EOS_ID]

            # Truncate if too long (keep the last block_size+1 tokens
            # so we preserve the answer portion)
            if len(token_ids) > block_size + 1:
                token_ids = token_ids[-(block_size + 1):]

            # Only keep examples that are long enough to be useful
            # (at least 10 tokens: some context + question + answer)
            if len(token_ids) >= 10:
                self.data.append(token_ids)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (input_ids, target_ids) tensors of shape (block_size,).

        - input_ids:  tokens [0, 1, ..., n-2] (what the model sees)
        - target_ids: tokens [1, 2, ..., n-1] (what the model predicts)

        Padded positions use target=-1 so they're ignored in the loss.
        """
        token_ids = self.data[idx]

        # Pad to block_size + 1 (we need one extra for the target shift)
        padded = token_ids + [0] * (self.block_size + 1 - len(token_ids))
        padded = padded[:self.block_size + 1]  # safety truncation

        x = torch.tensor(padded[:-1], dtype=torch.long)   # input
        y = torch.tensor(padded[1:], dtype=torch.long)    # target

        # Mark padding positions as -1 in targets (CrossEntropyLoss ignores -1)
        # Padding starts after the actual token sequence
        actual_len = min(len(token_ids), self.block_size + 1)
        if actual_len < self.block_size + 1:
            y[actual_len - 1:] = -1  # -1 = ignore in loss

        return x, y


def get_dataloaders(
    model_config: ModelConfig,
    train_config: TrainConfig,
    tokenizer: Tokenizer,
) -> tuple[DataLoader, DataLoader]:
    """
    Prepare train and validation DataLoaders from SQuAD.

    Steps:
    1. Download SQuAD if needed
    2. Parse into examples
    3. Split into train/val
    4. Create Datasets and DataLoaders
    """
    # Download and parse
    train_path, _ = download_squad(train_config.data_dir)
    examples = parse_squad(train_path)
    print(f"Loaded {len(examples)} Q&A examples from SQuAD")

    # Shuffle deterministically and split
    random.seed(42)
    random.shuffle(examples)

    split_idx = int(len(examples) * train_config.train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    print(f"Train: {len(train_examples)} examples, Val: {len(val_examples)} examples")

    # Create datasets
    train_dataset = QADataset(train_examples, tokenizer, model_config.block_size)
    val_dataset = QADataset(val_examples, tokenizer, model_config.block_size)
    print(f"After filtering: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,    # Drop incomplete final batch (simpler training loop)
        num_workers=0,     # Keep it simple — no multiprocess data loading
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    return train_loader, val_loader


def prepare_tokenizer_corpus(data_dir: str = "data/squad",
                             output_file: str = "data/corpus.txt") -> str:
    """
    Extract all text from SQuAD into a plain text file for tokenizer training.

    The tokenizer needs to see all the text it will encounter during training
    so it can learn the best subword splits for this domain.
    """
    train_path, dev_path = download_squad(data_dir)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as out:
        for path in [train_path, dev_path]:
            examples = parse_squad(path)
            for ex in examples:
                # Write each formatted example as a line
                text = format_example(ex)
                # Replace newlines with spaces for SentencePiece (expects one doc per line)
                out.write(text.replace("\n", " ") + "\n")

    print(f"Wrote tokenizer corpus to {output_file} "
          f"({os.path.getsize(output_file) / 1024 / 1024:.1f} MB)")
    return output_file
