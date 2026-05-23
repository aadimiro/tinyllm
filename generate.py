"""
TinyLLM Text Generation / Inference
====================================
Interactive Q&A with the trained model.

How autoregressive generation works:
-------------------------------------
1. Start with a prompt: "Context: ... Question: ... Answer:"
2. Feed all tokens to the model
3. Model outputs probability distribution over next token
4. Sample (or pick the most likely) next token
5. Append that token to the sequence
6. Repeat from step 2 until we hit EOS or max length

Sampling strategies:
--------------------
- Greedy (temperature=0): Always pick the most likely token. Deterministic but boring.
- Temperature sampling: Scale logits by 1/T before softmax.
    T < 1.0 → sharper distribution (more confident, less diverse)
    T > 1.0 → flatter distribution (more creative, more random)
- Top-k: Only consider the K most likely tokens. Prevents low-probability gibberish.
- Top-p (nucleus): Only consider tokens whose cumulative probability exceeds p.
  Adaptive: uses fewer tokens when the model is confident, more when uncertain.

For Q&A, we typically want low temperature (0.3-0.7) and top_k=50 for
focused, factual answers.
"""

import argparse
import sys

import torch

from config import ModelConfig
from model import TinyLLM
from tokenizer import Tokenizer, EOS_ID
from utils import load_checkpoint


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[TinyLLM, ModelConfig]:
    """
    Load a trained model from a checkpoint.

    Returns the model and its config.
    """
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]

    # Create model and load weights
    model = TinyLLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set to evaluation mode (disables dropout)

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Memory: {model.estimate_memory_mb():.1f} MB")
    print(f"  Trained for {checkpoint['step']} steps")

    return model, model_config


def generate_answer(model: TinyLLM, tokenizer: Tokenizer,
                    context: str, question: str,
                    max_tokens: int = 64, temperature: float = 0.5,
                    top_k: int = 50, device: str = "cpu") -> str:
    """
    Generate an answer given a context and question.

    This formats the input as the model expects, then generates tokens
    until EOS or max_tokens is reached.
    """
    # Format prompt (same format as training data, but stop before the answer)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    # Encode prompt to token IDs
    input_ids = tokenizer.encode(prompt)

    # Check if prompt fits in context window
    if len(input_ids) > model.config.block_size - max_tokens:
        # Truncate context from the left (keep question + answer space)
        max_prompt_len = model.config.block_size - max_tokens
        input_ids = input_ids[-max_prompt_len:]
        print(f"  (Prompt truncated to {max_prompt_len} tokens)")

    # Convert to tensor
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate tokens one by one
    generated_ids = []
    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop to block_size
            idx_cond = idx[:, -model.config.block_size:]

            # Get model predictions
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]  # Last position only

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            else:
                # temperature=0 → greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_token], dim=1)
                token_id = next_token.item()
                if token_id == EOS_ID:
                    break
                generated_ids.append(token_id)
                continue

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, [-1]]] = float("-inf")

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)

            token_id = next_token.item()
            if token_id == EOS_ID:
                break
            generated_ids.append(token_id)

    # Decode generated tokens back to text
    answer = tokenizer.decode(generated_ids).strip()
    return answer


def interactive_qa(checkpoint_path: str, device: str = "cpu"):
    """
    Interactive Q&A session: user provides context and question, model answers.

    Type 'quit' to exit.
    """
    # Load model and tokenizer
    model, model_config = load_model(checkpoint_path, device)
    tokenizer = Tokenizer.load("data/tokenizer.model")

    print(f"\n{'='*60}")
    print("TinyLLM Interactive Q&A")
    print("="*60)
    print("Provide a context paragraph and a question.")
    print("The model will try to extract the answer from the context.")
    print("Type 'quit' to exit.\n")

    while True:
        # Get context from user
        print("-" * 40)
        print("Enter context (or 'quit'):")
        context = input("> ").strip()
        if context.lower() == "quit":
            break

        # Get question
        print("Enter question:")
        question = input("> ").strip()
        if question.lower() == "quit":
            break

        # Generate answer
        print("\nGenerating answer...")
        answer = generate_answer(
            model, tokenizer, context, question,
            max_tokens=64, temperature=0.5, top_k=50, device=device
        )
        print(f"\n  Answer: {answer}\n")


def batch_generate(checkpoint_path: str, examples: list[dict],
                   device: str = "cpu", **kwargs) -> list[str]:
    """
    Generate answers for a batch of examples (for evaluation).

    Args:
        examples: List of dicts with 'context' and 'question' keys
        **kwargs: Generation parameters (temperature, top_k, max_tokens)

    Returns:
        List of generated answer strings
    """
    model, _ = load_model(checkpoint_path, device)
    tokenizer = Tokenizer.load("data/tokenizer.model")

    answers = []
    for ex in examples:
        answer = generate_answer(
            model, tokenizer, ex["context"], ex["question"],
            device=device, **kwargs
        )
        answers.append(answer)

    return answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with TinyLLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--context", type=str, default=None,
                        help="Context text (for non-interactive use)")
    parser.add_argument("--question", type=str, default=None,
                        help="Question (for non-interactive use)")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature (0=greedy, 1=normal)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0=disabled)")
    parser.add_argument("--max_tokens", type=int, default=64,
                        help="Maximum tokens to generate")
    args = parser.parse_args()

    if args.context and args.question:
        # Non-interactive: single question
        model, _ = load_model(args.checkpoint, args.device)
        tokenizer = Tokenizer.load("data/tokenizer.model")
        answer = generate_answer(
            model, tokenizer, args.context, args.question,
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_k=args.top_k, device=args.device
        )
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        interactive_qa(args.checkpoint, args.device)
