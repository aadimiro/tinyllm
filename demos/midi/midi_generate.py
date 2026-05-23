"""
MIDI Melody Generation
======================
Generate new melodies from the trained model and save as MIDI files.

The model generates token sequences autoregressively (one token at a time),
then we convert those tokens back into MIDI notes and write a .mid file
that you can play in any media player or DAW.
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from model import TinyLLM
from midi_config import (
    MidiModelConfig, BOS_TOKEN, EOS_TOKEN, BAR_TOKEN, VOCAB_SIZE,
    NOTE_OFFSET, DUR_OFFSET, REST_OFFSET
)
from midi_tokenizer import (
    tokens_to_midi_notes, token_to_name,
    is_note_token, is_duration_token, is_rest_token
)


def load_midi_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained MIDI model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = TinyLLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded MIDI model from {checkpoint_path}")
    print(f"  Parameters: {model.count_parameters():,}")
    return model, model_config


@torch.no_grad()
def generate_melody(model, max_tokens: int = 256,
                    temperature: float = 0.8, top_k: int = 20,
                    device: str = "cpu") -> list[int]:
    """
    Generate a melody token sequence.

    Args:
        max_tokens: Maximum tokens to generate
        temperature: Controls randomness (0.5=conservative, 1.0=varied, 1.5=wild)
        top_k: Only sample from top K tokens

    Returns:
        List of token IDs representing a melody
    """
    # Start with BOS + BAR
    tokens = [BOS_TOKEN, BAR_TOKEN]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        # Crop to block size
        idx_cond = idx[:, -model.config.block_size:]

        # Get predictions
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # Last position

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
            if next_token.item() == EOS_TOKEN:
                break
            tokens.append(next_token.item())
            continue

        # Top-k filtering
        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_vals[:, [-1]]] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

        token_id = next_token.item()
        tokens.append(token_id)

        if token_id == EOS_TOKEN:
            break

    return tokens


def tokens_to_midi_file(tokens: list[int], output_path: str,
                        tempo: int = 120):
    """
    Convert token sequence to a MIDI file.

    Args:
        tokens: List of token IDs from the model
        output_path: Where to save the .mid file
        tempo: Beats per minute
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi required: pip install pretty_midi")

    # Convert tokens to note events
    notes = tokens_to_midi_notes(tokens)

    if not notes:
        print("Warning: No notes generated!")
        return

    # Create MIDI file
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # Convert tick-based notes to seconds
    ticks_per_beat = 480
    seconds_per_tick = 60.0 / (tempo * ticks_per_beat)

    for note in notes:
        start_sec = note["start"] * seconds_per_tick
        end_sec = note["end"] * seconds_per_tick
        midi_note = pretty_midi.Note(
            velocity=note["velocity"],
            pitch=note["pitch"],
            start=start_sec,
            end=end_sec,
        )
        instrument.notes.append(midi_note)

    midi.instruments.append(instrument)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    midi.write(output_path)
    duration = max(n["end"] for n in notes) * seconds_per_tick
    print(f"Saved: {output_path} ({len(notes)} notes, {duration:.1f}s, {tempo} BPM)")


def print_melody(tokens: list[int]):
    """Print melody tokens in human-readable format."""
    line = []
    for t in tokens:
        name = token_to_name(t)
        if t == BAR_TOKEN:
            if line:
                print("  " + " ".join(line))
            line = ["|"]
        elif t in (BOS_TOKEN, EOS_TOKEN):
            continue
        else:
            line.append(name)
    if line:
        print("  " + " ".join(line))


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI melodies")
    parser.add_argument("--checkpoint", type=str,
                        default="demos/midi/checkpoints/best.pt")
    parser.add_argument("--output", type=str, default="demos/midi/output")
    parser.add_argument("--num", type=int, default=5,
                        help="Number of melodies to generate")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tempo", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if torch.cuda.is_available() and args.device == "cpu":
        args.device = "cuda"

    model, _ = load_midi_model(args.checkpoint, args.device)

    print(f"\nGenerating {args.num} melodies (temp={args.temperature}, "
          f"top_k={args.top_k}, tempo={args.tempo} BPM)\n")

    os.makedirs(args.output, exist_ok=True)

    for i in range(args.num):
        print(f"─── Melody {i+1} ───")
        tokens = generate_melody(
            model, max_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
            device=args.device
        )

        # Print readable representation
        print_melody(tokens)

        # Save as MIDI
        output_path = os.path.join(args.output, f"melody_{i+1:03d}.mid")
        tokens_to_midi_file(tokens, output_path, tempo=args.tempo)
        print()


if __name__ == "__main__":
    main()
