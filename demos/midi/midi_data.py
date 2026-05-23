"""
MIDI Data Pipeline
==================
Downloads a collection of folk melodies in MIDI format and prepares them
for training.

Dataset: Nottingham Music Database
    ~1000 folk tunes (jigs, reels, hornpipes, etc.)
    Monophonic melodies — perfect for our tiny model.
    Free and well-studied in music generation research.

Alternative: We also include a fallback that generates synthetic melodies
(scales, arpeggios, simple patterns) if the download fails.
"""

import os
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

from midi_config import MidiModelConfig, MidiTrainConfig, BOS_TOKEN, EOS_TOKEN, VOCAB_SIZE
from midi_tokenizer import midi_to_tokens, pitch_to_token, duration_to_token, rest_to_token
from midi_tokenizer import BAR_TOKEN, DURATION_BINS, PITCH_MIN, PITCH_MAX, NOTE_OFFSET, DUR_OFFSET


# Nottingham dataset URL (ABC format converted to MIDI)
NOTTINGHAM_URL = "https://github.com/jukedeck/nottingham-dataset/archive/refs/heads/master.zip"


def download_nottingham(data_dir: str = "demos/midi/data") -> str:
    """Download the Nottingham MIDI dataset."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "nottingham.zip")
    extract_dir = os.path.join(data_dir, "nottingham")

    if os.path.exists(extract_dir) and len(list(Path(extract_dir).rglob("*.mid"))) > 0:
        print(f"Nottingham dataset already downloaded at {extract_dir}")
        return extract_dir

    print("Downloading Nottingham MIDI dataset...")
    urlretrieve(NOTTINGHAM_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    # The zip extracts to nottingham-dataset-master/
    extracted = os.path.join(data_dir, "nottingham-dataset-master", "MIDI")
    if os.path.exists(extracted):
        os.rename(extracted, extract_dir)
    else:
        # Try alternative structure
        for d in Path(data_dir).iterdir():
            if d.is_dir() and "nottingham" in d.name.lower():
                midi_dir = d / "MIDI"
                if midi_dir.exists():
                    os.rename(str(midi_dir), extract_dir)
                    break

    # Cleanup
    if os.path.exists(zip_path):
        os.remove(zip_path)

    midi_files = list(Path(extract_dir).rglob("*.mid"))
    print(f"Downloaded {len(midi_files)} MIDI files")
    return extract_dir


def load_midi_file(path: str) -> list[dict]:
    """
    Load a MIDI file and extract monophonic melody notes.

    Returns list of note dicts: {pitch, start, end} in ticks.
    """
    if not HAS_PRETTY_MIDI:
        raise ImportError("pretty_midi is required: pip install pretty_midi")

    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return []

    # Get all notes from all instruments, take the melody (highest notes)
    all_notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            all_notes.append({
                "pitch": note.pitch,
                "start": int(note.start * 480),  # Convert seconds to ticks (480 tpb)
                "end": int(note.end * 480),
            })

    if not all_notes:
        return []

    # Sort by start time
    all_notes.sort(key=lambda n: (n["start"], -n["pitch"]))

    # For polyphonic input, extract the melody (top voice):
    # Keep only the highest note at each time position
    melody = []
    for note in all_notes:
        # Skip if overlaps with previous note and is lower
        if melody and note["start"] < melody[-1]["end"]:
            if note["pitch"] <= melody[-1]["pitch"]:
                continue
            else:
                # This note is higher — truncate previous
                melody[-1]["end"] = note["start"]
        melody.append(note)

    return melody


def generate_synthetic_melodies(num_melodies: int = 500) -> list[list[int]]:
    """
    Generate synthetic melody token sequences as fallback data.

    Creates simple but musically valid patterns:
    - Major/minor scales (ascending, descending)
    - Arpeggios (triads, 7th chords)
    - Simple repeated patterns with variation
    """
    random.seed(42)
    melodies = []

    # Major scale intervals (in semitones from root)
    major = [0, 2, 4, 5, 7, 9, 11, 12]
    minor = [0, 2, 3, 5, 7, 8, 10, 12]
    pentatonic = [0, 2, 4, 7, 9, 12]

    scales = [major, minor, pentatonic]
    roots = list(range(48, 72, 1))  # C3 to B4 (various starting notes)

    for _ in range(num_melodies):
        tokens = [BOS_TOKEN, BAR_TOKEN]
        scale = random.choice(scales)
        root = random.choice(roots)
        num_bars = random.randint(4, 8)
        notes_per_bar = random.choice([4, 6, 8])

        for bar in range(num_bars):
            if bar > 0:
                tokens.append(BAR_TOKEN)

            for beat in range(notes_per_bar):
                # Choose a scale degree with some randomness
                degree = random.choice(range(len(scale)))
                pitch = root + scale[degree]

                # Occasional octave jump
                if random.random() < 0.1:
                    pitch += 12 if random.random() < 0.5 else -12

                # Clamp to valid range
                pitch = max(PITCH_MIN, min(PITCH_MAX, pitch))

                # Choose duration
                dur_idx = random.choices(
                    range(len(DURATION_BINS)),
                    weights=[1, 4, 1, 4, 2, 2, 1, 1],  # Favor eighths and quarters
                )[0]

                tokens.append(pitch_to_token(pitch))
                tokens.append(DUR_OFFSET + dur_idx)

                # Occasional rest
                if random.random() < 0.15:
                    rest_idx = random.choice([0, 1, 2])  # Short rests
                    tokens.append(rest_to_token(DURATION_BINS[rest_idx]))

        tokens.append(EOS_TOKEN)
        melodies.append(tokens)

    return melodies


def load_all_melodies(data_dir: str = "demos/midi/data") -> list[list[int]]:
    """
    Load all MIDI files and convert to token sequences.
    Falls back to synthetic data if MIDI loading fails.
    """
    token_sequences = []

    # Try loading real MIDI files
    if HAS_PRETTY_MIDI:
        midi_dir = download_nottingham(data_dir)
        midi_files = list(Path(midi_dir).rglob("*.mid"))
        print(f"Processing {len(midi_files)} MIDI files...")

        for path in midi_files:
            notes = load_midi_file(str(path))
            if len(notes) >= 8:  # Need at least 8 notes
                tokens = midi_to_tokens(notes)
                if 20 <= len(tokens) <= 1000:  # Reasonable length
                    token_sequences.append(tokens)

        print(f"Successfully tokenized {len(token_sequences)} melodies")

    # Fall back to synthetic if not enough real data
    if len(token_sequences) < 100:
        print("Generating synthetic melodies as supplement...")
        synthetic = generate_synthetic_melodies(500)
        token_sequences.extend(synthetic)
        print(f"Total melodies: {len(token_sequences)}")

    return token_sequences


class MelodyDataset(Dataset):
    """
    Dataset of tokenized melodies for training.

    Same principle as QADataset: fixed-length sequences with
    input shifted by 1 for next-token prediction.
    """

    def __init__(self, sequences: list[list[int]], block_size: int = 512):
        self.block_size = block_size
        self.data = []

        for seq in sequences:
            # Truncate long sequences
            if len(seq) > block_size + 1:
                # Take a random window (more data augmentation)
                seq = seq[:block_size + 1]
            if len(seq) >= 10:  # Minimum useful length
                self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]

        # Pad to block_size + 1
        padded = seq + [0] * (self.block_size + 1 - len(seq))
        padded = padded[:self.block_size + 1]

        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)

        # Mark padding as -1 (ignored in loss)
        actual_len = min(len(seq), self.block_size + 1)
        if actual_len < self.block_size + 1:
            y[actual_len - 1:] = -1

        return x, y


def get_midi_dataloaders(model_config: MidiModelConfig,
                         train_config: MidiTrainConfig):
    """Prepare train/val DataLoaders for melody training."""
    sequences = load_all_melodies(train_config.data_dir)

    # Shuffle and split
    random.seed(42)
    random.shuffle(sequences)
    split_idx = int(len(sequences) * train_config.train_split)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]

    print(f"Train: {len(train_seqs)} melodies, Val: {len(val_seqs)} melodies")

    train_dataset = MelodyDataset(train_seqs, model_config.block_size)
    val_dataset = MelodyDataset(val_seqs, model_config.block_size)

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size,
                            shuffle=False, drop_last=True, num_workers=0)

    return train_loader, val_loader
