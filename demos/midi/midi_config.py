"""
MIDI Melody Generation Demo — Configuration
=============================================
Same TinyLLM architecture, but trained on melodies instead of text.

The model learns musical patterns (scales, chord progressions, rhythm)
from a collection of folk melodies, then generates new ones.

Tokenization scheme (simple, monophonic):
    NOTE_{pitch}  — which note to play (48 values: C2 to B5)
    DUR_{steps}   — how long to hold it (8 quantized durations)
    REST_{steps}  — silence for a duration (8 values)
    BAR           — marks the start of a new measure

Total vocabulary: 48 + 8 + 8 + 3 special = 67 tokens
(Much smaller than text! This helps the tiny model learn faster.)
"""

from dataclasses import dataclass


# --- MIDI Token Constants ---
# Pitches: MIDI notes 36 (C2) to 83 (B5) — 4 octaves, covers most melodies
PITCH_MIN = 36
PITCH_MAX = 83
NUM_PITCHES = PITCH_MAX - PITCH_MIN + 1  # 48

# Durations: quantized to musical subdivisions (in 16th notes)
# 1=sixteenth, 2=eighth, 3=dotted-eighth, 4=quarter,
# 6=dotted-quarter, 8=half, 12=dotted-half, 16=whole
DURATION_BINS = [1, 2, 3, 4, 6, 8, 12, 16]
NUM_DURATIONS = len(DURATION_BINS)  # 8

# Special tokens
PAD_TOKEN = 0
BOS_TOKEN = 1   # Beginning of melody
EOS_TOKEN = 2   # End of melody
BAR_TOKEN = 3   # Bar line

# Token ranges in vocabulary:
# [0]      PAD
# [1]      BOS
# [2]      EOS
# [3]      BAR
# [4..51]  NOTE_{pitch}  (48 pitches)
# [52..59] DUR_{steps}   (8 durations)
# [60..67] REST_{steps}  (8 durations)
SPECIAL_OFFSET = 4
NOTE_OFFSET = SPECIAL_OFFSET                    # 4
DUR_OFFSET = NOTE_OFFSET + NUM_PITCHES          # 52
REST_OFFSET = DUR_OFFSET + NUM_DURATIONS        # 60
VOCAB_SIZE = REST_OFFSET + NUM_DURATIONS        # 68


@dataclass
class MidiModelConfig:
    """
    Model config for MIDI melody generation.
    Smaller than the text model since vocabulary is tiny (68 vs 8000).
    """
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    block_size: int = 512     # Longer context: melodies need ~200-400 tokens
    vocab_size: int = VOCAB_SIZE  # 68 tokens (vs 8000 for text)
    dropout: float = 0.1
    bias: bool = False


@dataclass
class MidiTrainConfig:
    """Training config tuned for MIDI."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0

    max_steps: int = 10000        # Less data → fewer steps needed
    warmup_steps: int = 300
    min_lr: float = 3e-5

    batch_size: int = 16          # Can be larger with short sequences
    eval_interval: int = 500
    eval_steps: int = 30
    log_interval: int = 100
    checkpoint_interval: int = 2000

    train_split: float = 0.9
    data_dir: str = "demos/midi/data"
    checkpoint_dir: str = "demos/midi/checkpoints"
    device: str = "cpu"
