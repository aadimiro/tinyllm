# TinyLLM MIDI — Melody Generation with a Tiny Transformer

Same model architecture as the Q&A demo, but trained on **music** instead of text.
The transformer learns musical patterns from folk melodies, then generates new ones.

## The Core Idea

In text:
```
"The cat sat on the" → model predicts → "mat"
```

In music:
```
[NOTE_C4, DUR_4, NOTE_E4, DUR_4, NOTE_G4] → model predicts → [DUR_8]
```

The model doesn't "know" it's processing music. It just learns statistical patterns
between tokens — which happen to encode notes, durations, and rests instead of words.

## How Music Becomes Tokens

A melody is a sequence of events. We encode each event as one or two tokens:

```
Original melody:    C4 quarter → E4 quarter → rest eighth → G4 half
                    ↓
Token sequence:     [BOS, BAR, NOTE_C4, DUR_4, NOTE_E4, DUR_4, REST_2, NOTE_G4, DUR_8, EOS]
                    ↓
Token IDs:          [1,   3,   40,      54,    44,      54,    61,     47,      57,    2]
```

### Vocabulary (68 tokens total)

| Token Type | Count | Meaning |
|-----------|-------|---------|
| Special | 4 | PAD, BOS (start), EOS (end), BAR (bar line) |
| NOTE_{pitch} | 48 | Which note to play (C2 to B5 = 4 octaves) |
| DUR_{steps} | 8 | How long to hold it (sixteenth to whole note) |
| REST_{steps} | 8 | Silence for a duration |

This is much smaller than the text vocabulary (8,000 BPE tokens), which means
the model can focus entirely on learning *musical structure* rather than spending
capacity on vocabulary.

### Duration Quantization

Durations are measured in 16th notes and quantized to 8 bins:

| Bin | 16th notes | Musical name |
|-----|-----------|--------------|
| 1 | 1 | Sixteenth note |
| 2 | 2 | Eighth note |
| 3 | 3 | Dotted eighth |
| 4 | 4 | Quarter note |
| 6 | 6 | Dotted quarter |
| 8 | 8 | Half note |
| 12 | 12 | Dotted half |
| 16 | 16 | Whole note |

## Training Data

**Nottingham Music Database**: ~1000 British/Irish folk tunes (jigs, reels, hornpipes).

Why folk music?
- **Monophonic** (single melody line) — no need to handle chords or harmony
- **Repetitive structure** — clear patterns the tiny model can learn
- **Short** — most tunes fit within our 512-token context window
- **Well-studied** — used in many music generation papers for benchmarking

If the download fails, the code generates synthetic melodies (scales, arpeggios,
simple patterns) as fallback training data.

## What the Model Learns

After training, the model captures:

1. **Scale patterns**: Notes tend to move in steps (C→D→E) rather than random jumps
2. **Rhythm patterns**: Quarter notes often follow eighth notes in folk music
3. **Phrase structure**: BAR tokens create regular groupings of 4 or 8 beats
4. **Melodic contour**: Melodies tend to rise and fall in arcs
5. **Repetition**: Folk tunes repeat phrases (AABB structure)

## Generation: Temperature Controls "Creativity"

```
Temperature = 0.3  → Very conservative, repetitive, "safe" melodies
Temperature = 0.8  → Balanced — sounds musical but varied (recommended)
Temperature = 1.2  → Surprising, occasionally dissonant, experimental
Temperature = 2.0  → Nearly random — probably not musical
```

Under the hood: temperature divides the logits before softmax.
Low temperature → peaked distribution (always picks the obvious next note).
High temperature → flat distribution (any note is equally likely).

## Architecture (identical to text model)

```
Token IDs: [1, 3, 40, 54, 44, 54, ...]
     │
     ▼
Token Embedding (68 × 256) ──── much smaller than text (8000 × 256)
     │
     ▼
Position Embedding (512 × 256)
     │
     ▼
4 × Transformer Block
  ├── Multi-Head Attention (8 heads) ← notes attend to previous notes
  └── Feed-Forward Network           ← learns note relationships
     │
     ▼
Linear Head → logits over 68 tokens
     │
     ▼
"Next token is NOTE_G4 with probability 0.34"
```

Parameters: ~1.3M (smaller than text model because vocabulary is tiny)

## Running

```bash
# Install
conda activate tinyllm
pip install pretty_midi

# Train (~5-10 min CPU, ~2 min GPU)
cd demos/midi
python midi_train.py

# Generate melodies
python midi_generate.py --num 5 --temperature 0.8 --tempo 120

# Play (Linux)
timidity output/melody_001.mid

# Or open .mid files in VLC, GarageBand, LMMS, etc.
```

### Generation Options

```
--num 5              Number of melodies to generate
--temperature 0.8    Randomness (0.3=safe, 0.8=balanced, 1.2=wild)
--top_k 20           Only sample from top 20 most likely tokens
--tempo 120          Playback speed in BPM
--max_tokens 256     Maximum melody length
```

## File Structure

```
demos/midi/
├── midi_config.py        # Token vocabulary + model/training hyperparameters
├── midi_tokenizer.py     # MIDI ↔ token conversion (the "language" of music)
├── midi_data.py          # Dataset download, processing, DataLoader
├── midi_train.py         # Training loop (reuses model.py from parent)
├── midi_generate.py      # Generate melodies → .mid files
├── TinyLLM_MIDI_Colab.ipynb  # Run everything on Google Colab with audio playback
├── data/                 # Downloaded MIDI files (created at runtime)
├── checkpoints/          # Trained model weights
└── output/               # Generated .mid files
```

## Colab

Run the full demo with audio playback:
```
https://colab.research.google.com/github/aadimiro/tinyllm/blob/master/demos/midi/TinyLLM_MIDI_Colab.ipynb
```
