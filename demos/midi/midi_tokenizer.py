"""
MIDI Tokenizer
==============
Converts MIDI files to/from token sequences.

A melody becomes a sequence like:
    [BOS, BAR, NOTE_C4, DUR_4, NOTE_E4, DUR_4, NOTE_G4, DUR_8, BAR, ...]

This is analogous to how text becomes token IDs, but instead of words,
we have musical events (notes, durations, rests, bar lines).

Each note event is TWO tokens: pitch + duration.
    "Play middle C for a quarter note" → [NOTE_60, DUR_4]

Rests are also two tokens:
    "Rest for an eighth note" → [REST, DUR_2]
    (Actually encoded as a single REST_dur token for simplicity)
"""

import numpy as np

from midi_config import (
    PITCH_MIN, PITCH_MAX, NUM_PITCHES, DURATION_BINS,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, BAR_TOKEN,
    NOTE_OFFSET, DUR_OFFSET, REST_OFFSET, VOCAB_SIZE
)


def pitch_to_token(pitch: int) -> int:
    """Convert MIDI pitch (36-83) to token ID."""
    pitch = max(PITCH_MIN, min(PITCH_MAX, pitch))
    return NOTE_OFFSET + (pitch - PITCH_MIN)


def token_to_pitch(token: int) -> int:
    """Convert token ID back to MIDI pitch."""
    return (token - NOTE_OFFSET) + PITCH_MIN


def duration_to_token(duration_16ths: float) -> int:
    """
    Quantize a duration (in 16th notes) to the nearest bin and return token.

    Example: a quarter note = 4 sixteenth notes → DUR_4 token
    """
    # Find closest duration bin
    closest_idx = min(range(len(DURATION_BINS)),
                      key=lambda i: abs(DURATION_BINS[i] - duration_16ths))
    return DUR_OFFSET + closest_idx


def token_to_duration(token: int) -> int:
    """Convert duration token back to number of 16th notes."""
    idx = token - DUR_OFFSET
    return DURATION_BINS[idx]


def rest_to_token(duration_16ths: float) -> int:
    """Quantize a rest duration and return rest token."""
    closest_idx = min(range(len(DURATION_BINS)),
                      key=lambda i: abs(DURATION_BINS[i] - duration_16ths))
    return REST_OFFSET + closest_idx


def token_to_rest_duration(token: int) -> int:
    """Convert rest token back to number of 16th notes."""
    idx = token - REST_OFFSET
    return DURATION_BINS[idx]


def is_note_token(token: int) -> bool:
    return NOTE_OFFSET <= token < DUR_OFFSET


def is_duration_token(token: int) -> bool:
    return DUR_OFFSET <= token < REST_OFFSET


def is_rest_token(token: int) -> bool:
    return REST_OFFSET <= token < VOCAB_SIZE


def midi_to_tokens(notes: list[dict], ticks_per_beat: int = 480,
                   beats_per_bar: int = 4) -> list[int]:
    """
    Convert a list of note events to a token sequence.

    Args:
        notes: List of dicts with keys: pitch, start, end (in ticks)
        ticks_per_beat: MIDI resolution (typically 480)
        beats_per_bar: Time signature numerator (typically 4)

    Returns:
        List of token IDs representing the melody
    """
    if not notes:
        return [BOS_TOKEN, EOS_TOKEN]

    # Sort by start time
    notes = sorted(notes, key=lambda n: n["start"])

    # Convert ticks to 16th notes
    ticks_per_16th = ticks_per_beat / 4
    ticks_per_bar = ticks_per_beat * beats_per_bar

    tokens = [BOS_TOKEN]
    current_time = 0  # in ticks

    for note in notes:
        start = note["start"]
        end = note["end"]
        pitch = note["pitch"]

        # Skip out-of-range pitches
        if pitch < PITCH_MIN or pitch > PITCH_MAX:
            continue

        # Add bar tokens at bar boundaries
        while current_time + ticks_per_bar <= start:
            current_time += ticks_per_bar
            tokens.append(BAR_TOKEN)

        # Add rest if there's a gap before this note
        gap = start - current_time
        if gap > ticks_per_16th * 0.5:  # At least half a 16th note
            rest_16ths = gap / ticks_per_16th
            tokens.append(rest_to_token(rest_16ths))

        # Add note event: pitch + duration
        duration_ticks = end - start
        duration_16ths = duration_ticks / ticks_per_16th
        tokens.append(pitch_to_token(pitch))
        tokens.append(duration_to_token(duration_16ths))

        current_time = end

    tokens.append(EOS_TOKEN)
    return tokens


def tokens_to_midi_notes(tokens: list[int], ticks_per_beat: int = 480,
                         beats_per_bar: int = 4) -> list[dict]:
    """
    Convert token sequence back to MIDI note events.

    Returns list of dicts with: pitch, start, end, velocity (in ticks)
    """
    ticks_per_16th = ticks_per_beat / 4
    ticks_per_bar = ticks_per_beat * beats_per_bar

    notes = []
    current_time = 0

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
            i += 1
            continue

        if token == BAR_TOKEN:
            # Snap to next bar boundary
            bar_position = current_time / ticks_per_bar
            current_time = int((int(bar_position) + 1) * ticks_per_bar)
            i += 1
            continue

        if is_rest_token(token):
            rest_16ths = token_to_rest_duration(token)
            current_time += int(rest_16ths * ticks_per_16th)
            i += 1
            continue

        if is_note_token(token):
            pitch = token_to_pitch(token)
            # Look for duration token
            if i + 1 < len(tokens) and is_duration_token(tokens[i + 1]):
                dur_16ths = token_to_duration(tokens[i + 1])
                i += 2
            else:
                dur_16ths = 4  # Default: quarter note
                i += 1

            start = current_time
            end = current_time + int(dur_16ths * ticks_per_16th)
            notes.append({
                "pitch": pitch,
                "start": start,
                "end": end,
                "velocity": 80,
            })
            current_time = end
            continue

        # Unknown token — skip
        i += 1

    return notes


def token_to_name(token: int) -> str:
    """Human-readable name for a token (for debugging)."""
    if token == PAD_TOKEN: return "PAD"
    if token == BOS_TOKEN: return "BOS"
    if token == EOS_TOKEN: return "EOS"
    if token == BAR_TOKEN: return "BAR"

    if is_note_token(token):
        pitch = token_to_pitch(token)
        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]
        name = note_names[pitch % 12]
        octave = (pitch // 12) - 1
        return f"NOTE_{name}{octave}"

    if is_duration_token(token):
        dur = token_to_duration(token)
        return f"DUR_{dur}"

    if is_rest_token(token):
        dur = token_to_rest_duration(token)
        return f"REST_{dur}"

    return f"UNK_{token}"
