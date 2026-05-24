# TinyLLM — A Tiny Language Model From Scratch

A complete, educational implementation of a GPT-style transformer language model
that fits in <50MB of RAM at inference and can train on a single CPU.

**Goal**: Demonstrate that modern LLM architecture works at any scale — same
principles as GPT-4, just 10,000× smaller. Every file is self-contained and
heavily commented to explain *why*, not just *what*.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        TinyLLM                             │
│                                                            │
│  Input tokens → Token Embed + Position Embed               │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────────────────────────────────────────┐          │
│  │  Transformer Block ×4                        │          │
│  │  ┌────────────────────────────────────────┐  │          │
│  │  │ LayerNorm → Multi-Head Attention (8h)  │  │          │
│  │  │    + Residual Connection               │  │          │
│  │  ├────────────────────────────────────────┤  │          │
│  │  │ LayerNorm → FFN (256→1024→256, GELU)  │  │          │
│  │  │    + Residual Connection               │  │          │
│  │  └────────────────────────────────────────┘  │          │
│  └──────────────────────────────────────────────┘          │
│       │                                                    │
│       ▼                                                    │
│  LayerNorm → Linear Head → Logits (vocab_size)             │
│                                                            │
│  Parameters: ~5.3M | Memory: ~21MB | Context: 256 tokens   │
└────────────────────────────────────────────────────────────┘
```

## Specs

| Property | Value |
|----------|-------|
| Architecture | Decoder-only transformer (GPT-2 style) |
| Layers | 4 |
| Attention heads | 8 |
| Embedding dimension | 256 |
| Context window | 256 tokens |
| Vocabulary | 8,000 (BPE) |
| Parameters | ~5.3M |
| Training memory | <400MB |
| Inference memory | <50MB |
| Training data | SQuAD v1.1 (100K Q&A pairs) |
| Task | Extractive Question Answering |

## Project Structure

The model architecture is **shared** — demos only change what goes in (data) and what comes out.

```
tinyllm/
├── config.py              # Hyperparameters (shared)
├── attention.py           # Multi-head self-attention (shared)
├── transformer_block.py   # Transformer block: attention + FFN (shared)
├── model.py               # Full model: embeddings + blocks + head (shared)
├── utils.py               # Checkpointing helpers (shared)
├── requirements.txt
│
├── demos/
│   ├── squad/             # Demo 1: Q&A on text (SQuAD dataset)
│   │   ├── squad_tokenizer.py    # BPE text tokenizer
│   │   ├── squad_data.py         # SQuAD download + formatting
│   │   ├── squad_train.py        # Training loop for Q&A
│   │   ├── squad_generate.py     # Interactive Q&A inference
│   │   ├── train_tokenizer.py    # Script to build BPE vocab
│   │   └── TinyLLM_Colab.ipynb   # Run on Google Colab
│   │
│   └── midi/              # Demo 2: Melody generation (Nottingham folk tunes)
│       ├── midi_config.py         # 68-token musical vocabulary
│       ├── midi_tokenizer.py      # MIDI ↔ token conversion
│       ├── midi_data.py           # Download + process MIDI files
│       ├── midi_train.py          # Training loop for music
│       ├── midi_generate.py       # Generate .mid files
│       └── TinyLLM_MIDI_Colab.ipynb
```

**Key insight:** `model.py` is identical for text and music. The transformer
doesn't "know" what it's processing — it just learns patterns between tokens.

## Reading Order (learn the architecture)

| # | File | What it teaches |
|---|------|-----------------|
| 1 | `config.py` | All hyperparameters — what knobs exist and what they control |
| 2 | `attention.py` | The attention mechanism — how tokens "look at" each other |
| 3 | `transformer_block.py` | One complete block — attention + FFN + residuals |
| 4 | `model.py` | Full model assembly — embeddings + blocks + output head |
| 5 | `demos/squad/` | Text Q&A demo — BPE tokenization, SQuAD data, generation |
| 6 | `demos/midi/` | Music demo — same model, completely different domain |

## Quick Start

### Demo 1: Text Q&A (SQuAD)

```bash
pip install -r requirements.txt
cd demos/squad
python train_tokenizer.py   # Download SQuAD + train BPE tokenizer
python squad_train.py       # Train (~hours on CPU, ~30 min on GPU)
python squad_generate.py --checkpoint checkpoints/best.pt
```

### Demo 2: MIDI Melody Generation

```bash
pip install -r requirements.txt
pip install pretty_midi
cd demos/midi
python midi_train.py        # Train on folk melodies (~5-10 min)
python midi_generate.py     # Generate .mid files you can listen to
```

### Google Colab (GPU, no setup)

- Q&A: `https://colab.research.google.com/github/aadimiro/tinyllm/blob/master/demos/squad/TinyLLM_Colab.ipynb`
- MIDI: `https://colab.research.google.com/github/aadimiro/tinyllm/blob/master/demos/midi/TinyLLM_MIDI_Colab.ipynb`

## How It Works (for learners)

### What is a Language Model?

A language model predicts the next word (token) given all previous words.
If you see "The cat sat on the ___", a good model assigns high probability to "mat".

### What is a Transformer?

A transformer is a neural network architecture that uses **attention** to let
every token in a sequence directly communicate with every other token. Before
transformers, models processed sequences one step at a time (RNNs). Transformers
process everything in parallel, which is faster and captures long-range dependencies better.

### What is Attention?

Attention is like a database lookup:
- Each token creates a **Query** ("what am I looking for?")
- Each token creates a **Key** ("what do I contain?")  
- Each token creates a **Value** ("what information can I give?")

The Query matches against all Keys to find relevant tokens, then retrieves
their Values. The formula: `softmax(Q·K^T / √d) · V`

### What makes this "GPT-style"?

1. **Decoder-only**: Only generates text (no separate encoder for input)
2. **Causal masking**: Each token can only see tokens before it (not future tokens)
3. **Autoregressive**: Generates one token at a time, feeding output back as input

### Why is it so small?

| What we scaled down | Full GPT-2 | TinyLLM | Factor |
|---------------------|-----------|---------|--------|
| Parameters | 124M | 5.3M | 23× smaller |
| Layers | 12 | 4 | 3× fewer |
| Embedding dim | 768 | 256 | 3× narrower |
| Vocabulary | 50,257 | 8,000 | 6× smaller |
| Context | 1,024 | 256 | 4× shorter |

Same architecture, same training procedure, just smaller numbers everywhere.

## Design Decisions

- **Pre-norm** (LayerNorm before attention): More stable training than post-norm
- **No bias** in linear layers: Modern practice (LLaMA, PaLM), slightly fewer params
- **Weight tying**: Embedding and output head share weights (saves params, improves quality)
- **GELU activation**: Smoother than ReLU, standard in modern transformers
- **AdamW optimizer**: Adam with proper weight decay (not L2 regularization)
- **Cosine LR decay**: Smooth schedule, better than step decay for language models

## License

MIT — use for learning, teaching, experimentation.
