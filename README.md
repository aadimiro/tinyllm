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

## File Structure (Reading Order)

Start here to understand the architecture bottom-up:

| # | File | What it teaches |
|---|------|-----------------|
| 1 | `config.py` | All hyperparameters — what knobs exist and what they control |
| 2 | `attention.py` | The attention mechanism — how tokens "look at" each other |
| 3 | `transformer_block.py` | One complete block — attention + FFN + residuals |
| 4 | `model.py` | Full model assembly — embeddings + blocks + output head |
| 5 | `tokenizer.py` | BPE tokenization — how text becomes numbers |
| 6 | `data.py` | Data pipeline — loading, formatting, batching |
| 7 | `train.py` | Training loop — forward, backward, optimize |
| 8 | `generate.py` | Inference — how the model generates text |
| 9 | `utils.py` | Checkpointing and logging helpers |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the tokenizer

Downloads SQuAD and trains a BPE tokenizer:

```bash
python train_tokenizer.py
```

### 3. Train the model

```bash
python train.py
```

Training on CPU takes several hours. On a GPU it takes ~30 minutes.
You'll see loss decrease from ~9.0 (random) to ~2.0 (learning patterns).

### 4. Ask questions

```bash
python generate.py --checkpoint checkpoints/best.pt
```

Or non-interactively:

```bash
python generate.py \
  --context "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889." \
  --question "When was the Eiffel Tower built?" \
  --checkpoint checkpoints/best.pt
```

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
