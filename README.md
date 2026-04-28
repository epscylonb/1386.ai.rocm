# 1386.ai.rocm

This is a fork of [1386.ai](https://github.com/eb1386/1386.ai) ported to **ROCm**, targeting specifically the AMD Strix Halo APU but compatible with any ROCm-supported hardware.

I found this repo through a Reddit post where the author (@eb1386) nonchalantly announced it after training a 235M-parameter model. Unlike most toy LLM implementations, this one is end-to-end — data prep, training, and fine-tuning included. The code is clean and accessible, making it an excellent reference for small-model training. Sadly the author has deleted their original post and comments, but you can see [others' feedback here](https://www.reddit.com/r/LocalLLaMA/comments/1srsxqs/235m_param_llm_from_scratch_on_a_single_rtx_5080/?show=original).

Regarding ROCm support on Strix Halo, there's good news and bad news.

The good news: despite ROCm's reputation lagging behind CUDA, virtually no PyTorch-specific code changes were needed to train a 500M-parameter model here. PyTorch's ROCm backend is genuinely solid.

The bad news: training a 500M-parameter model on the 128 GB Strix Halo APU (in a GMKTec Evo X2 mini PC) will take roughly three weeks. I'm seeing ~4,750 tokens/s — there's likely not much low-hanging fruit left without writing custom CUDA kernels or deeper fused-operator optimizations.

## Summary of Changes

- dataset.py
    - The original author omitted `ShardDataset` and `StreamingShardDataset` classes, so I have naively implemented these
    - Random shuffling of training data has been added to ensure that the model isn't trained on previously seen data when resuming training from a checkpoint
- `torch.compile`
    - Added to increase training perf
- Training workers changed from 2 to 0 (running on the main thread)
    - Couldn't get training to start using workers
- Added a `Dockerfile` and `run-docker.sh` helper script
    - ROCm drivers and libraries are notoriously difficult to install, configure, and maintain
    - Using a container avoids breaking the host with bad installs and config
    - Using the latest image from [https://hub.docker.com/r/rocm/pytorch/tags](rocm/pytorch)  

## Quick Start on Strix Halo

Consider editing the ENV vars in the `run-docker.sh` script to match your hardware and huggingface config.

```bash
# Build the image (base is > 6 GB)
docker build -t 1386-rocm .

# Run an interactive session:
bash run-docker.sh
```

Inside the container, follow the original instructions to download data and begin training.

What follows is the original readme from the forked repo.

# 1386.ai

A lightweight transformer language model built from scratch in PyTorch, trained on a single consumer GPU with a full pipeline for data processing, pretraining, and instruction tuning.

No pretrained weights, no HuggingFace model downloads. Every weight is learned from raw text on a single RTX 5080 using bf16 mixed precision with gradient checkpointing. The training infrastructure handles everything from data download through evaluation.

The current release is **Plasma 1.0** (235M parameters). **Plasma 1.1** (500M parameters, multi-turn conversation support, upgraded data pipeline) is in development.

---

## Architecture

The model follows the LLaMA architecture with modern training techniques throughout.

**Attention** uses Grouped-Query Attention (GQA) with query heads mapped to fewer key-value heads, reducing memory bandwidth during inference while maintaining quality. All positional information comes from Rotary Positional Embeddings (RoPE), encoding position directly into the attention computation rather than through learned position embeddings. KV caching is supported for fast autoregressive generation.

**Feed-forward layers** use SwiGLU, a gated activation function that replaces the traditional ReLU MLP. SwiGLU uses three linear projections (gate, up, down) with a SiLU-gated element-wise product, consistently outperforming standard two-projection FFNs at the same parameter count.

**Normalization** is RMSNorm applied before each sub-layer (pre-norm). RMSNorm drops the mean-centering of LayerNorm and only normalizes by the root mean square, which is faster and more stable during mixed-precision training.

The embedding and output projection weights are tied, cutting the parameter count without hurting performance.

### Plasma 1.0

| | |
|---|---|
| Parameters | 235M |
| Hidden size | 1024 |
| Layers | 18 |
| Attention | 16 heads, 4 KV heads (GQA) |
| FFN | SwiGLU, 2816 intermediate |
| Context | 1024 tokens |
| Vocab | 32,000 (SentencePiece BPE) |
| Precision | bf16 |

### Plasma 1.1 (in development)

| | |
|---|---|
| Parameters | 500M |
| Hidden size | 1280 |
| Layers | 26 |
| Attention | 20 heads, 4 KV heads (GQA) |
| FFN | SwiGLU, 3584 intermediate |
| Context | 1024 tokens |
| Vocab | 48,000 (SentencePiece BPE, byte fallback) |
| Multi-turn | yes |

---

## Demo (Plasma 1.0)

> **You:** Hello!
>
> **1386.ai:** Hi there, I'm the one who wrote this. How can I help you today?

> **You:** Name two famous people.
>
> **1386.ai:** Two of the most well-known individuals are John F. Kennedy, who was the 35th President of the United States and George W. Bush, a former president from the same country.

> **You:** When was World War 1?
>
> **1386.ai:** World War I began on June 26, 1914.

> **You:** Name 4 fruits.
>
> **1386.ai:** Fruits are the most common fruit that people eat, and they can be found in a variety of different types such as apples, oranges, bananas, pears, etc

> **You:** What is a steak made of?
>
> **1386.ai:** A steak can be made from various types of meat, including beef.

---

## Training

Training happens in two phases: pretraining on a large filtered corpus, then instruction tuning with loss masking.

**Pretraining** trains the model on billions of tokens of cleaned, deduplicated text from multiple sources. Training uses mixed-precision bf16 with gradient checkpointing to fit on a single consumer GPU. The learning rate follows a cosine schedule with linear warmup.

**Instruction tuning** teaches the model to follow a conversational format. Loss masking ensures the model only learns from assistant response tokens. User prompts are masked during backpropagation. Plasma 1.1 extends this to multi-turn conversations, masking all user turns across the full conversation history.

### Training Plasma 1.1

The 1.1 pipeline is a 12-stage process that handles everything from data download to a final inference test.

| Stage | What it does |
|-------|-------------|
| 0. Cleanup | Free disk from old checkpoints |
| 1. Download | Multi-source: FineWeb-Edu, Wikipedia, StackExchange, code (StarCoder), ArXiv |
| 2. Train classifiers | Train fasttext quality classifier on FineWeb-Edu scores + toxicity classifier on Jigsaw/Civil Comments |
| 3. Quality + toxicity scoring | Classifier-scored quality filtering (60% classifier, 40% heuristics) plus toxic content removal |
| 4. MinHash dedup | Near-duplicate removal across the entire corpus using locality-sensitive hashing |
| 5. Train tokenizer | 48k vocab SentencePiece BPE on 2 GB diverse sample with byte fallback |
| 6. Mix and shard | Domain-weighted mixing (45% web, 15% wiki, 15% code, 10% Q&A, etc.) then tokenization |
| 7. Pretrain | 200k steps, 500M parameters |
| 8. Synthetic instruct | Generate 50k instruction pairs using Claude API (optional) |
| 9. Build instruct shards | Multi-turn loss masking across all instruct sources |
| 10. Finetune | 30k steps with masked loss |
| 11. Test | Inference on benchmark prompts |

Run the full pipeline:

```bash
python scripts/run_1.1.py
```

Run individual stages:

```bash
python scripts/run_1.1.py --stage download
python scripts/run_1.1.py --stage classifiers
python scripts/run_1.1.py --stage quality
python scripts/run_1.1.py --stage dedup
python scripts/run_1.1.py --stage tokenizer
python scripts/run_1.1.py --stage shards
python scripts/run_1.1.py --stage pretrain
python scripts/run_1.1.py --stage synthetic
python scripts/run_1.1.py --stage instruct
python scripts/run_1.1.py --stage finetune
```

To resume pretraining from a checkpoint:

```bash
python -m src.train.train --config configs/pretrain_1.1.yaml --resume checkpoints/1.1_step_50000.pt
```

### Synthetic Instruction Data (optional)

The pipeline can generate high-quality instruction-response pairs using the Claude API. This has the highest impact on instruction following quality for small models.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/generate_synthetic.py --n-samples 50000
```

---

## Infrastructure

- **Data processing** (`src/data/`): quality scoring, MinHash dedup, domain mixing, streaming shard datasets
- **Model** (`src/model/`): transformer with GQA, SwiGLU, RoPE, RMSNorm, KV cache
- **Training** (`src/train/`): gradient accumulation, mixed precision, cosine LR, checkpointing
- **Inference** (`src/inference/`): autoregressive generation with KV caching, temperature/top-k sampling
- **Evaluation** (`src/eval/`): perplexity, math benchmarks, code benchmarks
- **Web UI** (`web/`): FastAPI backend with model management and switching

---

## Running

```bash
pip install -r requirements.txt
python run.py
```

Opens the web UI at `http://localhost:8000`. Available models are detected automatically from the checkpoints directory.

---

MIT License.
