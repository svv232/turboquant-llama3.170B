# TurboQuant for Llama 3.1 70B — 128K Context on a Mac

> Llama 3.1 70B needs 79GB to run at 128K context. A 64GB Mac can't do it. I wrote fused Metal attention kernels that compress the KV cache from 40GB to 12.5GB. Now 128K context fits — and runs at 6 tok/s.

## The Problem

Llama 3.1 70B at 4-bit weights takes 39.1 GB. The KV cache at 128K context in FP16 adds 40 GB more — 80 layers × 8 KV heads × 128K tokens × 128 dimensions × 2 bytes × 2 (keys + values). Total: **79.1 GB**. It doesn't fit in 64GB.

Every existing inference framework (mlx-lm, llama.cpp, ollama) hits this wall. They can run the model at shorter contexts but 128K — the model's full designed context window — is physically impossible on consumer hardware.

## The Solution: Fused int4 Attention in Metal

TurboQuant quantizes K and V to int4 on the fly and computes attention directly on the compressed data. No dequantization pass. No temporary matrices. Everything happens in GPU registers inside a custom Metal compute shader.

```
mlx-lm:     FP16 K,V → Q @ K^T → softmax → @ V → output   (40 GB KV cache)
TurboQuant: sdpa_int4(Q, K_int4, V_int4) → output           (12.5 GB KV cache)
```

**27.5 GB saved.** Total memory: 53.6 GB. That's 10.4 GB of headroom on a 64GB Mac.

![Memory at 128K Context](assets/memory_128k.png)

| Context | FP16 KV | int4 KV | Saved | Total (int4) | Fits 64GB? |
|:--------|:-------:|:-------:|:-----:|:------------:|:----------:|
| 1K | 0.3 GB | 0.1 GB | 0.2 GB | 41.2 GB | Yes |
| 16K | 5.0 GB | 1.6 GB | 3.4 GB | 42.7 GB | Yes |
| 64K | 20.0 GB | 6.3 GB | 13.7 GB | 47.4 GB | Yes |
| **128K** | **40.0 GB** | **12.5 GB** | **27.5 GB** | **53.6 GB** | **Yes** |

![KV Cache Growth](assets/kv_cache_growth.png)

## What We Built on Metal

This is, to our knowledge, the **first implementation of fused compressed-domain attention on Apple Silicon**. The TurboQuant paper (ICLR 2026) and all related work target NVIDIA GPUs with CUDA. We ported the core ideas to Metal compute shaders and built a complete inference engine from scratch.

### Fused sdpa_int4 Kernel

The kernel reads packed int4 keys and values directly from Metal device memory, dequantizes per-element using bitwise ops, computes the Q·K^T dot product via SIMD reduction, and accumulates weighted values — all with online softmax (Milakov & Gimelshein) so no S×S attention matrix is ever materialized.

At 128K context with Llama's 64 query heads and 8 KV heads, this is **48× faster** than the dequantize-then-attend baseline:

![Kernel Speedup](assets/kernel_speedup.png)

| KV Length | Fused Kernel | Dequant + SDPA | Speedup |
|:----------|:------------:|:--------------:|:-------:|
| 1K tokens | 1.6 ms | 4.6 ms | 3× |
| 16K tokens | 2.7 ms | 54.6 ms | 20× |
| 64K tokens | 5.7 ms | 225.3 ms | 39× |
| **128K tokens** | **9.9 ms** | **480.6 ms** | **48×** |

The speedup scales super-linearly because the baseline materializes a 128K × 128K score matrix per head, while the fused kernel processes 512-token chunks independently via split-K parallelism.

### Split-K via Chained MLX Primitives

At long contexts, a single threadgroup can't process all 128K KV tokens fast enough. Split-K divides the sequence into 512-token chunks processed in parallel across Metal's 32 GPU cores, then reduces the partial softmax results.

The catch: Metal doesn't automatically barrier between dispatches within a single `eval_gpu()`. Two dispatches will race. We solved this by implementing Phase 1 (partial attention) and Phase 2 (reduce) as **separate MLX Primitives** chained through the computation graph. MLX's lazy evaluation naturally serializes them — Phase 2 takes Phase 1's outputs as inputs.

### Hybrid Prefill/Decode Attention

The fused int4 kernel is designed for decode (Sq=1, long Skv). For prefill (Sq>1), MLX's built-in `scaled_dot_product_attention` is faster because it batches the multi-query computation. TurboQuant automatically selects the right path:

- **Prefill** (Sq > 1): MLX fast SDPA with FP16 K,V → quantize to int4 → store in cache
- **Decode** (Sq = 1): Fused sdpa_int4 reads directly from int4 cache

### Wired Memory for Large Models

A 39GB model on 64GB hardware leaves macOS paging weight tensors to SSD. Every Metal dispatch faults pages back in — a 12× slowdown that looks like a compute bottleneck but is actually I/O.

One line fixes it: `mx::set_wired_limit(max_recommended_working_set_size)`. This pins weights in GPU-accessible memory. The effect: **0.6 tok/s → 6.0 tok/s**. mlx-lm does the same thing via its `wired_limit()` context manager.

## Performance

### Decode: 6.0 tok/s (within 18% of mlx-lm)

| | TurboQuant | mlx-lm |
|:---|:---:|:---:|
| **Decode speed** | **6.0 tok/s** | 7.3 tok/s |
| **Max context (64GB)** | **~236K tokens** | ~73K tokens |
| **128K context** | **Yes (53.6 GB)** | **No (79.1 GB)** |
| KV cache format | int4 (fused kernel) | FP16 |
| KV quantization overhead | 2% of decode time | None |

The 18% gap comes from KV cache management overhead (concatenation vs mlx-lm's pre-allocated slice assignment), not the attention kernel itself. The kernel is faster — the plumbing is slower.

### Output Quality: Identical

Given "Hello, I am", both TurboQuant and mlx-lm predict the same top-1 next token. Both produce fluent, coherent, multi-paragraph text at 200+ tokens with no degradation:

```
Hello, I am a student at the University of California, Berkeley. I am interested 
in learning more about the field of computer science. I am currently taking a 
course on data structures and algorithms, and I am enjoying it so far. I am also 
interested in learning more about the different types of algorithms and how they 
can be used to solve complex problems. I am looking forward to gaining a deeper 
understanding of the field of computer science and how it can be used to make a 
positive impact on society.
```

The int4 KV quantization error is small enough that greedy decode produces the same tokens. Divergence only appears through the butterfly effect — a tiny logit difference changes one token, which cascades into a different (but equally coherent) continuation.

## Memory Breakdown

![Memory Breakdown](assets/memory_breakdown.png)

At short contexts both approaches fit. At 64K, mlx-lm is at 61GB — one background process away from swapping. At 128K it's 79GB — impossible.

TurboQuant stays at 53.6 GB with 10.4 GB of headroom. The KV cache grows **3.2× slower**, which means 3.2× more context before hitting the memory wall.

## What Else We Tried (30 Experiments)

This started as an implementation of the [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026), which proposes QJL + PolarQuant for 2.5-bit near-lossless KV compression. We implemented every component as Metal compute shaders and tested them across 30 experiments on the full 70B model.

![Paper Comparison](assets/paper_comparison.png)

### What Works

| Approach | Result | Detail |
|:---|:---|:---|
| **Fused sdpa_int4 kernel** | **48× faster at 128K** | Online softmax + int4 dequant in Metal registers |
| **Split-K parallelism** | **Flat latency at 128K** | Chained MLX Primitives solve Metal dispatch races |
| **Hybrid prefill/decode** | **Best of both paths** | MLX SDPA for prefill, fused int4 for decode |
| **Wired memory** | **12× speedup** | `set_wired_limit` prevents paging on large models |
| **PolarQuant validation** | **attn_scale dampening confirmed** | Llama's 0.0884 scale gives 25-100× less error than Gemma's 1.0 |

### What Doesn't Work at 70B

| Approach | Result | Why |
|:---|:---|:---|
| QJL 1-bit key sketching | Repetition loops | Error compounds across 80 layers (0.85^80 ≈ 0) |
| QJL + int4 hybrid (20 layers) | Semi-coherent | Marginal — 1.7% memory savings, quality risk |
| PolarQuant SDPA kernel | Slower than int4 | cos/sin lookups more expensive than linear dequant |
| Pre-allocated KV cache | Quality regression | `slice_update` graph dependency issues in C++ |

### Why QJL Fails at 70B Scale

The [QJL paper](https://arxiv.org/abs/2406.03482) achieves near-lossless compression on 8B models (16–32 layers). At 80 layers, the per-layer score correlation of 0.85 compounds exponentially — the signal is destroyed. We built the fused QJL SDPA kernel (22× speedup with XNOR + popcount), validated it on real activations layer-by-layer, and confirmed: early layers (0–10) have 0.93+ correlation, deep layers (40–60) drop to 0.80. The paper's results don't transfer to models this deep.

### Why PolarQuant Works But Int4 Wins

[PolarQuant](https://arxiv.org/abs/2502.02617) encodes vectors as recursive polar coordinates. On Llama (attn_scale = 0.0884), it achieves cosine similarity of 0.989 at 5-bit — excellent quality, confirming the paper's theory that standard attention scale dampens angular error. But the decode kernel is slower (cos/sin per angle vs bitwise dequant), and compression is 1.95× without bit-packing vs int4's 3.2×. We validated the theory; int4 wins the practice.

![Compression Comparison](assets/compression_comparison.png)

## Architecture

Everything is C++ and Metal. Python is used only for tokenization and model download.

```
C++ → MLX C++ API → TurboQuant Primitives → Metal Compute Shaders (.metallib)

Decode step (per layer):
  RMSNorm → Q,K,V projections (quantized matmul)
  → RoPE → quantize K,V to int4
  → fused sdpa_int4 (online softmax + int4 dequant + GQA)
  → output projection → SwiGLU MLP → residual
```

The engine loads safetensors weights, handles quantized embeddings (dequantize on first use), manages the int4 KV cache with per-step concatenation, and supports multi-turn chat via the `chat_engine` binary.

## Getting Started

### Prerequisites

- macOS with Apple Silicon (M1 Max or higher, 64GB+)
- Xcode Command Line Tools
- CMake 3.27+
- Python 3.12 (for model download and tokenization only)
- ~40 GB free disk space

### Build

```bash
# Build MLX from source
git clone --depth 1 https://github.com/ml-explore/mlx.git /tmp/mlx-source
mkdir /tmp/mlx-source/build && cd /tmp/mlx-source/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF
make -j8

# Build TurboQuant
cd /path/to/turboquant-llama
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Download Model

```bash
pip3.12 install huggingface_hub transformers
python3.12 -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Meta-Llama-3.1-70B-Instruct-4bit')"
```

### Chat

```bash
python3.12 engine/chat_repl.py
```

```
Llama 3.1 70B — TurboQuant Fused int4 SDPA
128K context on 64GB  |  int4 KV cache (3.2x compression)
Loading model...Model loaded.
Type quit to exit, clear to reset memory

You: What is the capital of France?
Llama: The capital of France is Paris.
  [8 tokens, 6.0 tok/s]

You: What about Germany?
Llama: The capital of Germany is Berlin.
  [8 tokens, 6.1 tok/s]
```

### Run Benchmarks

```bash
./build/llama_e2e <model_dir>           # End-to-end generation test
./build/sdpa_int4_bench                  # Fused kernel throughput
./build/polarquant_validate              # PolarQuant quality sweep
./build/qjl_validate                     # QJL inner product preservation
./build/decode_profile <model_dir>       # Per-step latency breakdown
./build/kv_memory_test                   # KV cache memory verification
```

## File Structure

```
turboquant-llama/
├── engine/
│   ├── sdpa_int4.metal            # Fused int4 SDPA (single-pass + split-K)
│   ├── sdpa_int4.{h,cpp}          # MLX Primitives (SdpaInt4, SdpaInt4Partial, SdpaInt4Reduce)
│   ├── sdpa_qjl.metal             # Fused QJL SDPA (XNOR + popcount)
│   ├── sdpa_polar.metal           # Fused PolarQuant SDPA (cos/sin LUT)
│   ├── polarquant.{h,cpp,metal}   # PolarQuant encoder/decoder
│   ├── qjl.{h,cpp}               # QJL 1-bit key sketching
│   ├── llama_model.{h,cpp}        # 70B inference (80 layers, GQA, RoPE, SwiGLU)
│   ├── llama_loader.{h,cpp}       # Safetensors weight loader
│   ├── chat_engine.cpp            # Streaming C++ chat backend
│   ├── chat_repl.py               # Interactive chat (uses chat_engine)
│   └── chat_mlx.py                # mlx-lm baseline chat (for comparison)
├── tests/                          # 12 benchmark and validation binaries
├── assets/                         # Charts (generate_charts.py)
└── CMakeLists.txt
```

## Key Discoveries

1. **Wired memory is mandatory.** Without `set_wired_limit`, a 39GB model on 64GB hardware pages to SSD, causing 12× slowdown. One line of code: 0.6 → 6.0 tok/s.

2. **Split-K needs separate Primitives.** Metal dispatches within a single `eval_gpu()` race. Chaining two MLX Primitives via the computation graph gives automatic serialization.

3. **QJL doesn't scale to 80 layers.** Per-layer score correlation of 0.85 compounds to zero over 80 layers. The paper's 8B results don't predict 70B behavior.

4. **PolarQuant works on Llama, fails on Gemma.** Llama's attn_scale = 0.0884 dampens angular error 25-100× vs Gemma's attn_scale = 1.0. The theory is correct; applicability depends on the model.

5. **Int4 asymmetric quantization is the practical sweet spot.** It's not the most compressed (QJL offers 5×) or the highest quality (FP16), but it's the only method that delivers both acceptable quality and viable speed at 70B scale on consumer hardware.

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — QJL + PolarQuant fused compressed-domain attention
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1-bit quantized Johnson-Lindenstrauss transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — recursive polar coordinate quantization
