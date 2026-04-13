# TurboQuant for Llama 3.1 70B — Fused Compressed Attention on Apple Silicon

## Goal
Run Llama 3.1 70B at 128K context on M1 Max 64GB with fused compressed-domain attention.
PolarQuant may actually work here (standard attn_scale = 1/sqrt(head_dim), unlike Gemma 4).

## CRITICAL: C++ and Metal Only
All inference code MUST be C++ and Metal. No Python in the hot path. No mlx-lm for inference.
Python is acceptable ONLY for tokenization (converting text to token IDs) and model download.

**Why:** In the Gemma 4 project, the C++ engine with custom Metal kernels achieved 37% faster
attention than MLX's built-in SDPA. mlx-lm's Python path uses mx.compile which is fast for
weight matmuls but prevents custom kernel integration. The whole point of this project is
custom Metal compute shaders — the inference engine must be C++ calling Metal directly via
MLX's Primitive system, not Python calling mlx-lm.

**Stack:** C++ → MLX C++ API → turboquant Primitives → Metal compute shaders (.metallib)

## Project Structure (follow this layout — proven in Gemma 4 project)

```
turboquant-llama/
├── CLAUDE.md                     # This file — project context and architecture
├── CMakeLists.txt                # Top-level build (links MLX, compiles .metallib)
├── lib/                          # TurboQuant library (reusable across models)
│   ├── turboquant.h              #   C++ API: sdpa_int4, PolarQuant primitives
│   ├── turboquant.cpp            #   MLX Primitive implementations + Metal dispatch
│   └── turboquant.metal          #   Metal compute shaders (fused SDPA + PolarQuant)
├── engine/                       # Llama 3.1 70B inference engine
│   ├── llama_multilayer.cpp      #   Full forward pass (80 layers, GQA, RoPE, SwiGLU)
│   ├── chat.py                   #   Single-prompt wrapper (tokenize → prompt.bin → engine)
│   ├── chat_repl.py              #   Streaming multi-turn REPL with memory + thinking
│   └── run_tests.sh              #   Regression tests
├── tests/
│   └── test_sdpa.cpp             #   Kernel correctness test (compare fused vs reference)
├── assets/                       #   Benchmark charts
├── vocab.bin                     #   BPE vocabulary (or use tokenizer via Python)
└── README.md                     #   Results and documentation
```

### Key files explained:
- **lib/turboquant.metal** — THE core artifact. Metal compute shaders that run on GPU.
  Contains `sdpa_int4_*` (fused attention from int4 KV) and PolarQuant kernels.
- **lib/turboquant.cpp** — Dispatches Metal kernels via MLX's Primitive system.
  Each kernel is a class (e.g. `FusedInt4SDPA`) with `eval_gpu()` that sets buffers
  and calls `encoder.dispatch_threadgroups()`.
- **lib/turboquant.h** — Public C++ API. The engine includes this and calls functions
  like `turboquant::sdpa_int4(queries, k_quant, ...)`.
- **engine/llama_multilayer.cpp** — The inference engine. Loads 4-bit weights from
  safetensors, runs the full transformer forward pass, uses `turboquant::sdpa_int4()`
  for attention during decode. Handles KV cache (int4 quantized), RoPE, GQA, sampling.
- **CMakeLists.txt** — Builds lib as a static library, compiles .metal → .metallib,
  links engine against lib + MLX.

## Target Hardware
- Apple M1 Max, 64GB unified memory
- ~7 GB/s SSD sequential read
- Metal GPU with 32 cores

## Llama 3.1 70B Architecture
- 80 layers, hidden_size=8192
- 64 attention heads, 8 KV heads (GQA=8)
- head_dim=128
- attn_scale = 1/sqrt(128) = 0.0884 (standard — PolarQuant should work)
- RoPE theta=500000
- SwiGLU activation, intermediate_size=28672
- 4-bit weights: ~35GB (fits in 64GB RAM)
- Vocab: 128256

## KV Cache at 128K Context
- FP16: 80 layers × 8 KV heads × 128K × 128 dim × 2 bytes × 2 (K+V) = ~33GB
- int4: ~5.2GB (6.4x compression)
- PolarQuant: potentially ~4.7GB (7.1x compression)
- Total with int4: 35GB weights + 5.2GB KV = ~40GB (fits 64GB)
- Total with FP16: 35GB weights + 33GB KV = 68GB (DOESN'T FIT)

This is exactly where TurboQuant matters — int4/PolarQuant KV makes 128K context possible.

## Key Insights from Gemma 4 Project (transferable)
1. Fused sdpa_int4 Metal kernel: 37% faster than dequantize+SDPA at medium context
2. Vectorized 4-bit unpacking: MLX qdot pattern (pre-scaled query × uint16 masks)
3. Online softmax in single pass — no intermediate score matrix
4. D=128 (Llama head_dim) needs BN=32, elems_per_lane=4 — simpler than Gemma's D=256/512
5. nanobind v2.10.2 pin for MLX 0.31.1 ABI compatibility
6. PolarQuant failed on Gemma 4 ONLY because attn_scale=1.0 — Llama uses 0.0884

## What to Try
1. Verify PolarQuant works on Llama (attn_scale=0.0884 should dampen angular error)
2. Port fused sdpa_int4 kernel for head_dim=128
3. If PolarQuant works: build fused PolarQuant SDPA (the full TurboQuant pipeline)
4. Test at 128K context — this is where it matters
5. Benchmark vs mlx-lm at long context

## Quick Start
```bash
# Build MLX
git clone --depth 1 https://github.com/ml-explore/mlx.git /tmp/mlx-source
mkdir /tmp/mlx-source/build && cd /tmp/mlx-source/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF
make -j8

# Build TurboQuant
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8

# Download Llama 3.1 70B 4-bit
python3.12 -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Meta-Llama-3.1-70B-Instruct-4bit')"
```

## Papers
- TurboQuant: arxiv.org/abs/2504.19874 (ICLR 2026) — QJL + PolarQuant
- QJL: arxiv.org/abs/2406.03482 (AAAI) — 1-bit quantized JL transform
- PolarQuant: arxiv.org/abs/2502.02617 (AISTATS 2026) — recursive polar quantization
