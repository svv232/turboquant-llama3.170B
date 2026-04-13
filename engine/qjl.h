#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace turboquant {

// QJL: Quantized Johnson-Lindenstrauss transform for key compression
//
// Given a key vector k ∈ R^d, QJL computes:
//   sketch(k) = sign(R · k)  ∈ {0,1}^m
// where R ∈ {-1,+1}^{m×d} is a random projection matrix.
//
// Inner product estimation:
//   <q, k> ≈ (||q|| · ||k|| / m) · (2 · popcount(XNOR(sketch_q, sketch_k)) - m)
//
// Storage per key: m bits (packed as uint32) + 1 float16 (norm)
// At m=384: 48 bytes sketch + 2 bytes norm = 50 bytes (vs 256 bytes FP16 = 5.12x)

struct QJLParams {
  int head_dim;       // d = 128
  int sketch_dim;     // m = number of random projections (128, 256, 384, 512)
  int num_q_heads;    // 64
  int num_kv_heads;   // 8
  float attn_scale;   // 1/sqrt(128)
};

// Generate a random {-1, +1} projection matrix R
// Shape: [sketch_dim, head_dim], stored as int8 (-1 or +1)
// Uses a fixed seed for reproducibility across encode/decode
mx::array generate_projection_matrix(int sketch_dim, int head_dim, int seed = 42,
                                      mx::StreamOrDevice s = {});

// QJL encode: compute 1-bit sketch of key vectors
// Input: keys [B, H, S, D] float16
// Returns: {sketches [B, H, S, m/32] uint32 (packed bits), norms [B, H, S] float16}
std::pair<mx::array, mx::array> qjl_encode(
    const mx::array& keys,
    const mx::array& proj_matrix,  // [m, D] int8
    mx::StreamOrDevice s = {});

// QJL inner product estimation
// Q: [B, Hq, Sq, D] float16
// K_sketches: [B, Hkv, Skv, m/32] uint32
// K_norms: [B, Hkv, Skv] float16
// Returns: estimated Q·K^T scores [B, Hq, Sq, Skv] float32
mx::array qjl_scores(
    const mx::array& Q,
    const mx::array& K_sketches,
    const mx::array& K_norms,
    const mx::array& proj_matrix,
    const QJLParams& params,
    mx::StreamOrDevice s = {});

}  // namespace turboquant
