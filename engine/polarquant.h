#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace turboquant {

// PolarQuant encode: FP16 KV vectors → quantized polar representation
// Input: [batch, kv_heads, seq_len, head_dim] float16/float32
// Outputs:
//   angles: [batch, kv_heads, seq_len, head_dim/2 - 1] uint8 (4-bit packed pairs)
//   norms:  [batch, kv_heads, seq_len, 1] float16 (per-vector norm)
//
// PolarQuant recursively decomposes pairs of elements into (r, θ):
//   (x0, x1) → r = sqrt(x0² + x1²), θ = atan2(x1, x0)
// Then θ is quantized to 4 bits (16 levels over [-π, π))
// The radii from one level feed into the next recursion level.
// Final output: quantized angles at each level + the root norm.

struct PolarQuantParams {
  int head_dim;       // 128 for Llama 3.1
  int num_levels;     // log2(head_dim) = 7 recursion levels
  int bits;           // 4 bits per angle
};

// Encode FP16 vectors to PolarQuant compressed form
// Returns {angles_packed, norms}
std::vector<mx::array> polarquant_encode(
    const mx::array& input,  // [B, H, S, D] float16
    const PolarQuantParams& params,
    mx::StreamOrDevice s = {});

// Decode PolarQuant compressed form back to FP16
// Returns reconstructed [B, H, S, D] float16
mx::array polarquant_decode(
    const mx::array& angles,  // [B, H, S, total_angles] uint8
    const mx::array& norms,   // [B, H, S, 1] float16
    const PolarQuantParams& params,
    mx::StreamOrDevice s = {});

// ---- Primitives ----

class PolarQuantEncode : public mx::Primitive {
 public:
  explicit PolarQuantEncode(mx::Stream stream, PolarQuantParams params)
      : mx::Primitive(stream), params_(params) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "PolarQuantEncode"; }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  PolarQuantParams params_;
};

class PolarQuantDecode : public mx::Primitive {
 public:
  explicit PolarQuantDecode(mx::Stream stream, PolarQuantParams params)
      : mx::Primitive(stream), params_(params) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "PolarQuantDecode"; }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  PolarQuantParams params_;
};

}  // namespace turboquant
