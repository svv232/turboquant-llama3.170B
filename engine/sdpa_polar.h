#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace turboquant {

// Parameters for fused scaled dot-product attention with PolarQuant KV cache
struct SdpaPolarParams {
  int head_dim;       // 128 for Llama 3.1 70B
  int num_q_heads;    // 64 for Llama 3.1 70B
  int num_kv_heads;   // 8 for Llama 3.1 70B (GQA ratio = 8)
  int bits;           // PolarQuant quantization bits (5 = sweet spot)
  float attn_scale;   // 1/sqrt(head_dim) = 0.0884 for Llama
};

// Fused scaled dot-product attention operating directly on PolarQuant-encoded KV cache.
// Decodes PolarQuant angles+norms inline during the attention computation —
// no separate dequantize pass needed.
//
// Q:        [B, Hq, Sq, D]           float16
// K_angles: [B, Hkv, Skv, D-1]      uint8  (one byte per quantized angle)
// K_norms:  [B, Hkv, Skv, 1]        float32 (per-vector norm)
// V_angles: [B, Hkv, Skv, D-1]      uint8
// V_norms:  [B, Hkv, Skv, 1]        float32
//
// Returns:  [B, Hq, Sq, D]           float16
mx::array sdpa_polar(
    const mx::array& Q,
    const mx::array& K_angles, const mx::array& K_norms,
    const mx::array& V_angles, const mx::array& V_norms,
    const SdpaPolarParams& params,
    mx::StreamOrDevice s = {});

// ---- Primitive ----

class SdpaPolar : public mx::Primitive {
 public:
  explicit SdpaPolar(mx::Stream stream, SdpaPolarParams params)
      : mx::Primitive(stream), params_(params) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "SdpaPolar"; }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  SdpaPolarParams params_;
};

}  // namespace turboquant
