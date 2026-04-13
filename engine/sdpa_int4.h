#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace turboquant {

// Parameters for fused scaled dot-product attention with int4 KV cache
struct SdpaInt4Params {
  int head_dim;       // 128 for Llama 3.1 70B
  int num_q_heads;    // 64 for Llama 3.1 70B
  int num_kv_heads;   // 8 for Llama 3.1 70B (GQA ratio = 8)
  int group_size;     // int4 quantization group size (32 or 64)
  float attn_scale;   // 1/sqrt(head_dim) = 0.0884 for Llama
};

// Fused scaled dot-product attention operating directly on int4-quantized KV cache.
// No dequantize-then-attend — everything is fused in a single Metal kernel.
//
// Q:        [B, Hq, Sq, D]           float16
// K_quant:  [B, Hkv, Skv, D/2]      uint8  (packed int4, two values per byte)
// K_scales: [B, Hkv, Skv, D/group]  float16
// K_zeros:  [B, Hkv, Skv, D/group]  float16
// V_quant:  [B, Hkv, Skv, D/2]      uint8
// V_scales: [B, Hkv, Skv, D/group]  float16
// V_zeros:  [B, Hkv, Skv, D/group]  float16
//
// Returns:  [B, Hq, Sq, D]           float16
mx::array sdpa_int4(
    const mx::array& Q,
    const mx::array& K_quant, const mx::array& K_scales, const mx::array& K_zeros,
    const mx::array& V_quant, const mx::array& V_scales, const mx::array& V_zeros,
    const SdpaInt4Params& params,
    mx::StreamOrDevice s = {});

// ---- Primitive ----

class SdpaInt4 : public mx::Primitive {
 public:
  explicit SdpaInt4(mx::Stream stream, SdpaInt4Params params)
      : mx::Primitive(stream), params_(params) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "SdpaInt4"; }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  SdpaInt4Params params_;
};

// ---- Split-K Primitives ----
// Phase 1: compute partial attention over KV chunks
class SdpaInt4Partial : public mx::Primitive {
 public:
  explicit SdpaInt4Partial(mx::Stream stream, SdpaInt4Params params, int num_chunks, int chunk_size)
      : mx::Primitive(stream), params_(params), num_chunks_(num_chunks), chunk_size_(chunk_size) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  const char* name() const override { return "SdpaInt4Partial"; }
  bool is_equivalent(const mx::Primitive& other) const override;

  SdpaInt4Params params_;
  int num_chunks_;
  int chunk_size_;
};

// Phase 2: reduce partial results into final output
class SdpaInt4Reduce : public mx::Primitive {
 public:
  explicit SdpaInt4Reduce(mx::Stream stream, SdpaInt4Params params, int num_chunks)
      : mx::Primitive(stream), params_(params), num_chunks_(num_chunks) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  const char* name() const override { return "SdpaInt4Reduce"; }
  bool is_equivalent(const mx::Primitive& other) const override;

  SdpaInt4Params params_;
  int num_chunks_;
};

}  // namespace turboquant
