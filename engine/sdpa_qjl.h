#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace turboquant {

// Parameters for fused QJL SDPA with int4 V cache
struct SdpaQJLParams {
  int head_dim;        // 128
  int sketch_dim;      // m (256, 384, 512, 768)
  int num_q_heads;     // 64
  int num_kv_heads;    // 8
  int v_group_size;    // 32 (int4 V quantization group size)
  float attn_scale;    // 1/sqrt(128) = 0.0884
};

// Fused QJL-scored scaled dot-product attention with int4 V.
//
// Replaces full Q*K dot products with QJL sketch XNOR+popcount scoring,
// then applies online softmax and accumulates int4-dequantized V.
//
// Q_sketches:  [B, Hq, Sq, m/32]      uint32 (packed 1-bit sketches)
// Q_norms:     [B, Hq, Sq]            float16
// K_sketches:  [B, Hkv, Skv, m/32]    uint32 (packed 1-bit sketches)
// K_norms:     [B, Hkv, Skv]          float16
// V_quant:     [B, Hkv, Skv, D/2]     uint8 (packed int4)
// V_scales:    [B, Hkv, Skv, D/group] float16
// V_zeros:     [B, Hkv, Skv, D/group] float16
//
// Returns:     [B, Hq, Sq, D]          float16
mx::array sdpa_qjl(
    const mx::array& Q_sketches,
    const mx::array& Q_norms,
    const mx::array& K_sketches,
    const mx::array& K_norms,
    const mx::array& V_quant,
    const mx::array& V_scales,
    const mx::array& V_zeros,
    const SdpaQJLParams& params,
    mx::StreamOrDevice s = {});

// ---- Primitive: single-pass ----

class SdpaQJL : public mx::Primitive {
 public:
  explicit SdpaQJL(mx::Stream stream, SdpaQJLParams params)
      : mx::Primitive(stream), params_(params) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "SdpaQJL"; }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  SdpaQJLParams params_;
};

// ---- Split-K Phase 1: partial attention over KV chunks ----

class SdpaQJLPartial : public mx::Primitive {
 public:
  explicit SdpaQJLPartial(mx::Stream stream, SdpaQJLParams params,
                           int num_chunks, int chunk_size)
      : mx::Primitive(stream), params_(params),
        num_chunks_(num_chunks), chunk_size_(chunk_size) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  const char* name() const override { return "SdpaQJLPartial"; }
  bool is_equivalent(const mx::Primitive& other) const override;

  SdpaQJLParams params_;
  int num_chunks_;
  int chunk_size_;
};

// ---- Split-K Phase 2: reduce partials ----

class SdpaQJLReduce : public mx::Primitive {
 public:
  explicit SdpaQJLReduce(mx::Stream stream, SdpaQJLParams params, int num_chunks)
      : mx::Primitive(stream), params_(params), num_chunks_(num_chunks) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  const char* name() const override { return "SdpaQJLReduce"; }
  bool is_equivalent(const mx::Primitive& other) const override;

  SdpaQJLParams params_;
  int num_chunks_;
};

}  // namespace turboquant
