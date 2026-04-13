#pragma once

#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

#include "engine/llama_loader.h"
#include "engine/sdpa_int4.h"
#include "engine/sdpa_qjl.h"
#include "engine/qjl.h"

namespace mx = mlx::core;

namespace turboquant {

// Int4-quantized KV cache entry for one layer
struct KVCacheEntry {
  mx::array k_quant{0.0f};
  mx::array k_scales{0.0f};
  mx::array k_zeros{0.0f};
  mx::array v_quant{0.0f};
  mx::array v_scales{0.0f};
  mx::array v_zeros{0.0f};
  int seq_len = 0;
  bool preallocated = false;
  int max_seq_len = 0;
  // QJL mode: K stored as 1-bit sketches
  mx::array k_sketches{0.0f};  // [B, H, S, m/32] uint32
  mx::array k_norms_qjl{0.0f}; // [B, H, S] float16
};

// KV cache for all layers
struct KVCache {
  std::vector<KVCacheEntry> layers;
  int max_seq_len = 0;
};

// Llama 3.1 70B model — minimal inference pipeline
class LlamaModel {
 public:
  explicit LlamaModel(LlamaWeights weights);

  // Run a forward pass on input token IDs.
  // tokens: [batch_size, seq_len] int32
  // Returns logits: [batch_size, seq_len, vocab_size] float16
  //
  // If cache is provided, appends new KV entries and uses cached KV for
  // attention. offset is the position offset for RoPE (= cache seq_len).
  mx::array forward(
      const mx::array& tokens,
      KVCache* cache = nullptr,
      mx::StreamOrDevice s = {});

  const LlamaConfig& config() const { return weights_.config; }

  // Greedy-decode next token from logits (takes last position)
  static int argmax_last(const mx::array& logits);

 private:
  // Quantized linear: x @ dequant(w).T
  mx::array qlinear(
      const mx::array& x,
      const QuantizedWeight& qw,
      mx::StreamOrDevice s);

  // One transformer layer forward
  mx::array layer_forward(
      const mx::array& x,        // [B, S, hidden]
      int layer_idx,
      int rope_offset,
      KVCacheEntry* kv_entry,
      mx::StreamOrDevice s);

  // Quantize a KV tensor from float16 to int4 (asymmetric per-group)
  // Returns {quant, scales, zeros}
  std::tuple<mx::array, mx::array, mx::array> quantize_kv(
      const mx::array& x,  // [B, H, S, D] float16
      mx::StreamOrDevice s);

  LlamaWeights weights_;
  SdpaInt4Params sdpa_params_;
  int kv_group_size_ = 32;
  mutable mx::array embed_cache_{0.0f};
  mutable bool embed_dequantized_ = false;

 public:
  // QJL mode: use QJL 1-bit sketches for K + int4 for V
  bool use_qjl_ = false;
  int qjl_sketch_dim_ = 512;  // m
  mx::array qjl_proj_{0.0f};  // [m, D] projection matrix
  void enable_qjl(int sketch_dim = 512);
};

}  // namespace turboquant
