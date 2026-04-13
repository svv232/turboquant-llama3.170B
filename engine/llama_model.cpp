#include <cmath>
#include <iostream>
#include <stdexcept>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

#include "engine/llama_model.h"

namespace turboquant {

///////////////////////////////////////////////////////////////////////////////
// Construction
///////////////////////////////////////////////////////////////////////////////

LlamaModel::LlamaModel(LlamaWeights weights)
    : weights_(std::move(weights)) {
  auto& cfg = weights_.config;
  sdpa_params_.head_dim = cfg.head_dim;
  sdpa_params_.num_q_heads = cfg.num_attention_heads;
  sdpa_params_.num_kv_heads = cfg.num_key_value_heads;
  sdpa_params_.group_size = kv_group_size_;
  sdpa_params_.attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
}

void LlamaModel::enable_qjl(int sketch_dim) {
  use_qjl_ = true;
  qjl_sketch_dim_ = sketch_dim;
  qjl_proj_ = generate_projection_matrix(sketch_dim, weights_.config.head_dim, 42);
  mx::eval(qjl_proj_);
  std::cerr << "[model] QJL enabled: m=" << sketch_dim << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
// Quantized linear
///////////////////////////////////////////////////////////////////////////////

mx::array LlamaModel::qlinear(
    const mx::array& x,
    const QuantizedWeight& qw,
    mx::StreamOrDevice s) {
  // MLX quantized_matmul: x @ dequant(w).T
  // x: [..., in_features], w: quantized [out_features, in_features]
  std::optional<mx::array> biases_opt;
  if (qw.biases.size() > 0) {
    biases_opt = qw.biases;
  }
  return mx::quantized_matmul(
      x,
      qw.weight,
      qw.scales,
      biases_opt,
      /*transpose=*/true,
      weights_.config.quantize_group_size,
      weights_.config.quantize_bits,
      /*mode=*/"affine",
      s);
}

///////////////////////////////////////////////////////////////////////////////
// KV quantization (float16 -> int4 asymmetric per-group)
///////////////////////////////////////////////////////////////////////////////

std::tuple<mx::array, mx::array, mx::array> LlamaModel::quantize_kv(
    const mx::array& x,  // [B, H, S, D] float16
    mx::StreamOrDevice s) {
  // Reshape to expose groups: [B, H, S, n_groups, group_size]
  int B = x.shape(0);
  int H = x.shape(1);
  int S = x.shape(2);
  int D = x.shape(3);
  int n_groups = D / kv_group_size_;

  auto reshaped = mx::reshape(x, {B * H * S, n_groups, kv_group_size_}, s);

  // Per-group min and max for asymmetric quantization
  auto gmin = mx::min(reshaped, /*axis=*/2, /*keepdims=*/true, s);
  auto gmax = mx::max(reshaped, /*axis=*/2, /*keepdims=*/true, s);

  // Asymmetric quantization with (q - zero) * scale convention:
  // scale = (max - min) / 15
  // zero_point = min / scale  (so that dequant(0) = min)
  // q = round(clamp(x / scale - zero_point, 0, 15))
  // dequant: x = (q - zero_point) * scale

  auto scales = mx::divide(mx::subtract(gmax, gmin, s),
                            mx::array(15.0f, mx::float16), s);
  scales = mx::maximum(scales, mx::array(1e-8f, mx::float16), s);

  // zero_point = -min / scale (so that (0 - zp) * scale = min → zp = -min/scale)
  // Actually: we want dequant(q) = (q - zp) * scale
  // At q=0: (0 - zp) * scale = min → zp = -min/scale
  auto zeros = mx::divide(mx::negative(gmin, s), scales, s);

  // Quantize: q = round(clamp(x / scale + zp_offset, 0, 15))
  // where x = (q - zp) * scale → q = x/scale + zp
  auto normalized = mx::add(
      mx::divide(reshaped, scales, s), zeros, s);
  normalized = mx::clip(normalized, mx::array(0.0f), mx::array(15.0f), s);
  auto quantized = mx::astype(mx::round(normalized, s), mx::uint8, s);

  // Pack two 4-bit values into one uint8: low | (high << 4)
  // quantized shape: [B*H*S, n_groups, group_size]
  // Pack along the group_size dim (pairs of consecutive elements)
  auto even = mx::slice(quantized, {0, 0, 0}, {B * H * S, n_groups, kv_group_size_}, {1, 1, 2}, s);
  auto odd = mx::slice(quantized, {0, 0, 1}, {B * H * S, n_groups, kv_group_size_}, {1, 1, 2}, s);
  auto packed = mx::bitwise_or(even, mx::left_shift(odd, mx::array(4, mx::uint8), s), s);

  // Reshape back to [B, H, S, ...]
  packed = mx::reshape(packed, {B, H, S, D / 2}, s);
  scales = mx::reshape(mx::squeeze(scales, /*axis=*/2, s), {B, H, S, n_groups}, s);
  zeros = mx::reshape(mx::squeeze(zeros, /*axis=*/2, s), {B, H, S, n_groups}, s);

  return {packed, mx::astype(scales, mx::float16, s), mx::astype(zeros, mx::float16, s)};
}

///////////////////////////////////////////////////////////////////////////////
// Single layer forward
///////////////////////////////////////////////////////////////////////////////

mx::array LlamaModel::layer_forward(
    const mx::array& x,       // [B, S, hidden_size]
    int layer_idx,
    int rope_offset,
    KVCacheEntry* kv_entry,
    mx::StreamOrDevice s) {
  auto& cfg = weights_.config;
  auto& lw = weights_.layers[layer_idx];
  int B = x.shape(0);
  int S = x.shape(1);
  int H = cfg.num_attention_heads;
  int Hkv = cfg.num_key_value_heads;
  int D = cfg.head_dim;

  // Pre-attention RMSNorm
  auto normed = mx::fast::rms_norm(x, lw.input_layernorm, cfg.rms_norm_eps, s);

  // QKV projections
  auto q = qlinear(normed, lw.q_proj, s);   // [B, S, H*D]
  auto k = qlinear(normed, lw.k_proj, s);   // [B, S, Hkv*D]
  auto v = qlinear(normed, lw.v_proj, s);   // [B, S, Hkv*D]

  // Reshape to [B, S, num_heads, head_dim] then transpose to [B, num_heads, S, head_dim]
  q = mx::transpose(mx::reshape(q, {B, S, H, D}, s), {0, 2, 1, 3}, s);
  k = mx::transpose(mx::reshape(k, {B, S, Hkv, D}, s), {0, 2, 1, 3}, s);
  v = mx::transpose(mx::reshape(v, {B, S, Hkv, D}, s), {0, 2, 1, 3}, s);

  // RoPE — applied before KV cache
  q = mx::fast::rope(q, D, /*traditional=*/false, cfg.rope_theta, 1.0f,
                      rope_offset, std::nullopt, s);
  k = mx::fast::rope(k, D, /*traditional=*/false, cfg.rope_theta, 1.0f,
                      rope_offset, std::nullopt, s);

  mx::array attn_out(0.0f);

  if (D == 128 && !fp16_kv_mode_) {
    // Hybrid attention: fast SDPA for prefill, fused sdpa_int4 for decode
    bool is_prefill = (S > 1);

    if (is_prefill) {
      // PREFILL: use MLX's optimized SDPA with FP16 KV (faster for Sq>1)
      attn_out = mx::fast::scaled_dot_product_attention(
          q, k, v, sdpa_params_.attn_scale, /*mask_mode=*/"", /*mask_arrs=*/{},
          /*sinks=*/std::nullopt, s);

      // Store K/V in cache for future decode steps
      if (kv_entry) {
        auto [v_q, v_s, v_z] = quantize_kv(v, s);
        kv_entry->v_quant = v_q;
        kv_entry->v_scales = v_s;
        kv_entry->v_zeros = v_z;

        bool use_qjl_this_layer_pf = use_qjl_ && (layer_idx < 20);
        if (use_qjl_this_layer_pf) {
          // QJL mode: store K as 1-bit sketches (early layers only)
          auto [sketches, norms] = qjl_encode(k, qjl_proj_, s);
          kv_entry->k_sketches = sketches;
          kv_entry->k_norms_qjl = norms;
        } else {
          // Int4 mode: store K as int4
          auto [k_q, k_s, k_z] = quantize_kv(k, s);
          kv_entry->k_quant = k_q;
          kv_entry->k_scales = k_s;
          kv_entry->k_zeros = k_z;
        }
        kv_entry->seq_len = v_q.shape(2);
      }
    } else {
      // DECODE (Sq=1): use fused kernel with cached KV
      auto [v_q, v_s, v_z] = quantize_kv(v, s);

      // Use QJL only for early layers (0-19) where correlation > 0.93
      bool use_qjl_this_layer = use_qjl_ && (layer_idx < 20);

      if (use_qjl_this_layer) {
        // QJL decode: encode new K as sketch, concat with cached sketches
        auto [k_sk, k_nm] = qjl_encode(k, qjl_proj_, s);
        // Also encode Q on the fly
        auto [q_sk, q_nm] = qjl_encode(q, qjl_proj_, s);

        if (kv_entry && kv_entry->seq_len > 0) {
          k_sk = mx::concatenate({kv_entry->k_sketches, k_sk}, 2, s);
          k_nm = mx::concatenate({kv_entry->k_norms_qjl, k_nm}, 2, s);
          v_q = mx::concatenate({kv_entry->v_quant, v_q}, 2, s);
          v_s = mx::concatenate({kv_entry->v_scales, v_s}, 2, s);
          v_z = mx::concatenate({kv_entry->v_zeros, v_z}, 2, s);
        }
        if (kv_entry) {
          kv_entry->k_sketches = k_sk;
          kv_entry->k_norms_qjl = k_nm;
          kv_entry->v_quant = v_q;
          kv_entry->v_scales = v_s;
          kv_entry->v_zeros = v_z;
          kv_entry->seq_len = k_sk.shape(2);
        }

        SdpaQJLParams qjl_params{D, qjl_sketch_dim_, H, Hkv, kv_group_size_, sdpa_params_.attn_scale};
        attn_out = sdpa_qjl(q_sk, q_nm, k_sk, k_nm, v_q, v_s, v_z, qjl_params, s);
      } else {
        // Int4 decode
        auto [k_q, k_s, k_z] = quantize_kv(k, s);

        if (kv_entry && kv_entry->seq_len > 0) {
          k_q = mx::concatenate({kv_entry->k_quant, k_q}, 2, s);
          k_s = mx::concatenate({kv_entry->k_scales, k_s}, 2, s);
          k_z = mx::concatenate({kv_entry->k_zeros, k_z}, 2, s);
          v_q = mx::concatenate({kv_entry->v_quant, v_q}, 2, s);
          v_s = mx::concatenate({kv_entry->v_scales, v_s}, 2, s);
          v_z = mx::concatenate({kv_entry->v_zeros, v_z}, 2, s);
        }
        if (kv_entry) {
          kv_entry->k_quant = k_q;
          kv_entry->k_scales = k_s;
          kv_entry->k_zeros = k_z;
          kv_entry->v_quant = v_q;
          kv_entry->v_scales = v_s;
          kv_entry->v_zeros = v_z;
          kv_entry->seq_len = k_q.shape(2);
        }
        attn_out = sdpa_int4(q, k_q, k_s, k_z, v_q, v_s, v_z, sdpa_params_, s);
      }
    }
  } else {
    // FP16 KV cache + MLX fast SDPA (for head_dim != 128 or fp16_kv_mode)
    if (kv_entry) {
      if (kv_entry->seq_len > 0) {
        k = mx::concatenate({kv_entry->k_quant, k}, 2, s);
        v = mx::concatenate({kv_entry->v_quant, v}, 2, s);
      }
      kv_entry->k_quant = k;
      kv_entry->v_quant = v;
      kv_entry->seq_len = k.shape(2);
    }
    // Use MLX's optimized SDPA (handles GQA natively)
    attn_out = mx::fast::scaled_dot_product_attention(
        q, k, v, sdpa_params_.attn_scale, /*mask_mode=*/"", /*mask_arrs=*/{},
        /*sinks=*/std::nullopt, s);
  }

  // Reshape back: [B, H, S, D] -> [B, S, H*D]
  attn_out = mx::reshape(
      mx::transpose(attn_out, {0, 2, 1, 3}, s),
      {B, S, H * D}, s);

  // Output projection
  auto attn_proj = qlinear(attn_out, lw.o_proj, s);

  // Residual connection
  auto h = mx::add(x, attn_proj, s);

  // Post-attention RMSNorm
  auto normed2 = mx::fast::rms_norm(h, lw.post_attention_layernorm, cfg.rms_norm_eps, s);

  // SwiGLU MLP: gate_proj and up_proj applied in parallel, then down_proj
  auto gate = qlinear(normed2, lw.gate_proj, s);
  auto up = qlinear(normed2, lw.up_proj, s);

  // SiLU(gate) * up
  auto silu_gate = mx::multiply(gate, mx::sigmoid(gate, s), s);  // SiLU = x * sigmoid(x)
  auto mlp_out = mx::multiply(silu_gate, up, s);
  mlp_out = qlinear(mlp_out, lw.down_proj, s);

  // Residual connection
  return mx::add(h, mlp_out, s);
}

///////////////////////////////////////////////////////////////////////////////
// Full forward pass
///////////////////////////////////////////////////////////////////////////////

mx::array LlamaModel::forward(
    const mx::array& tokens,  // [B, S] int32
    KVCache* cache,
    mx::StreamOrDevice s) {
  auto& cfg = weights_.config;
  int B = tokens.shape(0);
  int S = tokens.shape(1);

  // Initialize cache if needed
  if (cache && cache->layers.empty()) {
    cache->layers.resize(cfg.num_hidden_layers);
  }

  // Compute RoPE offset from cache
  int rope_offset = 0;
  if (cache && !cache->layers.empty() && cache->layers[0].seq_len > 0) {
    rope_offset = cache->layers[0].seq_len;
  }

  // Embedding lookup
  mx::array h(0.0f);
  if (weights_.embed_is_quantized) {
    // Dequantize embedding table rows on the fly
    // Dequant the full table (expensive but simple — could optimize with gather_qmm later)
    if (!embed_dequantized_) {
      auto& eq = weights_.embed_tokens_q;
      embed_cache_ = mx::dequantize(
          eq.weight, eq.scales, eq.biases,
          cfg.quantize_group_size, cfg.quantize_bits, "affine", s);
      mx::eval(embed_cache_);
      embed_dequantized_ = true;
    }
    h = mx::take(embed_cache_, tokens, 0, s);
  } else {
    h = mx::take(weights_.embed_tokens, tokens, 0, s);
  }

  // Transformer layers
  for (int i = 0; i < cfg.num_hidden_layers; i++) {
    KVCacheEntry* kv = cache ? &cache->layers[i] : nullptr;
    h = layer_forward(h, i, rope_offset, kv, s);
  }

  // Final RMSNorm
  h = mx::fast::rms_norm(h, weights_.model_norm, cfg.rms_norm_eps, s);

  // LM head projection
  mx::array logits(0.0f);
  if (weights_.tie_word_embeddings) {
    if (weights_.embed_is_quantized) {
      // Tied quantized weights: use quantized_matmul
      logits = qlinear(h, weights_.embed_tokens_q, s);
    } else {
      // Tied fp16 weights: h @ embed_tokens.T
      logits = mx::matmul(h, mx::transpose(weights_.embed_tokens, s), s);
    }
  } else {
    logits = qlinear(h, weights_.lm_head, s);
  }

  return logits;  // [B, S, vocab_size]
}

///////////////////////////////////////////////////////////////////////////////
// Greedy decode helper
///////////////////////////////////////////////////////////////////////////////

int LlamaModel::argmax_last(const mx::array& logits) {
  // logits: [B, S, V] — take last position of first batch
  auto last = mx::slice(logits, {0, logits.shape(1) - 1, 0},
                         {1, logits.shape(1), logits.shape(2)});
  last = mx::reshape(last, {logits.shape(2)});
  auto idx = mx::argmax(last);
  mx::eval(idx);
  return idx.item<int>();
}

}  // namespace turboquant
