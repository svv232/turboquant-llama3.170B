#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/array.h"

namespace mx = mlx::core;

namespace turboquant {

// Model configuration parsed from config.json
struct LlamaConfig {
  int hidden_size = 8192;
  int num_hidden_layers = 80;
  int num_attention_heads = 64;
  int num_key_value_heads = 8;
  int head_dim = 128;
  int intermediate_size = 28672;
  int vocab_size = 128256;
  float rope_theta = 500000.0f;
  float rms_norm_eps = 1e-5f;
  // Quantization params (from MLX 4-bit format)
  int quantize_group_size = 64;
  int quantize_bits = 4;
};

// Quantized linear weight: packed uint32 + scales + biases
struct QuantizedWeight {
  mx::array weight{0.0f};   // quantized packed (uint32)
  mx::array scales{0.0f};   // per-group scales (float16)
  mx::array biases{0.0f};   // per-group biases (float16)
};

// Weights for one transformer layer
struct LlamaLayerWeights {
  // Self-attention projections (quantized)
  QuantizedWeight q_proj;
  QuantizedWeight k_proj;
  QuantizedWeight v_proj;
  QuantizedWeight o_proj;

  // MLP projections (quantized)
  QuantizedWeight gate_proj;
  QuantizedWeight up_proj;
  QuantizedWeight down_proj;

  // RMSNorm weights (float16)
  mx::array input_layernorm{0.0f};
  mx::array post_attention_layernorm{0.0f};
};

// All model weights
struct LlamaWeights {
  LlamaConfig config;

  // Embedding and output head
  mx::array embed_tokens{0.0f};    // [vocab_size, hidden_size] fp16 (if not quantized)
  QuantizedWeight embed_tokens_q;   // quantized embedding (if quantized)
  bool embed_is_quantized = false;
  QuantizedWeight lm_head;          // quantized output projection
  mx::array model_norm{0.0f};      // final RMSNorm weight

  // Per-layer weights
  std::vector<LlamaLayerWeights> layers;

  // True if lm_head is tied to embed_tokens (no separate lm_head weights)
  bool tie_word_embeddings = false;
};

// Load model configuration from config.json
LlamaConfig load_config(const std::string& model_dir);

// Load all weights from safetensors files in model_dir
LlamaWeights load_weights(const std::string& model_dir);

// Utility: get total memory footprint of loaded weights in bytes
size_t weight_memory_bytes(const LlamaWeights& weights);

}  // namespace turboquant
