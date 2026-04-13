#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "nlohmann/json.hpp"

#include "mlx/io.h"
#include "mlx/ops.h"

#include "engine/llama_loader.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace turboquant {

///////////////////////////////////////////////////////////////////////////////
// Config loading
///////////////////////////////////////////////////////////////////////////////

LlamaConfig load_config(const std::string& model_dir) {
  auto config_path = fs::path(model_dir) / "config.json";
  if (!fs::exists(config_path)) {
    throw std::runtime_error("config.json not found in " + model_dir);
  }

  std::ifstream f(config_path);
  json cfg = json::parse(f);

  LlamaConfig c;
  c.hidden_size = cfg.value("hidden_size", 8192);
  c.num_hidden_layers = cfg.value("num_hidden_layers", 80);
  c.num_attention_heads = cfg.value("num_attention_heads", 64);
  c.num_key_value_heads = cfg.value("num_key_value_heads", 8);
  c.head_dim = c.hidden_size / c.num_attention_heads;  // 128
  c.intermediate_size = cfg.value("intermediate_size", 28672);
  c.vocab_size = cfg.value("vocab_size", 128256);
  c.rope_theta = cfg.value("rope_theta", 500000.0f);
  c.rms_norm_eps = cfg.value("rms_norm_eps", 1e-5f);

  // Check for quantization config (MLX format)
  if (cfg.contains("quantization")) {
    auto& qcfg = cfg["quantization"];
    c.quantize_group_size = qcfg.value("group_size", 64);
    c.quantize_bits = qcfg.value("bits", 4);
  }

  // Check for tie_word_embeddings
  // (stored in config but we handle it in load_weights)

  return c;
}

///////////////////////////////////////////////////////////////////////////////
// Weight loading
///////////////////////////////////////////////////////////////////////////////

// Collect all .safetensors files in model_dir, sorted by name
static std::vector<std::string> find_safetensors(const std::string& model_dir) {
  std::vector<std::string> files;
  for (auto& entry : fs::directory_iterator(model_dir)) {
    if (entry.path().extension() == ".safetensors") {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

// Load a quantized weight triplet (weight, scales, biases) from the flat map
static QuantizedWeight load_quantized(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix) {
  QuantizedWeight qw;

  auto it_w = tensors.find(prefix + ".weight");
  auto it_s = tensors.find(prefix + ".scales");
  auto it_b = tensors.find(prefix + ".biases");

  if (it_w == tensors.end()) {
    throw std::runtime_error("Missing weight: " + prefix + ".weight");
  }
  if (it_s == tensors.end()) {
    throw std::runtime_error("Missing scales: " + prefix + ".scales");
  }

  qw.weight = it_w->second;
  qw.scales = it_s->second;

  // biases may be absent (some quantization modes don't use them)
  if (it_b != tensors.end()) {
    qw.biases = it_b->second;
  }

  return qw;
}

// Check if a key exists in the tensor map
static bool has_tensor(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& key) {
  return tensors.find(key) != tensors.end();
}

LlamaWeights load_weights(const std::string& model_dir) {
  LlamaWeights w;
  w.config = load_config(model_dir);

  // Load all safetensors files into one flat map
  auto st_files = find_safetensors(model_dir);
  if (st_files.empty()) {
    throw std::runtime_error("No .safetensors files found in " + model_dir);
  }

  std::unordered_map<std::string, mx::array> tensors;
  for (auto& file : st_files) {
    std::cerr << "[loader] loading " << fs::path(file).filename() << std::endl;
    auto [file_tensors, metadata] = mx::load_safetensors(file);
    for (auto& [name, arr] : file_tensors) {
      tensors.insert_or_assign(name, std::move(arr));
    }
  }

  std::cerr << "[loader] loaded " << tensors.size() << " tensors from "
            << st_files.size() << " files" << std::endl;

  // Embedding tokens — may be quantized or fp16
  auto emb_it = tensors.find("model.embed_tokens.weight");
  if (emb_it == tensors.end()) {
    throw std::runtime_error("Missing model.embed_tokens.weight");
  }
  if (has_tensor(tensors, "model.embed_tokens.scales")) {
    // Quantized embedding
    w.embed_tokens_q = load_quantized(tensors, "model.embed_tokens");
    w.embed_is_quantized = true;
    std::cerr << "[loader] embed_tokens is quantized" << std::endl;
  } else {
    w.embed_tokens = emb_it->second;
    w.embed_is_quantized = false;
  }

  // Final RMSNorm
  auto norm_it = tensors.find("model.norm.weight");
  if (norm_it == tensors.end()) {
    throw std::runtime_error("Missing model.norm.weight");
  }
  w.model_norm = norm_it->second;

  // LM head — may be tied to embedding
  if (has_tensor(tensors, "lm_head.weight")) {
    w.lm_head = load_quantized(tensors, "lm_head");
    w.tie_word_embeddings = false;
  } else {
    // Tied embeddings: lm_head shares embed_tokens
    w.tie_word_embeddings = true;
  }

  // Per-layer weights
  int n_layers = w.config.num_hidden_layers;
  w.layers.resize(n_layers);

  for (int i = 0; i < n_layers; i++) {
    std::string prefix = "model.layers." + std::to_string(i);
    auto& layer = w.layers[i];

    // Attention projections
    layer.q_proj = load_quantized(tensors, prefix + ".self_attn.q_proj");
    layer.k_proj = load_quantized(tensors, prefix + ".self_attn.k_proj");
    layer.v_proj = load_quantized(tensors, prefix + ".self_attn.v_proj");
    layer.o_proj = load_quantized(tensors, prefix + ".self_attn.o_proj");

    // MLP projections
    layer.gate_proj = load_quantized(tensors, prefix + ".mlp.gate_proj");
    layer.up_proj = load_quantized(tensors, prefix + ".mlp.up_proj");
    layer.down_proj = load_quantized(tensors, prefix + ".mlp.down_proj");

    // RMSNorm weights
    auto in_norm_it = tensors.find(prefix + ".input_layernorm.weight");
    if (in_norm_it == tensors.end()) {
      throw std::runtime_error("Missing " + prefix + ".input_layernorm.weight");
    }
    layer.input_layernorm = in_norm_it->second;

    auto post_norm_it = tensors.find(prefix + ".post_attention_layernorm.weight");
    if (post_norm_it == tensors.end()) {
      throw std::runtime_error("Missing " + prefix + ".post_attention_layernorm.weight");
    }
    layer.post_attention_layernorm = post_norm_it->second;

    if ((i + 1) % 20 == 0 || i == n_layers - 1) {
      std::cerr << "[loader] organized layer " << i + 1 << "/" << n_layers << std::endl;
    }
  }

  return w;
}

///////////////////////////////////////////////////////////////////////////////
// Memory accounting
///////////////////////////////////////////////////////////////////////////////

static size_t array_bytes(const mx::array& a) {
  return a.nbytes();
}

static size_t qweight_bytes(const QuantizedWeight& qw) {
  size_t total = array_bytes(qw.weight) + array_bytes(qw.scales);
  if (qw.biases.size() > 0) {
    total += array_bytes(qw.biases);
  }
  return total;
}

size_t weight_memory_bytes(const LlamaWeights& weights) {
  size_t total = 0;

  total += array_bytes(weights.embed_tokens);
  total += array_bytes(weights.model_norm);

  if (!weights.tie_word_embeddings) {
    total += qweight_bytes(weights.lm_head);
  }

  for (auto& layer : weights.layers) {
    total += qweight_bytes(layer.q_proj);
    total += qweight_bytes(layer.k_proj);
    total += qweight_bytes(layer.v_proj);
    total += qweight_bytes(layer.o_proj);
    total += qweight_bytes(layer.gate_proj);
    total += qweight_bytes(layer.up_proj);
    total += qweight_bytes(layer.down_proj);
    total += array_bytes(layer.input_layernorm);
    total += array_bytes(layer.post_attention_layernorm);
  }

  return total;
}

}  // namespace turboquant
