#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>

#include "mlx/mlx.h"

#include "engine/llama_loader.h"
#include "engine/llama_model.h"

namespace mx = mlx::core;

// Tee output to both stderr and a log file
class TeeStream {
 public:
  explicit TeeStream(const std::string& path) : file_(path) {}
  template <typename T>
  TeeStream& operator<<(const T& val) {
    std::cerr << val;
    file_ << val;
    return *this;
  }
  TeeStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
    manip(std::cerr);
    manip(file_);
    return *this;
  }
  void flush() { std::cerr.flush(); file_.flush(); }
 private:
  std::ofstream file_;
};

int main(int argc, char** argv) {
  // Model directory from CLI arg or env var
  std::string model_dir;
  if (argc > 1) {
    model_dir = argv[1];
  } else {
    const char* env = std::getenv("LLAMA_MODEL_DIR");
    if (env) {
      model_dir = env;
    } else {
      std::cerr << "Usage: llama_e2e <model_dir>" << std::endl;
      std::cerr << "  or set LLAMA_MODEL_DIR env var" << std::endl;
      return 1;
    }
  }

  TeeStream log("run.log");

  log << "=== TurboQuant Llama 3.1 70B E2E Test ===" << std::endl;
  log << "model_dir: " << model_dir << std::endl;

  // ---- Load config ----
  log << "[phase] loading config..." << std::endl;
  auto config = turboquant::load_config(model_dir);
  log << "  hidden_size=" << config.hidden_size
      << " layers=" << config.num_hidden_layers
      << " heads=" << config.num_attention_heads
      << "/" << config.num_key_value_heads
      << " head_dim=" << config.head_dim
      << " vocab=" << config.vocab_size
      << " rope_theta=" << config.rope_theta
      << " quant=" << config.quantize_bits << "bit"
      << " group=" << config.quantize_group_size
      << std::endl;

  // ---- Load weights ----
  log << "[phase] loading weights..." << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();

  auto weights = turboquant::load_weights(model_dir);

  auto t1 = std::chrono::high_resolution_clock::now();
  double load_secs = std::chrono::duration<double>(t1 - t0).count();
  size_t mem_bytes = turboquant::weight_memory_bytes(weights);
  log << "  loaded in " << std::fixed << std::setprecision(1) << load_secs << "s"
      << "  weight_mem=" << std::setprecision(2) << (mem_bytes / 1e9) << " GB"
      << std::endl;

  // ---- Build model ----
  log << "[phase] building model..." << std::endl;
  turboquant::LlamaModel model(std::move(weights));

  // ---- Prepare test input ----
  // Hardcoded token IDs: "<|begin_of_text|> Hello, I am"
  // Llama 3.1 tokenizer: 128000=BOS, 9906=Hello, 11=,, 358=I, 1097=am
  std::vector<int32_t> prompt_tokens = {128000, 9906, 11, 358, 1097};
  int prompt_len = static_cast<int>(prompt_tokens.size());

  auto tokens = mx::array(prompt_tokens.data(), {1, prompt_len}, mx::int32);
  log << "  prompt_len=" << prompt_len << " tokens" << std::endl;

  // ---- Forward pass (prefill) ----
  log << "[phase] running prefill forward pass..." << std::endl;
  turboquant::KVCache cache;

  auto t2 = std::chrono::high_resolution_clock::now();
  auto logits = model.forward(tokens, &cache);
  mx::eval(logits);
  auto t3 = std::chrono::high_resolution_clock::now();

  double prefill_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
  log << "  prefill_time=" << std::fixed << std::setprecision(1) << prefill_ms << " ms"
      << "  logits_shape=[" << logits.shape(0) << "," << logits.shape(1)
      << "," << logits.shape(2) << "]" << std::endl;

  // ---- Validate logits ----
  bool valid = true;
  auto logits_flat = mx::reshape(logits, {logits.shape(1) * logits.shape(2)});
  mx::eval(logits_flat);

  // Check for NaN/Inf
  auto has_nan = mx::any(mx::isnan(logits_flat));
  auto has_inf = mx::any(mx::isinf(logits_flat));
  mx::eval(has_nan);
  mx::eval(has_inf);

  if (has_nan.item<bool>()) {
    log << "  ERROR: logits contain NaN!" << std::endl;
    valid = false;
  }
  if (has_inf.item<bool>()) {
    log << "  ERROR: logits contain Inf!" << std::endl;
    valid = false;
  }

  // Check logits range (should be roughly [-50, 50] for a healthy model)
  auto logits_min = mx::min(logits_flat);
  auto logits_max = mx::max(logits_flat);
  auto logits_mean = mx::mean(mx::astype(logits_flat, mx::float32));
  mx::eval(logits_min);
  mx::eval(logits_max);
  mx::eval(logits_mean);

  float lmin = logits_min.item<float>();
  float lmax = logits_max.item<float>();
  float lmean = logits_mean.item<float>();
  log << "  logits: min=" << lmin << " max=" << lmax << " mean=" << lmean << std::endl;

  if (std::abs(lmin) > 1000 || std::abs(lmax) > 1000) {
    log << "  WARNING: logits have extreme values, model may be broken" << std::endl;
    valid = false;
  }

  // ---- Greedy decode one token ----
  int next_token = turboquant::LlamaModel::argmax_last(logits);
  log << "  greedy_next_token=" << next_token << std::endl;

  // ---- Generate a few more tokens (decode steps) ----
  log << "[phase] running decode steps..." << std::endl;
  int num_decode = 50;
  std::vector<int32_t> generated = {next_token};

  auto t4 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_decode; i++) {
    auto tok = mx::array(&generated.back(), {1, 1}, mx::int32);
    auto dec_logits = model.forward(tok, &cache);
    mx::eval(dec_logits);
    int next = turboquant::LlamaModel::argmax_last(dec_logits);
    generated.push_back(next);
  }
  auto t5 = std::chrono::high_resolution_clock::now();

  double decode_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
  double ms_per_token = decode_ms / num_decode;
  double tokens_per_sec = 1000.0 / ms_per_token;

  log << "  decode_tokens=" << num_decode
      << " total_ms=" << std::fixed << std::setprecision(1) << decode_ms
      << " ms_per_token=" << std::setprecision(1) << ms_per_token
      << " tok/s=" << std::setprecision(1) << tokens_per_sec
      << std::endl;

  log << "  generated_ids=[";
  for (size_t i = 0; i < generated.size(); i++) {
    if (i > 0) log << ", ";
    log << generated[i];
  }
  log << "]" << std::endl;

  // ---- KV cache memory ----
  size_t kv_bytes = 0;
  for (auto& entry : cache.layers) {
    kv_bytes += entry.k_quant.nbytes() + entry.k_scales.nbytes() + entry.k_zeros.nbytes();
    kv_bytes += entry.v_quant.nbytes() + entry.v_scales.nbytes() + entry.v_zeros.nbytes();
  }
  int total_seq = cache.layers[0].seq_len;
  log << "  kv_cache: seq_len=" << total_seq
      << " mem=" << std::setprecision(2) << (kv_bytes / 1e6) << " MB"
      << std::endl;

  // Extrapolate KV cache to 128K
  double kv_128k_gb = (kv_bytes / static_cast<double>(total_seq)) * 131072.0 / 1e9;
  log << "  kv_cache_extrapolated_128k=" << std::setprecision(2) << kv_128k_gb << " GB" << std::endl;

  // ---- Summary ----
  log << std::endl;
  log << "=== RESULT ===" << std::endl;
  log << "status: " << (valid ? "pass" : "fail") << std::endl;
  log << "prefill_ms: " << std::setprecision(1) << prefill_ms << std::endl;
  log << "decode_tok_per_sec: " << std::setprecision(1) << tokens_per_sec << std::endl;
  log << "weight_mem_gb: " << std::setprecision(2) << (mem_bytes / 1e9) << std::endl;
  log << "kv_cache_128k_gb: " << std::setprecision(2) << kv_128k_gb << std::endl;
  log << "logits_valid: " << (valid ? "true" : "false") << std::endl;

  // Clean up log file — Ensue is the durable record
  std::remove("run.log");

  return valid ? 0 : 1;
}
