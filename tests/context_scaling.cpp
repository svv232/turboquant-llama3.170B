#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "mlx/mlx.h"
#include "engine/llama_loader.h"
#include "engine/llama_model.h"

namespace mx = mlx::core;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: context_scaling <model_dir>" << std::endl;
    return 1;
  }
  std::string model_dir = argv[1];

  std::cout << "=== Context Length Scaling Test ===" << std::endl;

  // Load model
  std::cout << "[loading model...]" << std::endl;
  auto weights = turboquant::load_weights(model_dir);
  turboquant::LlamaModel model(std::move(weights));
  auto& cfg = model.config();
  std::cout << "  " << cfg.num_hidden_layers << " layers, head_dim="
            << cfg.head_dim << std::endl;

  // Test context lengths
  // Create synthetic prompts of different lengths using repeated token patterns
  // Use a realistic token sequence (from our 1B test: "Hello, I am a new member...")
  std::vector<int32_t> base_tokens = {
    128000, 9906, 11, 358, 1097, 264, 502, 4562, 315, 279,
    4029, 13, 358, 1097, 12304, 311, 387, 1618, 323, 358,
    1097, 24450, 311, 4048, 323, 4430, 856, 6677, 449, 3885,
    13, 358
  };

  int test_lengths[] = {32, 128, 256, 512};

  std::cout << std::fixed << std::setprecision(1);
  std::cout << "\nSeqLen | Prefill(ms) | Decode(ms) | Tok/s | First Token\n";
  std::cout << "-------|-------------|------------|-------|------------\n";

  for (int target_len : test_lengths) {
    // Build prompt of target_len tokens
    std::vector<int32_t> prompt;
    while ((int)prompt.size() < target_len) {
      int remaining = target_len - prompt.size();
      int copy_len = std::min(remaining, (int)base_tokens.size());
      prompt.insert(prompt.end(), base_tokens.begin(), base_tokens.begin() + copy_len);
    }

    auto tokens = mx::array(prompt.data(), {1, target_len}, mx::int32);

    // Fresh cache for each test
    turboquant::KVCache cache;

    // Prefill
    auto t0 = std::chrono::high_resolution_clock::now();
    auto logits = model.forward(tokens, &cache);
    mx::eval(logits);
    auto t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Get first predicted token
    int first_token = turboquant::LlamaModel::argmax_last(logits);

    // Decode 5 tokens
    auto t2 = std::chrono::high_resolution_clock::now();
    int n_decode = 5;
    std::vector<int32_t> generated = {first_token};
    for (int i = 0; i < n_decode - 1; i++) {
      auto tok = mx::array(&generated.back(), {1, 1}, mx::int32);
      auto dec_logits = model.forward(tok, &cache);
      mx::eval(dec_logits);
      generated.push_back(turboquant::LlamaModel::argmax_last(dec_logits));
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double tok_per_sec = n_decode / (decode_ms / 1000.0);

    // Check for NaN/Inf
    auto has_nan = mx::any(mx::isnan(mx::reshape(logits, {-1})));
    mx::eval(has_nan);
    bool valid = !has_nan.item<bool>();

    std::cout << std::setw(6) << target_len
              << " | " << std::setw(11) << prefill_ms
              << " | " << std::setw(10) << (decode_ms / n_decode)
              << " | " << std::setw(5) << tok_per_sec
              << " | " << first_token
              << (valid ? "" : " NAN!")
              << std::endl;
  }

  std::cout << "\nhead_dim=" << cfg.head_dim
            << (cfg.head_dim == 128 ? " (fused sdpa_int4 ACTIVE)" : " (FP16 fallback)")
            << std::endl;

  std::cout << "\n=== Done ===" << std::endl;
  return 0;
}
