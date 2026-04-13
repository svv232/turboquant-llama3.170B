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
    std::cerr << "Usage: decode_profile <model_dir>" << std::endl;
    return 1;
  }

  std::cout << "=== Decode Profiling ===" << std::endl;

  auto weights = turboquant::load_weights(argv[1]);
  turboquant::LlamaModel model(std::move(weights));
  auto& cfg = model.config();
  std::cout << cfg.num_hidden_layers << " layers, head_dim=" << cfg.head_dim << std::endl;

  // Prefill with a short prompt
  std::vector<int32_t> prompt = {128000, 9906, 11, 358, 1097};
  auto tokens = mx::array(prompt.data(), {1, 5}, mx::int32);
  turboquant::KVCache cache;

  std::cout << "\n[prefill]" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto logits = model.forward(tokens, &cache);
  mx::eval(logits);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "  prefill: " << std::fixed << std::setprecision(1)
            << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;

  int next_token = turboquant::LlamaModel::argmax_last(logits);

  // Profile 10 decode steps with detailed timing
  std::cout << "\n[decode steps]" << std::endl;
  std::cout << "Step | Total(ms) | Forward(ms) | Eval(ms) | Argmax(ms)\n";
  std::cout << "-----|-----------|-------------|----------|----------\n";

  std::vector<double> totals;
  for (int step = 0; step < 10; step++) {
    auto tok = mx::array(&next_token, {1, 1}, mx::int32);

    auto ta = std::chrono::high_resolution_clock::now();

    // Forward pass (graph construction, no computation yet)
    auto dec_logits = model.forward(tok, &cache);

    auto tb = std::chrono::high_resolution_clock::now();

    // Eval (actual GPU computation)
    mx::eval(dec_logits);

    auto tc = std::chrono::high_resolution_clock::now();

    // Argmax
    next_token = turboquant::LlamaModel::argmax_last(dec_logits);

    auto td = std::chrono::high_resolution_clock::now();

    double forward_ms = std::chrono::duration<double, std::milli>(tb - ta).count();
    double eval_ms = std::chrono::duration<double, std::milli>(tc - tb).count();
    double argmax_ms = std::chrono::duration<double, std::milli>(td - tc).count();
    double total_ms = std::chrono::duration<double, std::milli>(td - ta).count();
    totals.push_back(total_ms);

    std::cout << std::setw(4) << step
              << " | " << std::setw(9) << total_ms
              << " | " << std::setw(11) << forward_ms
              << " | " << std::setw(8) << eval_ms
              << " | " << std::setw(9) << argmax_ms
              << std::endl;
  }

  // Summary
  double avg = 0;
  for (auto t : totals) avg += t;
  avg /= totals.size();
  std::cout << "\nAvg total: " << std::setprecision(1) << avg << " ms/token ("
            << std::setprecision(2) << (1000.0 / avg) << " tok/s)" << std::endl;

  // KV cache size
  size_t kv_bytes = 0;
  for (auto& e : cache.layers) {
    kv_bytes += e.k_quant.nbytes() + e.k_scales.nbytes() + e.k_zeros.nbytes();
    kv_bytes += e.v_quant.nbytes() + e.v_scales.nbytes() + e.v_zeros.nbytes();
  }
  std::cout << "KV cache: " << cache.layers[0].seq_len << " tokens, "
            << std::setprecision(2) << (kv_bytes / 1e6) << " MB" << std::endl;

  std::cout << "\n=== Done ===" << std::endl;
  return 0;
}
