#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/memory.h"
#include "mlx/backend/metal/metal.h"
#include "engine/llama_loader.h"
#include "engine/llama_model.h"

namespace mx = mlx::core;

// Read a line of token IDs from stdin (space-separated integers)
// Protocol: Python sends "TOKEN_IDS: 1 2 3 4 5\n", engine generates and prints tokens
static std::vector<int32_t> read_token_ids() {
  std::string line;
  if (!std::getline(std::cin, line)) return {};
  if (line.empty() || line.substr(0, 10) != "TOKEN_IDS:") return {};

  std::vector<int32_t> ids;
  std::istringstream iss(line.substr(10));
  int32_t id;
  while (iss >> id) ids.push_back(id);
  return ids;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: chat_engine <model_dir> [max_tokens]" << std::endl;
    return 1;
  }
  std::string model_dir = argv[1];
  int max_tokens = argc > 2 ? std::atoi(argv[2]) : 256;

  // Check for --fp16 flag
  bool fp16_mode = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--fp16") fp16_mode = true;
  }

  // Wire memory to prevent paging — critical for 39GB model on 64GB system
  // Without this, Metal dispatches fault pages in/out causing 10x slowdown
  auto device_info = mx::metal::device_info();
  size_t max_wired = std::get<size_t>(device_info.at("max_recommended_working_set_size"));
  mx::set_wired_limit(max_wired);
  std::cerr << "WIRED_LIMIT " << (max_wired / (1024*1024)) << "MB" << std::endl;

  // Load model
  std::cerr << "LOADING_MODEL" << std::endl;
  auto weights = turboquant::load_weights(model_dir);
  turboquant::LlamaModel model(std::move(weights));
  if (fp16_mode) {
    model.set_fp16_kv_mode(true);
    std::cerr << "FP16_KV_MODE" << std::endl;
  }
  std::cerr << "MODEL_READY" << std::endl;

  // Signal readiness
  std::cout << "READY" << std::endl;
  std::cout.flush();

  // Main loop: read prompts, generate responses
  while (true) {
    auto prompt_ids = read_token_ids();
    if (prompt_ids.empty()) break;

    // Fresh KV cache each turn (stateless — Python manages conversation)
    turboquant::KVCache cache;

    auto tokens = mx::array(prompt_ids.data(), {1, (int)prompt_ids.size()}, mx::int32);

    // Prefill
    auto t0 = std::chrono::high_resolution_clock::now();
    auto logits = model.forward(tokens, &cache);
    mx::eval(logits);
    auto t1 = std::chrono::high_resolution_clock::now();

    int next_token = turboquant::LlamaModel::argmax_last(logits);

    // Print prefill timing
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cerr << "PREFILL_DONE " << std::fixed << std::setprecision(1)
              << prefill_ms << "ms" << std::endl;

    // Decode loop — double-buffered async eval for pipeline parallelism
    // While GPU evaluates step N, CPU builds graph for step N+1
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "TOKEN " << next_token << std::endl;
    std::cout.flush();

    // Start first decode step
    auto tok_arr = mx::array(&next_token, {1, 1}, mx::int32);
    auto cur_logits = model.forward(tok_arr, &cache);
    mx::async_eval(cur_logits);

    for (int i = 0; i < max_tokens - 1; i++) {
      // Wait for current step's result
      mx::eval(cur_logits);
      next_token = turboquant::LlamaModel::argmax_last(cur_logits);

      if (next_token == 128001 || next_token == 128008 || next_token == 128009) break;

      std::cout << "TOKEN " << next_token << std::endl;
      std::cout.flush();

      // Build next step's graph (while current eval may still be flushing)
      tok_arr = mx::array(&next_token, {1, 1}, mx::int32);
      cur_logits = model.forward(tok_arr, &cache);
      mx::async_eval(cur_logits);  // Start eval in background
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    int n_generated = cache.layers[0].seq_len - (int)prompt_ids.size();

    std::cout << "DONE" << std::endl;
    std::cout.flush();

    double tok_per_sec = n_generated > 0 ? (n_generated / (decode_ms / 1000.0)) : 0;
    std::cerr << "DECODE_DONE " << n_generated << " tokens "
              << std::fixed << std::setprecision(1)
              << decode_ms << "ms "
              << tok_per_sec << " tok/s" << std::endl;
  }

  return 0;
}
