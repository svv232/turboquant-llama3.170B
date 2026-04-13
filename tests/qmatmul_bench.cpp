#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "mlx/mlx.h"

namespace mx = mlx::core;

// Benchmark quantized matmul for Llama 3.1 70B linear layers
// All 7 projections per layer at decode (Sq=1):
//   q_proj:    [1, 8192] @ [8192, 8192] → [1, 8192]   (64 heads × 128 dim)
//   k_proj:    [1, 8192] @ [1024, 8192] → [1, 1024]   (8 KV heads × 128 dim)
//   v_proj:    [1, 8192] @ [1024, 8192] → [1, 1024]
//   o_proj:    [1, 8192] @ [8192, 8192] → [1, 8192]
//   gate_proj: [1, 8192] @ [28672, 8192] → [1, 28672]
//   up_proj:   [1, 8192] @ [28672, 8192] → [1, 28672]
//   down_proj: [1, 28672] @ [8192, 28672] → [1, 8192]

struct LayerConfig {
  const char* name;
  int in_features;
  int out_features;
};

double bench_qmatmul(int in_features, int out_features, int group_size, int bits, int n_iters) {
  // Create fake quantized weight: [out_features, in_features/pack_factor] uint32
  int pack_factor = 32 / bits;  // 8 for 4-bit
  int packed_cols = in_features / pack_factor;
  int n_groups = in_features / group_size;

  auto weight = mx::zeros({out_features, packed_cols}, mx::uint32);
  auto scales = mx::ones({out_features, n_groups}, mx::float16);
  auto biases = mx::zeros({out_features, n_groups}, mx::float16);
  auto x = mx::ones({1, in_features}, mx::float16);
  mx::eval(weight, scales, biases, x);

  // Warmup
  auto out = mx::quantized_matmul(x, weight, scales, biases, true, group_size, bits);
  mx::eval(out);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n_iters; i++) {
    out = mx::quantized_matmul(x, weight, scales, biases, true, group_size, bits);
    mx::eval(out);
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double, std::micro>(t1 - t0).count() / n_iters;
}

int main() {
  std::cout << "=== Quantized Matmul Benchmark (Llama 3.1 70B) ===\n\n";

  const int group_size = 64;
  const int bits = 4;
  const int n_iters = 20;

  LayerConfig layers[] = {
    {"q_proj",    8192, 8192},
    {"k_proj",    8192, 1024},
    {"v_proj",    8192, 1024},
    {"o_proj",    8192, 8192},
    {"gate_proj", 8192, 28672},
    {"up_proj",   8192, 28672},
    {"down_proj", 28672, 8192},
  };

  std::cout << std::fixed << std::setprecision(1);
  std::cout << "Layer        | Shape           | Time (us) | GFLOPS\n";
  std::cout << "-------------|-----------------|-----------|-------\n";

  double total_us = 0;
  for (auto& l : layers) {
    double us = bench_qmatmul(l.in_features, l.out_features, group_size, bits, n_iters);
    double flops = 2.0 * l.in_features * l.out_features;
    double gflops = flops / (us * 1e3);

    std::cout << std::left << std::setw(13) << l.name
              << "| [1," << std::setw(5) << l.in_features << "]→" << std::setw(6) << l.out_features
              << " | " << std::setw(9) << us
              << " | " << gflops << "\n";
    total_us += us;
  }

  std::cout << "-------------|-----------------|-----------|-------\n";
  std::cout << "TOTAL (1 layer, 7 projections): " << total_us << " us = "
            << std::setprecision(2) << total_us / 1000.0 << " ms\n\n";

  // Full model estimate
  int n_layers = 80;
  double layer_matmul_ms = total_us / 1000.0;
  double total_matmul_ms = layer_matmul_ms * n_layers;
  double attn_per_layer_ms = 9.935;  // from 128K benchmark
  double total_attn_ms = attn_per_layer_ms * n_layers;
  double total_decode_ms = total_matmul_ms + total_attn_ms;

  std::cout << "=== Full Model Decode Estimate (128K context) ===\n";
  std::cout << "  Matmul per layer: " << std::setprecision(2) << layer_matmul_ms << " ms\n";
  std::cout << "  Attention per layer: " << attn_per_layer_ms << " ms\n";
  std::cout << "  Total per layer: " << (layer_matmul_ms + attn_per_layer_ms) << " ms\n";
  std::cout << "  80 layers matmul: " << std::setprecision(0) << total_matmul_ms << " ms\n";
  std::cout << "  80 layers attention: " << total_attn_ms << " ms\n";
  std::cout << "  Total decode per token: " << total_decode_ms << " ms\n";
  std::cout << "  Estimated tok/s: " << std::setprecision(1) << (1000.0 / total_decode_ms) << "\n";

  std::cout << "\n=== Done ===\n";
  return 0;
}
