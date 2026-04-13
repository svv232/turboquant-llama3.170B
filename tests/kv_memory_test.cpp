#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  std::cout << "=== KV Cache Memory Verification ===\n\n";

  // Llama 3.1 70B parameters
  const int n_layers = 80;
  const int n_kv_heads = 8;
  const int head_dim = 128;
  const int group_size = 32;  // int4 quantization group size
  const int batch = 1;

  // Test at different sequence lengths
  int seq_lens[] = {1024, 4096, 16384, 65536, 131072};

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "SeqLen  | Int4 KV (GB) | FP16 KV (GB) | Ratio | Alloc Time(ms)\n";
  std::cout << "--------|-------------|-------------|-------|---------------\n";

  for (int seq_len : seq_lens) {
    int n_groups = head_dim / group_size;
    int packed_dim = head_dim / 2;  // int4: 2 values per byte

    // Calculate expected sizes per layer
    // K: quant[B,H,S,D/2] + scales[B,H,S,D/gs] + zeros[B,H,S,D/gs]
    // V: same
    size_t quant_bytes = (size_t)batch * n_kv_heads * seq_len * packed_dim * sizeof(uint8_t);
    size_t scale_bytes = (size_t)batch * n_kv_heads * seq_len * n_groups * 2;  // float16
    size_t zero_bytes = scale_bytes;
    size_t kv_per_layer = 2 * (quant_bytes + scale_bytes + zero_bytes);  // K + V
    size_t total_int4 = kv_per_layer * n_layers;

    // FP16 KV for comparison
    size_t fp16_per_layer = 2 * (size_t)batch * n_kv_heads * seq_len * head_dim * 2;  // 2 bytes per float16
    size_t total_fp16 = fp16_per_layer * n_layers;

    double int4_gb = total_int4 / (1024.0 * 1024.0 * 1024.0);
    double fp16_gb = total_fp16 / (1024.0 * 1024.0 * 1024.0);
    double ratio = fp16_gb / int4_gb;

    // Actually allocate the arrays to verify memory
    auto t0 = std::chrono::high_resolution_clock::now();

    // Only allocate for a manageable number of layers to verify pattern
    int test_layers = (seq_len <= 16384) ? n_layers : 10;
    size_t allocated = 0;
    for (int l = 0; l < test_layers; l++) {
      auto k_q = mx::zeros({batch, n_kv_heads, seq_len, packed_dim}, mx::uint8);
      auto k_s = mx::zeros({batch, n_kv_heads, seq_len, n_groups}, mx::float16);
      auto k_z = mx::zeros({batch, n_kv_heads, seq_len, n_groups}, mx::float16);
      auto v_q = mx::zeros({batch, n_kv_heads, seq_len, packed_dim}, mx::uint8);
      auto v_s = mx::zeros({batch, n_kv_heads, seq_len, n_groups}, mx::float16);
      auto v_z = mx::zeros({batch, n_kv_heads, seq_len, n_groups}, mx::float16);
      mx::eval(k_q, k_s, k_z, v_q, v_s, v_z);
      allocated += k_q.nbytes() + k_s.nbytes() + k_z.nbytes() +
                   v_q.nbytes() + v_s.nbytes() + v_z.nbytes();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double alloc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Verify allocated matches calculation
    size_t expected = kv_per_layer * test_layers;
    bool match = (allocated == expected);

    std::cout << std::setw(7) << seq_len
              << " | " << std::setw(11) << int4_gb
              << " | " << std::setw(11) << fp16_gb
              << " | " << std::setw(5) << ratio << "x"
              << " | " << std::setw(8) << alloc_ms
              << (match ? "" : " MISMATCH!")
              << " (" << test_layers << " layers)"
              << "\n";
  }

  // Memory budget analysis
  std::cout << "\n=== Memory Budget (M1 Max 64GB) ===\n";
  double weights_gb = 35.0;
  double kv_int4_128k_gb = (double)(2 * batch * n_kv_heads * 131072 *
      (head_dim/2 + 2*(head_dim/group_size)*2)) * n_layers / (1024.0*1024.0*1024.0);
  double overhead_gb = 2.0;  // OS, MLX runtime, etc.

  std::cout << "  4-bit weights:     " << weights_gb << " GB\n";
  std::cout << "  Int4 KV (128K):    " << kv_int4_128k_gb << " GB\n";
  std::cout << "  Runtime overhead:  " << overhead_gb << " GB\n";
  std::cout << "  Total:             " << (weights_gb + kv_int4_128k_gb + overhead_gb) << " GB\n";
  std::cout << "  Available:         64.0 GB\n";
  std::cout << "  Headroom:          " << (64.0 - weights_gb - kv_int4_128k_gb - overhead_gb) << " GB\n";
  std::cout << "  Status:            " << ((weights_gb + kv_int4_128k_gb + overhead_gb < 64.0) ? "FITS" : "DOESN'T FIT") << "\n";

  std::cout << "\n=== Done ===\n";
  return 0;
}
