#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include "mlx/mlx.h"
#include "engine/sdpa_int4.h"

namespace mx = mlx::core;

// Quick benchmark: measure fused sdpa_int4 at 128K with different chunk sizes
// We do this by modifying the CHUNK_SIZE constant in the source and recompiling
// for each variant, but that's slow. Instead, we just benchmark the current
// chunk size at different effective sequence lengths that stress-test the split-K.

int main() {
  std::cout << "=== Split-K Chunk Size Analysis ===\n\n";

  const int D = 128;
  const int Hq = 64, Hkv = 8;
  const int group_size = 32;
  const float attn_scale = 1.0f / std::sqrt(128.0f);
  const int B = 1, Sq = 1;
  const int N_ITERS = 5;

  // Generate random data once
  std::mt19937 gen(42);

  turboquant::SdpaInt4Params params{D, Hq, Hkv, group_size, attn_scale};

  // Test at 128K with current chunk size (512)
  int Skv = 131072;
  int n_groups = D / group_size;

  // Create quantized KV data
  auto Q = mx::astype(
      mx::multiply(mx::random::normal({B, Hq, Sq, D}), mx::array(0.1f)),
      mx::float16);

  std::uniform_int_distribution<int> val_dist(0, 15);
  std::uniform_real_distribution<float> scale_dist(0.01f, 0.1f);
  std::uniform_real_distribution<float> zero_dist(7.0f, 9.0f);

  int total_vecs = B * Hkv * Skv;
  std::vector<uint8_t> packed(total_vecs * D / 2);
  std::vector<float> scales(total_vecs * n_groups);
  std::vector<float> zeros(total_vecs * n_groups);

  for (int v = 0; v < total_vecs; v++) {
    for (int g = 0; g < n_groups; g++) {
      scales[v * n_groups + g] = scale_dist(gen);
      zeros[v * n_groups + g] = zero_dist(gen);
      for (int i = 0; i < group_size; i++) {
        int dim_idx = g * group_size + i;
        int raw = val_dist(gen);
        int byte_idx = v * (D/2) + dim_idx/2;
        if (dim_idx % 2 == 0)
          packed[byte_idx] = (packed[byte_idx] & 0xF0) | (raw & 0x0F);
        else
          packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((raw & 0x0F) << 4);
      }
    }
  }

  auto K_q = mx::array(packed.data(), {B, Hkv, Skv, D/2}, mx::uint8);
  auto K_s = mx::astype(mx::array(scales.data(), {B, Hkv, Skv, n_groups}, mx::float32), mx::float16);
  auto K_z = mx::astype(mx::array(zeros.data(), {B, Hkv, Skv, n_groups}, mx::float32), mx::float16);
  auto V_q = mx::array(packed.data(), {B, Hkv, Skv, D/2}, mx::uint8);
  auto V_s = mx::astype(mx::array(scales.data(), {B, Hkv, Skv, n_groups}, mx::float32), mx::float16);
  auto V_z = mx::astype(mx::array(zeros.data(), {B, Hkv, Skv, n_groups}, mx::float32), mx::float16);

  mx::eval(Q, K_q, K_s, K_z, V_q, V_s, V_z);

  std::cout << "Benchmarking at Skv=" << Skv << " with Hq=" << Hq << " Hkv=" << Hkv << "\n";
  std::cout << "Current CHUNK_SIZE=512 (num_chunks=" << (Skv/512) << ")\n\n";

  // Warmup
  auto out = turboquant::sdpa_int4(Q, K_q, K_s, K_z, V_q, V_s, V_z, params);
  mx::eval(out);

  // Benchmark
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N_ITERS; i++) {
    out = turboquant::sdpa_int4(Q, K_q, K_s, K_z, V_q, V_s, V_z, params);
    mx::eval(out);
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_ITERS;
  std::cout << "128K fused SDPA: " << std::fixed << std::setprecision(0) << us << " us\n";
  std::cout << "Per-layer at 80 layers: " << std::setprecision(1) << (us * 80 / 1000.0) << " ms\n";
  std::cout << "Full model (80 layers attn + matmul): ~" << std::setprecision(0) << (us * 80 / 1000.0 + 467) << " ms/token\n";
  std::cout << "Estimated tok/s: " << std::setprecision(2) << (1000.0 / (us * 80 / 1000.0 + 467)) << "\n";

  // Also benchmark the overhead of the split-K reduce step
  // by comparing total time vs what single-pass would take
  // For this, we test at Skv=256 (single-pass) and extrapolate
  int Skv_short = 256;
  auto K_q_short = mx::slice(K_q, {0,0,0,0}, {B,Hkv,Skv_short,D/2});
  auto K_s_short = mx::slice(K_s, {0,0,0,0}, {B,Hkv,Skv_short,n_groups});
  auto K_z_short = mx::slice(K_z, {0,0,0,0}, {B,Hkv,Skv_short,n_groups});
  auto V_q_short = mx::slice(V_q, {0,0,0,0}, {B,Hkv,Skv_short,D/2});
  auto V_s_short = mx::slice(V_s, {0,0,0,0}, {B,Hkv,Skv_short,n_groups});
  auto V_z_short = mx::slice(V_z, {0,0,0,0}, {B,Hkv,Skv_short,n_groups});
  mx::eval(K_q_short, K_s_short, K_z_short, V_q_short, V_s_short, V_z_short);

  // Warmup
  out = turboquant::sdpa_int4(Q, K_q_short, K_s_short, K_z_short, V_q_short, V_s_short, V_z_short, params);
  mx::eval(out);

  auto t2 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N_ITERS; i++) {
    out = turboquant::sdpa_int4(Q, K_q_short, K_s_short, K_z_short, V_q_short, V_s_short, V_z_short, params);
    mx::eval(out);
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  double us_short = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_ITERS;

  std::cout << "\nSingle-pass (Skv=256): " << std::setprecision(0) << us_short << " us\n";
  std::cout << "Naive extrapolation to 128K (512x): " << std::setprecision(0) << (us_short * 512) << " us\n";
  std::cout << "Split-K speedup vs naive: " << std::setprecision(1) << ((us_short * 512) / us) << "x\n";

  std::cout << "\n=== Done ===\n";
  return 0;
}
