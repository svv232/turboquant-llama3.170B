#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "mlx/mlx.h"
#include "engine/qjl.h"

namespace mx = mlx::core;

int main() {
  std::cout << "=== QJL Validation (1-bit Key Sketching) ===\n\n";

  const int B = 1, Hq = 8, Hkv = 1, D = 128;
  const float attn_scale = 1.0f / std::sqrt(128.0f);

  int seq_lens[] = {128, 512, 2048};
  int sketch_dims[] = {128, 256, 384, 512, 768, 1024};

  std::cout << "Head_dim=" << D << ", attn_scale=" << attn_scale << "\n\n";

  std::cout << "Sketch_m | SeqLen | Score_MSE  | Score_Corr | Softmax_KL | Bytes/key | Compress | Encode(us)\n";
  std::cout << "---------|--------|------------|------------|------------|-----------|----------|----------\n";

  for (int Skv : seq_lens) {
    // Generate random Q and K
    auto Q = mx::random::normal({B, Hq, 1, D}, mx::float32);
    auto K = mx::random::normal({B, Hkv, Skv, D}, mx::float32);
    Q = mx::astype(Q, mx::float16);
    K = mx::astype(K, mx::float16);
    mx::eval(Q, K);

    // Exact scores: Q @ K^T * scale
    // Expand K for GQA
    auto K_exp = mx::reshape(
        mx::repeat(mx::reshape(K, {B, Hkv, 1, Skv, D}), Hq / Hkv, 2),
        {B, Hq, Skv, D});
    auto exact_scores = mx::multiply(
        mx::matmul(mx::astype(Q, mx::float32), mx::transpose(mx::astype(K_exp, mx::float32), {0, 1, 3, 2})),
        mx::array(attn_scale));
    mx::eval(exact_scores);
    // exact_scores: [B, Hq, 1, Skv]

    // Exact softmax
    auto exact_max = mx::max(exact_scores, -1, true);
    auto exact_exp = mx::exp(mx::subtract(exact_scores, exact_max));
    auto exact_softmax = mx::divide(exact_exp, mx::sum(exact_exp, -1, true));
    mx::eval(exact_softmax);

    for (int m : sketch_dims) {
      turboquant::QJLParams params{D, m, Hq, Hkv, attn_scale};

      // Generate projection matrix
      auto proj = turboquant::generate_projection_matrix(m, D, 42);
      mx::eval(proj);

      // Encode K
      auto t0 = std::chrono::high_resolution_clock::now();
      auto [sketches, norms] = turboquant::qjl_encode(K, proj);
      mx::eval(sketches, norms);
      auto t1 = std::chrono::high_resolution_clock::now();
      double encode_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

      // Estimate scores
      auto est_scores = turboquant::qjl_scores(Q, sketches, norms, proj, params);
      mx::eval(est_scores);

      // Score MSE
      auto score_diff = mx::subtract(
          mx::reshape(exact_scores, {-1}),
          mx::reshape(est_scores, {-1}));
      auto score_mse = mx::mean(mx::multiply(score_diff, score_diff));
      mx::eval(score_mse);

      // Score correlation (Pearson)
      auto exact_flat = mx::reshape(exact_scores, {-1});
      auto est_flat = mx::reshape(est_scores, {-1});
      auto ex_mean = mx::mean(exact_flat);
      auto es_mean = mx::mean(est_flat);
      mx::eval(ex_mean, es_mean);
      auto ex_centered = mx::subtract(exact_flat, ex_mean);
      auto es_centered = mx::subtract(est_flat, es_mean);
      auto cov = mx::mean(mx::multiply(ex_centered, es_centered));
      auto ex_std = mx::sqrt(mx::mean(mx::multiply(ex_centered, ex_centered)));
      auto es_std = mx::sqrt(mx::mean(mx::multiply(es_centered, es_centered)));
      auto corr = mx::divide(cov, mx::add(mx::multiply(ex_std, es_std), mx::array(1e-10f)));
      mx::eval(corr);

      // Softmax KL divergence
      auto est_max = mx::max(est_scores, -1, true);
      auto est_exp = mx::exp(mx::subtract(est_scores, est_max));
      auto est_softmax = mx::divide(est_exp, mx::sum(est_exp, -1, true));
      auto kl = mx::sum(mx::multiply(exact_softmax,
          mx::log(mx::divide(
              mx::add(exact_softmax, mx::array(1e-10f)),
              mx::add(est_softmax, mx::array(1e-10f))))));
      mx::eval(kl);

      // Compression: m/8 bytes sketch + 2 bytes norm vs D*2 bytes FP16
      float bytes_per_key = m / 8.0f + 2.0f;
      float fp16_bytes = D * 2.0f;
      float compress = fp16_bytes / bytes_per_key;

      std::cout << std::fixed
                << std::setw(8) << m
                << " | " << std::setw(6) << Skv
                << " | " << std::setprecision(6) << std::setw(10) << score_mse.item<float>()
                << " | " << std::setprecision(4) << std::setw(10) << corr.item<float>()
                << " | " << std::setprecision(4) << std::setw(10) << kl.item<float>()
                << " | " << std::setprecision(1) << std::setw(9) << bytes_per_key
                << " | " << std::setprecision(2) << std::setw(7) << compress << "x"
                << " | " << std::setprecision(0) << std::setw(9) << encode_us
                << "\n";
    }
    std::cout << "\n";
  }

  // Compare with int4 compression for reference
  std::cout << "=== Reference Compression Ratios ===\n";
  std::cout << "FP16:      256 bytes/key (1.00x)\n";
  std::cout << "Int4:       80 bytes/key (3.20x) [64 packed + 8 scale + 8 zero]\n";
  std::cout << "QJL m=384:  50 bytes/key (5.12x) [48 sketch + 2 norm]\n";
  std::cout << "QJL m=256:  34 bytes/key (7.53x) [32 sketch + 2 norm]\n";
  std::cout << "QJL m=128:  18 bytes/key (14.22x) [16 sketch + 2 norm]\n";

  // Combined K(QJL) + V(int4) compression
  std::cout << "\n=== Combined KV Compression (K=QJL, V=int4) ===\n";
  float v_int4_bytes = 80.0f;  // per value vector
  for (int m : {128, 256, 384, 512}) {
    float k_bytes = m / 8.0f + 2.0f;
    float kv_bytes = k_bytes + v_int4_bytes;
    float fp16_kv_bytes = 2 * D * 2.0f;  // K + V in FP16
    float compress = fp16_kv_bytes / kv_bytes;
    float kv_128k_gb = 80.0 * 8 * 131072 * kv_bytes / (1024.0 * 1024.0 * 1024.0);
    std::cout << "  m=" << m << ": " << std::setprecision(1) << kv_bytes
              << " bytes/KV (" << std::setprecision(2) << compress << "x)"
              << " → 128K KV = " << std::setprecision(1) << kv_128k_gb << " GB\n";
  }

  std::cout << "\n=== Done ===\n";
  return 0;
}
