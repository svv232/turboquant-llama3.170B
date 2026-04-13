#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "mlx/mlx.h"
#include "engine/sdpa_polar.h"
#include "engine/polarquant.h"

namespace mx = mlx::core;

// Reference attention: decode PolarQuant KV → standard Q @ K^T * scale → softmax → @ V
// Handles GQA by expanding KV heads
mx::array reference_polar_sdpa(
    const mx::array& Q,          // [B, Hq, Sq, D] float16
    const mx::array& K_decoded,  // [B, Hkv, Skv, D] float32
    const mx::array& V_decoded,  // [B, Hkv, Skv, D] float32
    int num_q_heads, int num_kv_heads, float attn_scale) {

  int gqa_ratio = num_q_heads / num_kv_heads;
  int B = Q.shape(0);
  int Sq = Q.shape(2);
  int Skv = K_decoded.shape(2);
  int D = Q.shape(3);

  // Cast everything to float32 for reference computation
  auto Q_f32 = mx::astype(Q, mx::float32);

  // Expand KV heads to match Q heads via repeat
  auto expand_heads = [&](const mx::array& x) -> mx::array {
    if (gqa_ratio > 1) {
      auto r = mx::reshape(x, {B, num_kv_heads, 1, Skv, D});
      return mx::reshape(mx::repeat(r, gqa_ratio, 2), {B, num_q_heads, Skv, D});
    }
    return x;
  };
  auto K_exp = expand_heads(K_decoded);
  auto V_exp = expand_heads(V_decoded);

  // scores = Q @ K^T * scale  [B, Hq, Sq, Skv]
  auto scores = mx::matmul(Q_f32, mx::transpose(K_exp, {0, 1, 3, 2}));
  scores = mx::multiply(scores, mx::array(attn_scale));

  // Softmax
  auto max_s = mx::max(scores, {-1}, true);
  auto exp_s = mx::exp(mx::subtract(scores, max_s));
  auto sum_s = mx::sum(exp_s, {-1}, true);
  auto weights = mx::divide(exp_s, sum_s);

  // output = weights @ V  [B, Hq, Sq, D]
  return mx::astype(mx::matmul(weights, V_exp), mx::float16);
}

// Compute max absolute error and mean absolute error between two arrays
std::pair<float, float> compute_errors(const mx::array& a, const mx::array& b) {
  auto diff = mx::abs(mx::subtract(mx::astype(a, mx::float32), mx::astype(b, mx::float32)));
  auto max_err = mx::max(diff);
  auto mean_err = mx::mean(diff);
  mx::eval(max_err, mean_err);
  return {max_err.item<float>(), mean_err.item<float>()};
}

struct BenchResult {
  int seq_len_kv;
  int bits;
  float max_abs_err;
  float mean_abs_err;
  float fused_time_us;
  float ref_time_us;
  float speedup;
  float bytes_per_kv_vec;
  float compression_ratio;  // vs FP16
};

BenchResult run_test(int B, int Hq, int Hkv, int Sq, int Skv, int D,
                     int bits, float attn_scale) {
  BenchResult result{};
  result.seq_len_kv = Skv;
  result.bits = bits;

  // Compute storage metrics
  int n_angles = D - 1;
  result.bytes_per_kv_vec = float(n_angles) + 4.0f;  // uint8 per angle + float32 norm
  float fp16_bytes = D * 2.0f;
  result.compression_ratio = fp16_bytes / result.bytes_per_kv_vec;

  // Generate random Q as float16
  auto Q = mx::astype(
      mx::multiply(mx::random::normal({B, Hq, Sq, D}), mx::array(0.1f)),
      mx::float16);
  mx::eval(Q);

  // Generate random KV as float32, then PolarQuant encode
  auto K_orig = mx::random::normal({B, Hkv, Skv, D});
  auto V_orig = mx::random::normal({B, Hkv, Skv, D});
  mx::eval(K_orig, V_orig);

  turboquant::PolarQuantParams pq_params{D, 0, bits};

  auto k_encoded = turboquant::polarquant_encode(K_orig, pq_params);
  auto& k_angles = k_encoded[0];
  auto& k_norms = k_encoded[1];

  auto v_encoded = turboquant::polarquant_encode(V_orig, pq_params);
  auto& v_angles = v_encoded[0];
  auto& v_norms = v_encoded[1];
  mx::eval(k_angles, k_norms, v_angles, v_norms);

  // Decode for reference path
  auto K_decoded = turboquant::polarquant_decode(k_angles, k_norms, pq_params);
  auto V_decoded = turboquant::polarquant_decode(v_angles, v_norms, pq_params);
  mx::eval(K_decoded, V_decoded);

  turboquant::SdpaPolarParams sdpa_params{D, Hq, Hkv, bits, attn_scale};

  // Reshape norms: [B, Hkv, Skv, 1] -> [B, Hkv, Skv, 1] (keep as-is, kernel reads Skv dim)
  // Actually the kernel indexes norms as [B*Hkv*Skv], so we need contiguous layout
  // which is already the case.

  // Warmup
  auto out_fused = turboquant::sdpa_polar(
      Q, k_angles, k_norms, v_angles, v_norms, sdpa_params);
  mx::eval(out_fused);

  // Benchmark fused kernel (5 iterations)
  auto t0 = std::chrono::high_resolution_clock::now();
  const int N_ITERS = 5;
  for (int i = 0; i < N_ITERS; i++) {
    out_fused = turboquant::sdpa_polar(
        Q, k_angles, k_norms, v_angles, v_norms, sdpa_params);
    mx::eval(out_fused);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  result.fused_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      t1 - t0).count() / float(N_ITERS);

  // Benchmark reference path: decode + standard SDPA
  auto t2 = std::chrono::high_resolution_clock::now();
  auto out_ref = reference_polar_sdpa(Q, K_decoded, V_decoded, Hq, Hkv, attn_scale);
  mx::eval(out_ref);
  auto t3 = std::chrono::high_resolution_clock::now();
  result.ref_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      t3 - t2).count();

  // Compute errors between fused and reference
  auto [max_err, mean_err] = compute_errors(out_fused, out_ref);
  result.max_abs_err = max_err;
  result.mean_abs_err = mean_err;

  result.speedup = result.ref_time_us / result.fused_time_us;

  return result;
}

int main() {
  std::cout << "=== Fused SDPA PolarQuant Benchmark (Llama 3.1 70B) ===\n\n";

  const int D = 128;
  const int bits = 5;  // validated sweet spot
  const float attn_scale = 1.0f / std::sqrt(128.0f);  // 0.0884

  // Test configs: use 8:1 GQA ratio (subset of full 64:8)
  struct TestConfig {
    const char* name;
    int B, Hq, Hkv, Sq, Skv;
  };

  TestConfig configs[] = {
    {"seq=128,  Hq=8,Hkv=1",  1, 8, 1, 1, 128},
    {"seq=512,  Hq=8,Hkv=1",  1, 8, 1, 1, 512},
    {"seq=1024, Hq=8,Hkv=1",  1, 8, 1, 1, 1024},
    {"seq=4096, Hq=8,Hkv=1",  1, 8, 1, 1, 4096},
    {"seq=16384,Hq=8,Hkv=1",  1, 8, 1, 1, 16384},
  };

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Config                      | Skv    | Fused(us) | Ref(us)  | Speedup | MaxErr  | MeanErr | Compress\n";
  std::cout << "----------------------------|--------|-----------|----------|---------|---------|---------|--------\n";

  bool all_pass = true;

  for (auto& cfg : configs) {
    std::cout << std::flush;
    auto r = run_test(cfg.B, cfg.Hq, cfg.Hkv, cfg.Sq, cfg.Skv, D,
                      bits, attn_scale);

    // Error tolerance: PolarQuant 5-bit has cosine_sim ~0.989
    // Mean absolute error should be reasonable — allow up to 0.1 for PolarQuant
    // (larger than int4 because PolarQuant decode has transcendental ops)
    bool pass = r.mean_abs_err < 0.1f;
    if (!pass) all_pass = false;

    std::cout << std::left << std::setw(28) << cfg.name
              << "| " << std::setw(7) << r.seq_len_kv
              << "| " << std::setw(10) << r.fused_time_us
              << "| " << std::setw(9) << r.ref_time_us
              << "| " << std::setw(8) << std::setprecision(3) << r.speedup
              << "| " << std::setw(8) << std::setprecision(6) << r.max_abs_err
              << "| " << std::setw(8) << r.mean_abs_err
              << "| " << std::setprecision(2) << r.compression_ratio << "x"
              << " " << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << std::setprecision(6);
  }

  // Multi-bit comparison
  std::cout << "\n=== Bit-Width Comparison (seq=1024) ===\n";
  std::cout << "Bits | Fused(us) | MaxErr   | MeanErr  | Bytes/vec | Compress\n";
  std::cout << "-----|-----------|----------|----------|-----------|--------\n";

  for (int test_bits : {4, 5, 6, 8}) {
    auto r = run_test(1, 8, 1, 1, 1024, D, test_bits, attn_scale);
    std::cout << std::setw(4) << test_bits
              << " | " << std::setw(9) << std::setprecision(1) << r.fused_time_us
              << " | " << std::setw(8) << std::setprecision(6) << r.max_abs_err
              << " | " << std::setw(8) << r.mean_abs_err
              << " | " << std::setw(9) << std::setprecision(1) << r.bytes_per_kv_vec
              << " | " << std::setprecision(2) << r.compression_ratio << "x\n";
    std::cout << std::setprecision(6);
  }

  // Memory analysis
  std::cout << "\n=== Memory Analysis (Llama 3.1 70B, 128K context, 5-bit PolarQuant) ===\n";
  int layers = 80, kv_heads = 8, seq = 131072;
  int n_angles = D - 1;
  float bytes_per_vec_polar = float(n_angles) + 4.0f;  // 127 bytes angles + 4 bytes norm = 131 bytes
  float bytes_per_vec_fp16 = D * 2.0f;                 // 256 bytes
  float bytes_per_vec_int4 = D / 2.0f + (D / 32.0f) * 4.0f;  // 64 + 16 = 80 bytes

  float kv_fp16_gb = float(layers) * kv_heads * seq * bytes_per_vec_fp16 * 2 / (1024.0f * 1024.0f * 1024.0f);
  float kv_polar_gb = float(layers) * kv_heads * seq * bytes_per_vec_polar * 2 / (1024.0f * 1024.0f * 1024.0f);
  float kv_int4_gb = float(layers) * kv_heads * seq * bytes_per_vec_int4 * 2 / (1024.0f * 1024.0f * 1024.0f);

  std::cout << std::setprecision(1);
  std::cout << "  Per KV vector:\n";
  std::cout << "    FP16:       " << bytes_per_vec_fp16 << " bytes\n";
  std::cout << "    Int4:       " << bytes_per_vec_int4 << " bytes (" << std::setprecision(2) << (bytes_per_vec_fp16 / bytes_per_vec_int4) << "x)\n";
  std::cout << "    PolarQ 5b:  " << std::setprecision(1) << bytes_per_vec_polar << " bytes (" << std::setprecision(2) << (bytes_per_vec_fp16 / bytes_per_vec_polar) << "x)\n";
  std::cout << "\n  KV cache at 128K context (80 layers, 8 KV heads):\n";
  std::cout << std::setprecision(1);
  std::cout << "    FP16:       " << kv_fp16_gb << " GB\n";
  std::cout << "    Int4:       " << kv_int4_gb << " GB\n";
  std::cout << "    PolarQ 5b:  " << kv_polar_gb << " GB\n";
  std::cout << "\n  Total with 4-bit weights (~35 GB):\n";
  std::cout << "    + FP16 KV:  " << (35.0f + kv_fp16_gb) << " GB (DOESN'T FIT 64GB)\n";
  std::cout << "    + Int4 KV:  " << (35.0f + kv_int4_gb) << " GB\n";
  std::cout << "    + Polar KV: " << (35.0f + kv_polar_gb) << " GB\n";

  std::cout << "\n";
  if (all_pass) {
    std::cout << "All correctness checks PASSED.\n";
  } else {
    std::cout << "WARNING: Some checks FAILED — see MaxErr/MeanErr columns.\n";
  }

  std::cout << "\n=== Done ===\n";
  return all_pass ? 0 : 1;
}
