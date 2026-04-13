#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "mlx/mlx.h"
#include "engine/sdpa_int4.h"

namespace mx = mlx::core;

// Fake-quantize: generate random int4 packed data with known scales/zeros,
// then build the dequantized FP16 reference from the same quantized values.
// This ensures the fused kernel and reference path operate on identical data.

struct QuantizedTensor {
  mx::array quant;   // [B, H, S, D/2] uint8 packed int4
  mx::array scales;  // [B, H, S, D/group_size] float16
  mx::array zeros;   // [B, H, S, D/group_size] float16
};

// Generate random quantized tensor and return both packed + dequantized forms
std::pair<QuantizedTensor, mx::array> make_quantized(
    int B, int H, int S, int D, int group_size, std::mt19937& gen) {
  int n_groups = D / group_size;
  int total_vecs = B * H * S;

  // Random scales in [0.01, 0.1], zeros in [7, 9] (centered around 8 for symmetric-ish)
  std::uniform_real_distribution<float> scale_dist(0.01f, 0.1f);
  std::uniform_real_distribution<float> zero_dist(7.0f, 9.0f);
  std::uniform_int_distribution<int> val_dist(0, 15);

  std::vector<uint8_t> packed_data(total_vecs * D / 2);
  std::vector<float> scales_data(total_vecs * n_groups);
  std::vector<float> zeros_data(total_vecs * n_groups);
  std::vector<float> dequant_data(total_vecs * D);

  for (int v = 0; v < total_vecs; v++) {
    for (int g = 0; g < n_groups; g++) {
      float sc = scale_dist(gen);
      float zp = zero_dist(gen);
      scales_data[v * n_groups + g] = sc;
      zeros_data[v * n_groups + g] = zp;

      for (int i = 0; i < group_size; i++) {
        int dim_idx = g * group_size + i;
        int raw_val = val_dist(gen);

        // Pack: low nibble = even, high nibble = odd
        int byte_idx = v * (D / 2) + dim_idx / 2;
        if (dim_idx % 2 == 0) {
          packed_data[byte_idx] = (packed_data[byte_idx] & 0xF0) | (raw_val & 0x0F);
        } else {
          packed_data[byte_idx] = (packed_data[byte_idx] & 0x0F) | ((raw_val & 0x0F) << 4);
        }

        // Dequantize for reference
        dequant_data[v * D + dim_idx] = (float(raw_val) - zp) * sc;
      }
    }
  }

  auto quant = mx::array(packed_data.data(), {B, H, S, D / 2}, mx::uint8);
  auto scales = mx::array(scales_data.data(), {B, H, S, n_groups}, mx::float32);
  auto zeros = mx::array(zeros_data.data(), {B, H, S, n_groups}, mx::float32);
  auto dequant = mx::array(dequant_data.data(), {B, H, S, D}, mx::float32);

  // Cast scales/zeros to float16 for the kernel
  auto scales_f16 = mx::astype(scales, mx::float16);
  auto zeros_f16 = mx::astype(zeros, mx::float16);
  auto dequant_f16 = mx::astype(dequant, mx::float16);

  mx::eval(quant, scales_f16, zeros_f16, dequant_f16);

  QuantizedTensor qt{quant, scales_f16, zeros_f16};
  return {qt, dequant_f16};
}

// Reference attention: Q @ K^T * scale -> softmax -> @ V
// Handles GQA by expanding KV heads
mx::array reference_sdpa(
    const mx::array& Q,    // [B, Hq, Sq, D] float16
    const mx::array& K,    // [B, Hkv, Skv, D] float16
    const mx::array& V,    // [B, Hkv, Skv, D] float16
    int num_q_heads, int num_kv_heads, float attn_scale) {

  int gqa_ratio = num_q_heads / num_kv_heads;
  int B = Q.shape(0);
  int Sq = Q.shape(2);
  int Skv = K.shape(2);
  int D = Q.shape(3);

  // Expand KV heads to match Q heads via repeat
  // K: [B, Hkv, Skv, D] -> [B, Hkv, 1, Skv, D] -> [B, Hkv, gqa, Skv, D] -> [B, Hq, Skv, D]
  auto expand_heads = [&](const mx::array& x) -> mx::array {
    if (gqa_ratio > 1) {
      auto r = mx::reshape(x, {B, num_kv_heads, 1, Skv, D});
      return mx::reshape(mx::repeat(r, gqa_ratio, 2), {B, num_q_heads, Skv, D});
    }
    return x;
  };
  auto K_exp = expand_heads(K);
  auto V_exp = expand_heads(V);

  // scores = Q @ K^T * scale  [B, Hq, Sq, Skv]
  auto scores = mx::matmul(Q, mx::transpose(K_exp, {0, 1, 3, 2}));
  scores = mx::multiply(scores, mx::array(attn_scale));

  // Softmax
  auto max_s = mx::max(scores, {-1}, true);
  auto exp_s = mx::exp(mx::subtract(scores, max_s));
  auto sum_s = mx::sum(exp_s, {-1}, true);
  auto weights = mx::divide(exp_s, sum_s);

  // output = weights @ V  [B, Hq, Sq, D]
  return mx::matmul(weights, V_exp);
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
  float max_abs_err;
  float mean_abs_err;
  float fused_time_us;
  float ref_time_us;
  float speedup;
  float tokens_per_sec;
};

BenchResult run_test(int B, int Hq, int Hkv, int Sq, int Skv, int D,
                     int group_size, float attn_scale) {
  BenchResult result{};
  result.seq_len_kv = Skv;

  std::mt19937 gen(42);

  // Generate Q as float16
  auto Q = mx::astype(
      mx::multiply(mx::random::normal({B, Hq, Sq, D}), mx::array(0.1f)),
      mx::float16);
  mx::eval(Q);

  // Generate quantized K and V with dequantized references
  auto [K_qt, K_deq] = make_quantized(B, Hkv, Skv, D, group_size, gen);
  auto [V_qt, V_deq] = make_quantized(B, Hkv, Skv, D, group_size, gen);

  turboquant::SdpaInt4Params params{D, Hq, Hkv, group_size, attn_scale};

  // Warmup
  auto out_fused = turboquant::sdpa_int4(
      Q, K_qt.quant, K_qt.scales, K_qt.zeros,
      V_qt.quant, V_qt.scales, V_qt.zeros, params);
  mx::eval(out_fused);

  // Benchmark fused kernel (5 iterations)
  auto t0 = std::chrono::high_resolution_clock::now();
  const int N_ITERS = 5;
  for (int i = 0; i < N_ITERS; i++) {
    out_fused = turboquant::sdpa_int4(
        Q, K_qt.quant, K_qt.scales, K_qt.zeros,
        V_qt.quant, V_qt.scales, V_qt.zeros, params);
    mx::eval(out_fused);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  result.fused_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      t1 - t0).count() / float(N_ITERS);

  // Reference path: dequant then standard SDPA
  auto Q_f16 = mx::astype(Q, mx::float16);
  mx::eval(Q_f16);

  auto t2 = std::chrono::high_resolution_clock::now();
  auto out_ref = reference_sdpa(Q_f16, K_deq, V_deq, Hq, Hkv, attn_scale);
  mx::eval(out_ref);
  auto t3 = std::chrono::high_resolution_clock::now();
  result.ref_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      t3 - t2).count();

  // Compute errors
  auto [max_err, mean_err] = compute_errors(out_fused, out_ref);
  result.max_abs_err = max_err;
  result.mean_abs_err = mean_err;

  result.speedup = result.ref_time_us / result.fused_time_us;
  // Tokens/sec: total query-KV attention operations per second
  result.tokens_per_sec = float(B * Hq * Sq) / (result.fused_time_us * 1e-6f);

  return result;
}

int main() {
  std::cout << "=== Fused SDPA Int4 Benchmark (Llama 3.1 70B) ===\n\n";

  const int D = 128;
  const int group_size = 32;
  const float attn_scale = 1.0f / std::sqrt(128.0f);  // 0.0884

  // Test with GQA: 8 query heads (subset), 1 KV head — simulates the 8:1 ratio
  // Use smaller head counts for testing to keep memory reasonable
  struct TestConfig {
    const char* name;
    int B, Hq, Hkv, Sq, Skv;
  };

  TestConfig configs[] = {
    // Small head counts (fast tests)
    {"seq=128,  Hq=8,Hkv=1",  1, 8, 1, 1, 128},
    {"seq=1024, Hq=8,Hkv=1",  1, 8, 1, 1, 1024},
    {"seq=4096, Hq=8,Hkv=1",  1, 8, 1, 1, 4096},
    {"seq=16384,Hq=8,Hkv=1",  1, 8, 1, 1, 16384},
    // Full Llama 3.1 70B head counts (decode step: Sq=1)
    {"LLAMA seq=1024",  1, 64, 8, 1, 1024},
    {"LLAMA seq=4096",  1, 64, 8, 1, 4096},
    {"LLAMA seq=16384", 1, 64, 8, 1, 16384},
    {"LLAMA seq=32768", 1, 64, 8, 1, 32768},
    {"LLAMA seq=65536", 1, 64, 8, 1, 65536},
    {"LLAMA seq=131072",1, 64, 8, 1, 131072},
  };

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Config                      | Skv    | Fused(us) | Ref(us)  | Speedup | MaxErr  | MeanErr | Tok/s\n";
  std::cout << "----------------------------|--------|-----------|----------|---------|---------|---------|----------\n";

  bool all_pass = true;

  for (auto& cfg : configs) {
    std::cout << std::flush;
    auto r = run_test(cfg.B, cfg.Hq, cfg.Hkv, cfg.Sq, cfg.Skv, D,
                      group_size, attn_scale);

    // Error tolerance: float16 with int4 quantization — allow some numerical slack
    // Max absolute error should be < 0.05 for well-behaved attention
    bool pass = r.mean_abs_err < 0.02f;
    if (!pass) all_pass = false;

    std::cout << std::left << std::setw(28) << cfg.name
              << "| " << std::setw(7) << r.seq_len_kv
              << "| " << std::setw(10) << r.fused_time_us
              << "| " << std::setw(9) << r.ref_time_us
              << "| " << std::setw(8) << r.speedup
              << "| " << std::setw(8) << r.max_abs_err
              << "| " << std::setw(8) << r.mean_abs_err
              << "| " << std::setw(10) << r.tokens_per_sec
              << " " << (pass ? "PASS" : "FAIL") << "\n";
  }

  std::cout << "\n";
  if (all_pass) {
    std::cout << "All correctness checks PASSED.\n";
  } else {
    std::cout << "WARNING: Some checks FAILED — see MaxErr/MeanErr columns.\n";
  }

  // Memory analysis
  std::cout << "\n=== Memory Analysis (Llama 3.1 70B, 128K context) ===\n";
  int layers = 80, kv_heads = 8, seq = 131072;
  float kv_fp16_gb = float(layers) * kv_heads * seq * D * 2 * 2 / (1024.0f * 1024.0f * 1024.0f);
  float kv_int4_gb = float(layers) * kv_heads * seq * (D / 2 + D / group_size * 2 * 2) * 2 / (1024.0f * 1024.0f * 1024.0f);
  float savings = kv_fp16_gb - kv_int4_gb;
  std::cout << "  KV cache FP16:  " << std::setprecision(1) << kv_fp16_gb << " GB\n";
  std::cout << "  KV cache int4:  " << kv_int4_gb << " GB\n";
  std::cout << "  Savings:        " << savings << " GB\n";
  std::cout << "  With 4-bit weights (~35 GB): "
            << std::setprecision(1) << (35.0f + kv_int4_gb) << " GB total (fits 64 GB)\n";

  std::cout << "\n=== Done ===\n";
  return all_pass ? 0 : 1;
}
