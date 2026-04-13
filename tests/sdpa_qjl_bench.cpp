#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "mlx/mlx.h"
#include "engine/sdpa_qjl.h"
#include "engine/qjl.h"

namespace mx = mlx::core;

// Generate random int4 quantized tensor and its dequantized float16 form
struct QuantizedTensor {
  mx::array quant;   // [B, H, S, D/2] uint8
  mx::array scales;  // [B, H, S, D/group] float16
  mx::array zeros;   // [B, H, S, D/group] float16
};

std::pair<QuantizedTensor, mx::array> make_quantized(
    int B, int H, int S, int D, int group_size, std::mt19937& gen) {
  int n_groups = D / group_size;
  int total_vecs = B * H * S;

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
        int byte_idx = v * (D / 2) + dim_idx / 2;
        if (dim_idx % 2 == 0) {
          packed_data[byte_idx] = (packed_data[byte_idx] & 0xF0) | (raw_val & 0x0F);
        } else {
          packed_data[byte_idx] = (packed_data[byte_idx] & 0x0F) | ((raw_val & 0x0F) << 4);
        }
        dequant_data[v * D + dim_idx] = (float(raw_val) - zp) * sc;
      }
    }
  }

  auto quant = mx::array(packed_data.data(), {B, H, S, D / 2}, mx::uint8);
  auto scales = mx::astype(mx::array(scales_data.data(), {B, H, S, n_groups}, mx::float32), mx::float16);
  auto zeros = mx::astype(mx::array(zeros_data.data(), {B, H, S, n_groups}, mx::float32), mx::float16);
  auto dequant = mx::astype(mx::array(dequant_data.data(), {B, H, S, D}, mx::float32), mx::float16);
  mx::eval(quant, scales, zeros, dequant);
  return {{quant, scales, zeros}, dequant};
}

// Reference path: use MLX-ops QJL scores + softmax + dequanted V matmul
mx::array reference_qjl_sdpa(
    const mx::array& Q,           // [B, Hq, Sq, D] float16
    const mx::array& K,           // [B, Hkv, Skv, D] float16 (dequantized)
    const mx::array& V,           // [B, Hkv, Skv, D] float16 (dequantized)
    const mx::array& proj_matrix, // [m, D] int8
    const turboquant::QJLParams& qjl_params,
    int num_q_heads, int num_kv_heads) {

  // Get QJL-estimated scores
  auto [K_sketches, K_norms] = turboquant::qjl_encode(K, proj_matrix);
  mx::eval(K_sketches, K_norms);
  auto scores = turboquant::qjl_scores(Q, K_sketches, K_norms, proj_matrix, qjl_params);
  // scores: [B, Hq, Sq, Skv]

  // Softmax
  auto max_s = mx::max(scores, {-1}, true);
  auto exp_s = mx::exp(mx::subtract(scores, max_s));
  auto sum_s = mx::sum(exp_s, {-1}, true);
  auto weights = mx::divide(exp_s, sum_s);

  // Expand V for GQA
  int gqa = num_q_heads / num_kv_heads;
  int B = V.shape(0);
  int Skv = V.shape(2);
  int D = V.shape(3);
  auto V_exp = V;
  if (gqa > 1) {
    V_exp = mx::reshape(
        mx::repeat(mx::reshape(V, {B, num_kv_heads, 1, Skv, D}), gqa, 2),
        {B, num_q_heads, Skv, D});
  }

  // output = weights @ V
  return mx::matmul(mx::astype(weights, mx::float16), V_exp);
}

std::pair<float, float> compute_errors(const mx::array& a, const mx::array& b) {
  auto diff = mx::abs(mx::subtract(mx::astype(a, mx::float32), mx::astype(b, mx::float32)));
  auto max_err = mx::max(diff);
  auto mean_err = mx::mean(diff);
  mx::eval(max_err, mean_err);
  return {max_err.item<float>(), mean_err.item<float>()};
}

struct BenchResult {
  int seq_len_kv;
  int sketch_dim;
  float max_abs_err;
  float mean_abs_err;
  float fused_time_us;
  float ref_time_us;
  float speedup;
};

BenchResult run_test(int B, int Hq, int Hkv, int Sq, int Skv, int D,
                     int sketch_dim, int v_group_size, float attn_scale) {
  BenchResult result{};
  result.seq_len_kv = Skv;
  result.sketch_dim = sketch_dim;

  std::mt19937 gen(42);

  // Generate random Q and K as float16
  auto Q = mx::astype(
      mx::multiply(mx::random::normal({B, Hq, Sq, D}), mx::array(0.1f)),
      mx::float16);
  auto K_fp16 = mx::astype(
      mx::multiply(mx::random::normal({B, Hkv, Skv, D}), mx::array(0.1f)),
      mx::float16);
  mx::eval(Q, K_fp16);

  // Generate quantized V with dequantized reference
  auto [V_qt, V_deq] = make_quantized(B, Hkv, Skv, D, v_group_size, gen);

  // Build QJL projection matrix and encode
  auto proj = turboquant::generate_projection_matrix(sketch_dim, D, 42);
  mx::eval(proj);

  auto [Q_sketches, Q_norms] = turboquant::qjl_encode(Q, proj);
  auto [K_sketches, K_norms] = turboquant::qjl_encode(K_fp16, proj);
  mx::eval(Q_sketches, Q_norms, K_sketches, K_norms);

  turboquant::SdpaQJLParams params{D, sketch_dim, Hq, Hkv, v_group_size, attn_scale};

  // Warmup fused kernel
  auto out_fused = turboquant::sdpa_qjl(
      Q_sketches, Q_norms, K_sketches, K_norms,
      V_qt.quant, V_qt.scales, V_qt.zeros, params);
  mx::eval(out_fused);

  // Benchmark fused kernel
  const int N_ITERS = 5;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N_ITERS; i++) {
    out_fused = turboquant::sdpa_qjl(
        Q_sketches, Q_norms, K_sketches, K_norms,
        V_qt.quant, V_qt.scales, V_qt.zeros, params);
    mx::eval(out_fused);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  result.fused_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      t1 - t0).count() / float(N_ITERS);

  // Reference: MLX-ops QJL scores + softmax + dequantized V
  turboquant::QJLParams qjl_params{D, sketch_dim, Hq, Hkv, attn_scale};
  auto t2 = std::chrono::high_resolution_clock::now();
  auto out_ref = reference_qjl_sdpa(Q, K_fp16, V_deq, proj, qjl_params, Hq, Hkv);
  mx::eval(out_ref);
  auto t3 = std::chrono::high_resolution_clock::now();
  result.ref_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      t3 - t2).count();

  // Errors: note these include both QJL approximation error AND int4 V quantization error.
  // We compare fused (sketch scores + int4 V) vs reference (sketch scores + dequantized V).
  // The V quantization contributes error; the scores should be identical.
  auto [max_err, mean_err] = compute_errors(out_fused, out_ref);
  result.max_abs_err = max_err;
  result.mean_abs_err = mean_err;
  result.speedup = result.ref_time_us / result.fused_time_us;

  return result;
}

int main() {
  std::cout << "=== Fused QJL SDPA Benchmark (Llama 3.1 70B) ===\n\n";

  const int D = 128;
  const int v_group_size = 32;
  const float attn_scale = 1.0f / std::sqrt(128.0f);

  struct TestConfig {
    const char* name;
    int B, Hq, Hkv, Sq, Skv, m;
  };

  TestConfig configs[] = {
    // Correctness tests (small, GQA=8)
    {"m=256  seq=128",   1, 8, 1, 1, 128,  256},
    {"m=384  seq=128",   1, 8, 1, 1, 128,  384},
    {"m=512  seq=128",   1, 8, 1, 1, 128,  512},
    {"m=512  seq=256",   1, 8, 1, 1, 256,  512},
    {"m=512  seq=1024",  1, 8, 1, 1, 1024, 512},
    // Scaling tests
    {"m=512  seq=4096",  1, 8, 1, 1, 4096,  512},
    {"m=512  seq=16384", 1, 8, 1, 1, 16384, 512},
    // Llama-scale (64 Q heads, 8 KV heads)
    {"LLAMA m=384 seq=1024",  1, 64, 8, 1, 1024,  384},
    {"LLAMA m=384 seq=4096",  1, 64, 8, 1, 4096,  384},
    {"LLAMA m=512 seq=4096",  1, 64, 8, 1, 4096,  512},
    {"LLAMA m=512 seq=16384", 1, 64, 8, 1, 16384, 512},
    {"LLAMA m=512 seq=32768", 1, 64, 8, 1, 32768, 512},
  };

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Config                       | Skv    | m   | Fused(us) | Ref(us)  | Speedup | MaxErr  | MeanErr | Status\n";
  std::cout << "-----------------------------|--------|-----|-----------|----------|---------|---------|---------|------\n";

  bool all_pass = true;

  for (auto& cfg : configs) {
    std::cout << std::flush;
    auto r = run_test(cfg.B, cfg.Hq, cfg.Hkv, cfg.Sq, cfg.Skv, D,
                      cfg.m, v_group_size, attn_scale);

    // Tolerance: QJL is approximate, plus int4 V quantization. Be lenient.
    // The fused path and reference path use the same sketch scores, so the only
    // error source is int4 V quantization (fused) vs dequantized V (reference).
    bool pass = r.mean_abs_err < 0.05f;
    if (!pass) all_pass = false;

    std::cout << std::left << std::setw(29) << cfg.name
              << "| " << std::setw(7) << r.seq_len_kv
              << "| " << std::setw(4) << r.sketch_dim
              << "| " << std::setw(10) << r.fused_time_us
              << "| " << std::setw(9) << r.ref_time_us
              << "| " << std::setw(8) << r.speedup
              << "| " << std::setw(8) << r.max_abs_err
              << "| " << std::setw(8) << r.mean_abs_err
              << "| " << (pass ? "PASS" : "FAIL") << "\n";
  }

  std::cout << "\n";
  if (all_pass) {
    std::cout << "All correctness checks PASSED.\n";
  } else {
    std::cout << "WARNING: Some checks FAILED.\n";
  }

  // Memory analysis for QJL K + int4 V at 128K
  std::cout << "\n=== Memory: QJL K + Int4 V (Llama 3.1 70B, 128K context) ===\n";
  int layers = 80, kv_heads = 8, seq = 131072;
  int m = 512;
  // QJL K: m/8 bytes sketch + 2 bytes norm per token per head
  float k_qjl_gb = float(layers) * kv_heads * seq * (m / 8 + 2) / (1024.0f * 1024.0f * 1024.0f);
  // Int4 V: D/2 + D/group*2*2 bytes per token per head
  float v_int4_gb = float(layers) * kv_heads * seq * (D / 2 + D / v_group_size * 2 * 2) / (1024.0f * 1024.0f * 1024.0f);
  float fp16_gb = float(layers) * kv_heads * seq * D * 2 * 2 / (1024.0f * 1024.0f * 1024.0f);
  std::cout << std::setprecision(2);
  std::cout << "  K (QJL, m=" << m << "):  " << k_qjl_gb << " GB\n";
  std::cout << "  V (int4):         " << v_int4_gb << " GB\n";
  std::cout << "  Total KV:         " << (k_qjl_gb + v_int4_gb) << " GB\n";
  std::cout << "  KV FP16 baseline: " << fp16_gb << " GB\n";
  std::cout << "  Compression:      " << (fp16_gb / (k_qjl_gb + v_int4_gb)) << "x\n";
  std::cout << "  With 4-bit weights (~35 GB): "
            << std::setprecision(1) << (35.0f + k_qjl_gb + v_int4_gb) << " GB total\n";

  std::cout << "\n=== Done ===\n";
  return all_pass ? 0 : 1;
}
