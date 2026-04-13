#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <chrono>

#include "mlx/mlx.h"
#include "engine/polarquant.h"

namespace mx = mlx::core;

// Compute cosine similarity between two float arrays
float cosine_similarity(const float* a, const float* b, int n) {
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
  for (int i = 0; i < n; i++) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12));
}

// Compute MSE between two float arrays
float mse(const float* a, const float* b, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    double d = a[i] - b[i];
    sum += d * d;
  }
  return static_cast<float>(sum / n);
}

// Reference softmax attention (no GQA expansion for simplicity — test with matching heads)
// Q: [B, Hq, S, D], K: [B, Hk, S, D], V: [B, Hk, S, D]
// Returns: [B, Hq, S, D]
mx::array reference_attention(
    const mx::array& Q, const mx::array& K, const mx::array& V,
    float attn_scale) {
  // For validation, use Hq == Hk (no GQA) to keep it simple
  // scores = Q @ K^T * scale  → [B, H, S, S]
  auto scores = mx::matmul(Q, mx::transpose(K, {0, 1, 3, 2}));
  scores = mx::multiply(scores, mx::array(attn_scale));

  // softmax over last dim
  auto max_scores = mx::max(scores, {-1}, true);
  auto exp_scores = mx::exp(mx::subtract(scores, max_scores));
  auto sum_exp = mx::sum(exp_scores, {-1}, true);
  auto weights = mx::divide(exp_scores, sum_exp);

  // output = weights @ V
  return mx::matmul(weights, V);
}

// KL divergence between attention weight distributions
float attention_kl_divergence(
    const mx::array& Q, const mx::array& K_orig, const mx::array& K_recon,
    float attn_scale) {
  auto scores_orig = mx::multiply(
      mx::matmul(Q, mx::transpose(K_orig, {0, 1, 3, 2})),
      mx::array(attn_scale));
  auto scores_recon = mx::multiply(
      mx::matmul(Q, mx::transpose(K_recon, {0, 1, 3, 2})),
      mx::array(attn_scale));

  // Softmax both
  auto softmax = [](const mx::array& x) {
    auto m = mx::max(x, {-1}, true);
    auto e = mx::exp(mx::subtract(x, m));
    auto s = mx::sum(e, {-1}, true);
    return mx::divide(e, s);
  };

  auto p = softmax(scores_orig);
  auto q = softmax(scores_recon);

  // KL(p || q) = sum(p * log(p / q))
  auto kl = mx::sum(mx::multiply(p, mx::log(mx::divide(
      mx::add(p, mx::array(1e-10f)),
      mx::add(q, mx::array(1e-10f))))));

  mx::eval(kl);
  return kl.item<float>();
}

struct TestResult {
  float cosine_sim_k;
  float cosine_sim_v;
  float output_mse;
  float kl_divergence;
  float encode_time_us;
  float decode_time_us;
};

TestResult run_validation(
    int batch, int heads, int seq_len, int head_dim,
    float attn_scale, int bits, bool use_gpu) {
  TestResult result = {};

  turboquant::PolarQuantParams params{head_dim, 0, bits};
  // num_levels computed internally

  // Generate random tensors
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  int total_elems = batch * heads * seq_len * head_dim;
  std::vector<float> q_data(total_elems), k_data(total_elems), v_data(total_elems);
  for (int i = 0; i < total_elems; i++) {
    q_data[i] = dist(gen);
    k_data[i] = dist(gen);
    v_data[i] = dist(gen);
  }

  auto Q = mx::array(q_data.data(), {batch, heads, seq_len, head_dim}, mx::float32);
  auto K = mx::array(k_data.data(), {batch, heads, seq_len, head_dim}, mx::float32);
  auto V = mx::array(v_data.data(), {batch, heads, seq_len, head_dim}, mx::float32);

  // Encode K and V
  auto t0 = std::chrono::high_resolution_clock::now();
  auto [k_angles, k_norms] = [&]() {
    auto r = turboquant::polarquant_encode(K, params);
    return std::make_pair(r[0], r[1]);
  }();
  auto [v_angles, v_norms] = [&]() {
    auto r = turboquant::polarquant_encode(V, params);
    return std::make_pair(r[0], r[1]);
  }();
  mx::eval(k_angles, k_norms, v_angles, v_norms);
  auto t1 = std::chrono::high_resolution_clock::now();
  result.encode_time_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  // Decode K and V
  auto t2 = std::chrono::high_resolution_clock::now();
  auto K_recon = turboquant::polarquant_decode(k_angles, k_norms, params);
  auto V_recon = turboquant::polarquant_decode(v_angles, v_norms, params);
  mx::eval(K_recon, V_recon);
  auto t3 = std::chrono::high_resolution_clock::now();
  result.decode_time_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

  // Compute cosine similarity (average over all vectors)
  {
    auto k_flat = mx::reshape(K, {-1, head_dim});
    auto kr_flat = mx::reshape(K_recon, {-1, head_dim});
    auto v_flat = mx::reshape(V, {-1, head_dim});
    auto vr_flat = mx::reshape(V_recon, {-1, head_dim});
    mx::eval(k_flat, kr_flat, v_flat, vr_flat);

    int n_vecs = batch * heads * seq_len;
    float cos_k_sum = 0.0f, cos_v_sum = 0.0f;
    for (int i = 0; i < n_vecs; i++) {
      cos_k_sum += cosine_similarity(
          k_flat.data<float>() + i * head_dim,
          kr_flat.data<float>() + i * head_dim,
          head_dim);
      cos_v_sum += cosine_similarity(
          v_flat.data<float>() + i * head_dim,
          vr_flat.data<float>() + i * head_dim,
          head_dim);
    }
    result.cosine_sim_k = cos_k_sum / n_vecs;
    result.cosine_sim_v = cos_v_sum / n_vecs;
  }

  // Compute attention output MSE
  {
    auto out_orig = reference_attention(Q, K, V, attn_scale);
    auto out_recon = reference_attention(Q, K_recon, V_recon, attn_scale);
    auto diff = mx::subtract(out_orig, out_recon);
    auto mse_val = mx::mean(mx::multiply(diff, diff));
    mx::eval(mse_val);
    result.output_mse = mse_val.item<float>();
  }

  // KL divergence
  result.kl_divergence = attention_kl_divergence(Q, K, K_recon, attn_scale);

  return result;
}

void print_result(const char* label, const TestResult& r) {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  " << label << ":\n";
  std::cout << "    cosine_sim(K):  " << r.cosine_sim_k << "\n";
  std::cout << "    cosine_sim(V):  " << r.cosine_sim_v << "\n";
  std::cout << "    output_mse:     " << r.output_mse << "\n";
  std::cout << "    kl_divergence:  " << r.kl_divergence << "\n";
  std::cout << "    encode_time_us: " << r.encode_time_us << "\n";
  std::cout << "    decode_time_us: " << r.decode_time_us << "\n";
}

int main() {
  std::cout << "=== PolarQuant Validation for Llama 3.1 ===\n\n";

  const int head_dim = 128;
  const int bits = 4;
  const float llama_scale = 1.0f / std::sqrt(128.0f);  // 0.0884
  const float gemma_scale = 1.0f;  // For comparison

  // Test configurations: (batch, heads, seq_len)
  struct TestConfig {
    const char* name;
    int batch, heads, seq_len;
  };

  TestConfig configs[] = {
    {"small (1,8,128)",   1, 8, 128},
    {"medium (1,8,1024)", 1, 8, 1024},
    {"large (1,8,4096)",  1, 8, 4096},
  };

  for (auto& cfg : configs) {
    std::cout << "--- Config: " << cfg.name << " ---\n";

    std::cout << "\n  Llama attn_scale = " << llama_scale << ":\n";
    auto r_llama = run_validation(
        cfg.batch, cfg.heads, cfg.seq_len, head_dim, llama_scale, bits, false);
    print_result("Llama (scale=0.0884)", r_llama);

    std::cout << "\n  Gemma attn_scale = " << gemma_scale << " (comparison):\n";
    auto r_gemma = run_validation(
        cfg.batch, cfg.heads, cfg.seq_len, head_dim, gemma_scale, bits, false);
    print_result("Gemma (scale=1.0)", r_gemma);

    // Dampening ratio
    float mse_ratio = (r_gemma.output_mse > 1e-12f)
        ? r_llama.output_mse / r_gemma.output_mse
        : 0.0f;
    float kl_ratio = (r_gemma.kl_divergence > 1e-12f)
        ? r_llama.kl_divergence / r_gemma.kl_divergence
        : 0.0f;

    std::cout << "\n  Dampening effect (Llama/Gemma ratio):\n";
    std::cout << "    MSE ratio:  " << mse_ratio
              << " (lower = Llama better)\n";
    std::cout << "    KL ratio:   " << kl_ratio
              << " (lower = Llama better)\n";

    // Pass/fail criteria
    bool pass_cosine = r_llama.cosine_sim_k > 0.95f && r_llama.cosine_sim_v > 0.95f;
    bool pass_kl = r_llama.kl_divergence < 0.1f;
    bool pass_dampening = mse_ratio < 0.5f;

    std::cout << "\n  Verdict:\n";
    std::cout << "    Cosine similarity > 0.95: " << (pass_cosine ? "PASS" : "FAIL") << "\n";
    std::cout << "    KL divergence < 0.1:      " << (pass_kl ? "PASS" : "FAIL") << "\n";
    std::cout << "    MSE dampening < 0.5x:     " << (pass_dampening ? "PASS" : "FAIL") << "\n";
    std::cout << "\n";
  }

  // === Multi-bit comparison ===
  std::cout << "=== Bit-Width Comparison (seq=1024, Llama scale) ===\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Bits | CosSim(K) | CosSim(V) | Output MSE | KL Div    | Bytes/vec | Ratio\n";
  std::cout << "-----|-----------|-----------|------------|-----------|-----------|------\n";

  for (int test_bits : {3, 4, 5, 6, 7, 8}) {
    auto r = run_validation(1, 8, 1024, head_dim, llama_scale, test_bits, false);
    int n_angles = head_dim - 1;
    float bytes_per_vec = n_angles * test_bits / 8.0f + 4.0f;  // packed angles + float32 norm
    float ratio = (head_dim * 2.0f) / bytes_per_vec;  // vs FP16

    std::cout << std::setw(4) << test_bits
              << " | " << std::setw(9) << r.cosine_sim_k
              << " | " << std::setw(9) << r.cosine_sim_v
              << " | " << std::setw(10) << r.output_mse
              << " | " << std::setw(9) << r.kl_divergence
              << " | " << std::setw(9) << bytes_per_vec
              << " | " << std::setprecision(2) << ratio << "x\n";
    std::cout << std::setprecision(6);
  }

  // Compression ratio analysis
  std::cout << "\n=== Compression Analysis (4-bit) ===\n";
  int n_angles = head_dim - 1;
  float bytes_fp16 = head_dim * 2.0f;
  float bytes_polarquant = n_angles * 0.5f + 4.0f;
  float ratio = bytes_fp16 / bytes_polarquant;
  std::cout << "  FP16 per vector:       " << bytes_fp16 << " bytes\n";
  std::cout << "  PolarQuant per vector: " << bytes_polarquant << " bytes\n";
  std::cout << "  Compression ratio:     " << std::setprecision(2) << ratio << "x\n";
  std::cout << "\n  KV cache at 128K context (8 KV heads, 80 layers):\n";
  float kv_fp16 = 80.0f * 8 * 131072 * 128 * 2 * 2 / (1024.0f * 1024.0f * 1024.0f);
  float kv_polar = kv_fp16 / ratio;
  std::cout << "    FP16:       " << std::setprecision(1) << kv_fp16 << " GB\n";
  std::cout << "    PolarQuant: " << kv_polar << " GB\n";

  std::cout << "\n=== Done ===\n";
  return 0;
}
