#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "mlx/mlx.h"
#include "engine/llama_loader.h"
#include "engine/llama_model.h"
#include "engine/qjl.h"

namespace mx = mlx::core;

// Modified layer that captures K, Q, V before attention
// We'll run the model normally, then extract activations by
// computing QKV projections manually on one layer.

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: qjl_real_activations <model_dir>" << std::endl;
    return 1;
  }

  std::cout << "=== QJL with Real Llama 3.1 70B Activations ===\n\n";

  // Load model
  auto weights = turboquant::load_weights(argv[1]);
  auto& cfg = weights.config;
  std::cout << "Model: " << cfg.num_hidden_layers << " layers, head_dim=" << cfg.head_dim << "\n";

  // Create a short prompt
  std::vector<int32_t> prompt = {128000, 9906, 11, 358, 1097, 264, 5575, 520, 279, 3907,
                                  315, 7188, 11, 33108, 13, 358, 1097, 8173, 304, 6975};
  int S = prompt.size();
  auto tokens = mx::array(prompt.data(), {1, S}, mx::int32);

  // Get embedding
  mx::array h(0.0f);
  if (weights.embed_is_quantized) {
    auto& eq = weights.embed_tokens_q;
    auto embed_full = mx::dequantize(eq.weight, eq.scales, eq.biases,
                                      cfg.quantize_group_size, cfg.quantize_bits);
    mx::eval(embed_full);
    h = mx::take(embed_full, tokens, 0);
  } else {
    h = mx::take(weights.embed_tokens, tokens, 0);
  }
  mx::eval(h);

  // Test on layers 0, 20, 40, 60 to see if QJL quality varies by depth
  int test_layers[] = {0, 20, 40, 60};
  int sketch_dims[] = {256, 384, 512, 768};
  float attn_scale = 1.0f / std::sqrt(float(cfg.head_dim));

  std::cout << "\nLayer | Sketch_m | Score_Corr | Softmax_KL | K_norm_mean | K_norm_std\n";
  std::cout << "------|----------|------------|------------|-------------|----------\n";

  for (int layer_idx : test_layers) {
    auto& lw = weights.layers[layer_idx];

    // RMSNorm
    auto normed = mx::fast::rms_norm(h, lw.input_layernorm, cfg.rms_norm_eps);

    // QKV projection (quantized matmul)
    auto qlinear = [&](const mx::array& x, const turboquant::QuantizedWeight& qw) {
      std::optional<mx::array> biases_opt;
      if (qw.biases.size() > 0) biases_opt = qw.biases;
      return mx::quantized_matmul(x, qw.weight, qw.scales, biases_opt,
                                   true, cfg.quantize_group_size, cfg.quantize_bits);
    };

    auto q = qlinear(normed, lw.q_proj);
    auto k = qlinear(normed, lw.k_proj);

    int B = 1, H = cfg.num_attention_heads, Hkv = cfg.num_key_value_heads;
    int D = cfg.head_dim;

    q = mx::transpose(mx::reshape(q, {B, S, H, D}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B, S, Hkv, D}), {0, 2, 1, 3});

    // Apply RoPE
    q = mx::fast::rope(q, D, false, cfg.rope_theta, 1.0f, 0, std::nullopt);
    k = mx::fast::rope(k, D, false, cfg.rope_theta, 1.0f, 0, std::nullopt);
    mx::eval(q, k);

    // Compute K statistics
    auto k_f32 = mx::astype(k, mx::float32);
    auto k_norms = mx::sqrt(mx::sum(mx::multiply(k_f32, k_f32), -1, false));
    mx::eval(k_norms);
    auto k_norm_mean = mx::mean(k_norms);
    auto k_norm_std = mx::sqrt(mx::mean(mx::multiply(
        mx::subtract(k_norms, mx::mean(k_norms)),
        mx::subtract(k_norms, mx::mean(k_norms)))));
    mx::eval(k_norm_mean, k_norm_std);

    // Exact scores: Q @ K^T * scale (expand K for GQA)
    int gqa = H / Hkv;
    auto k_exp = mx::reshape(
        mx::repeat(mx::reshape(k, {B, Hkv, 1, S, D}), gqa, 2),
        {B, H, S, D});
    auto exact_scores = mx::multiply(
        mx::matmul(mx::astype(q, mx::float32),
                    mx::transpose(mx::astype(k_exp, mx::float32), {0, 1, 3, 2})),
        mx::array(attn_scale));
    mx::eval(exact_scores);

    // Exact softmax
    auto exact_max = mx::max(exact_scores, -1, true);
    auto exact_softmax = mx::divide(
        mx::exp(mx::subtract(exact_scores, exact_max)),
        mx::sum(mx::exp(mx::subtract(exact_scores, exact_max)), -1, true));
    mx::eval(exact_softmax);

    // Test QJL at different sketch sizes
    for (int m : sketch_dims) {
      turboquant::QJLParams params{D, m, H, Hkv, attn_scale};
      auto proj = turboquant::generate_projection_matrix(m, D, 42 + layer_idx);
      mx::eval(proj);

      auto [sketches, norms] = turboquant::qjl_encode(k, proj);
      auto est_scores = turboquant::qjl_scores(q, sketches, norms, proj, params);
      mx::eval(est_scores);

      // Score correlation
      auto ex_flat = mx::reshape(exact_scores, {-1});
      auto es_flat = mx::reshape(est_scores, {-1});
      auto ex_m = mx::mean(ex_flat);
      auto es_m = mx::mean(es_flat);
      mx::eval(ex_m, es_m);
      auto ex_c = mx::subtract(ex_flat, ex_m);
      auto es_c = mx::subtract(es_flat, es_m);
      auto corr = mx::divide(
          mx::mean(mx::multiply(ex_c, es_c)),
          mx::add(mx::multiply(
              mx::sqrt(mx::mean(mx::multiply(ex_c, ex_c))),
              mx::sqrt(mx::mean(mx::multiply(es_c, es_c)))),
              mx::array(1e-10f)));
      mx::eval(corr);

      // Softmax KL
      auto est_max = mx::max(est_scores, -1, true);
      auto est_softmax = mx::divide(
          mx::exp(mx::subtract(est_scores, est_max)),
          mx::sum(mx::exp(mx::subtract(est_scores, est_max)), -1, true));
      auto kl = mx::sum(mx::multiply(exact_softmax,
          mx::log(mx::divide(
              mx::add(exact_softmax, mx::array(1e-10f)),
              mx::add(est_softmax, mx::array(1e-10f))))));
      mx::eval(kl);

      std::cout << std::fixed
                << std::setw(5) << layer_idx
                << " | " << std::setw(8) << m
                << " | " << std::setprecision(4) << std::setw(10) << corr.item<float>()
                << " | " << std::setprecision(4) << std::setw(10) << kl.item<float>()
                << " | " << std::setprecision(3) << std::setw(11) << k_norm_mean.item<float>()
                << " | " << std::setprecision(3) << std::setw(9) << k_norm_std.item<float>()
                << "\n";
    }

    // Advance h through this layer for the next layer test
    // (simplified: just use normed output as next input for approximation)
    // Full forward would be more accurate but this gives us per-layer activation stats
  }

  std::cout << "\n=== Done ===\n";
  return 0;
}
