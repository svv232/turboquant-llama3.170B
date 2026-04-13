#include <dlfcn.h>
#include <filesystem>
#include <iostream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "engine/sdpa_int4.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace turboquant {

// Reuse the binary dir locator from polarquant.cpp
extern std::string current_binary_dir();

///////////////////////////////////////////////////////////////////////////////
// Public API
///////////////////////////////////////////////////////////////////////////////

mx::array sdpa_int4(
    const mx::array& Q,
    const mx::array& K_quant, const mx::array& K_scales, const mx::array& K_zeros,
    const mx::array& V_quant, const mx::array& V_scales, const mx::array& V_zeros,
    const SdpaInt4Params& params,
    mx::StreamOrDevice s) {
  auto stream = to_stream(s);

  if (Q.ndim() != 4) {
    throw std::invalid_argument("sdpa_int4: Q must be 4D [B, Hq, Sq, D]");
  }
  if (K_quant.ndim() != 4) {
    throw std::invalid_argument("sdpa_int4: K_quant must be 4D [B, Hkv, Skv, D/2]");
  }

  int B = Q.shape(0);
  int Hq = Q.shape(1);
  int Sq = Q.shape(2);
  int D = Q.shape(3);
  int Skv = K_quant.shape(2);

  if (D != params.head_dim) {
    throw std::invalid_argument("sdpa_int4: Q last dim must equal head_dim");
  }
  if (Hq != params.num_q_heads) {
    throw std::invalid_argument("sdpa_int4: Q heads must equal num_q_heads");
  }
  if (params.num_q_heads % params.num_kv_heads != 0) {
    throw std::invalid_argument("sdpa_int4: num_q_heads must be divisible by num_kv_heads");
  }

  auto Q_f16 = mx::astype(Q, mx::float16, s);

  const int SPLIT_K_THRESHOLD = 512;
  const int CHUNK_SIZE = 512;

  if (Skv <= SPLIT_K_THRESHOLD) {
    // Single-pass for short sequences
    return mx::array(
        {B, Hq, Sq, D},
        mx::float16,
        std::make_shared<SdpaInt4>(stream, params),
        {Q_f16, K_quant, K_scales, K_zeros, V_quant, V_scales, V_zeros});
  }

  // Split-K: two separate primitives chained via MLX graph
  int num_chunks = (Skv + CHUNK_SIZE - 1) / CHUNK_SIZE;

  // Phase 1: produces 3 outputs (partial_out, partial_max, partial_sum_exp)
  auto partials = mx::array::make_arrays(
      {{B * Hq * Sq * num_chunks * D}, {B * Hq * Sq * num_chunks}, {B * Hq * Sq * num_chunks}},
      {mx::float32, mx::float32, mx::float32},
      std::make_shared<SdpaInt4Partial>(stream, params, num_chunks, CHUNK_SIZE),
      {Q_f16, K_quant, K_scales, K_zeros, V_quant, V_scales, V_zeros});

  // Phase 2: reduce partials → final output
  // MLX evaluates Phase 1 first because its outputs are inputs to Phase 2
  return mx::array(
      {B, Hq, Sq, D},
      mx::float16,
      std::make_shared<SdpaInt4Reduce>(stream, params, num_chunks),
      {partials[0], partials[1], partials[2]});
}

///////////////////////////////////////////////////////////////////////////////
// CPU eval — not implemented (GPU-only kernel)
///////////////////////////////////////////////////////////////////////////////

void SdpaInt4::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error(
      "SdpaInt4 is GPU-only. Use a Metal-capable device.");
}

///////////////////////////////////////////////////////////////////////////////
// GPU eval (Metal)
///////////////////////////////////////////////////////////////////////////////

#ifdef _METAL_

void SdpaInt4::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& Q_arr = inputs[0];
  auto& K_quant = inputs[1];
  auto& K_scales = inputs[2];
  auto& K_zeros = inputs[3];
  auto& V_quant = inputs[4];
  auto& V_scales = inputs[5];
  auto& V_zeros = inputs[6];
  auto& output = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  output.set_data(mx::allocator::malloc(output.nbytes()));

  int B = Q_arr.shape(0);
  int Hq = params_.num_q_heads;
  int Sq = Q_arr.shape(2);
  int Skv = K_quant.shape(2);

  struct MetalParams {
    int num_q_heads;
    int num_kv_heads;
    int seq_len_q;
    int seq_len_kv;
    int group_size;
    float attn_scale;
  };

  MetalParams metal_params;
  metal_params.num_q_heads = params_.num_q_heads;
  metal_params.num_kv_heads = params_.num_kv_heads;
  metal_params.seq_len_q = Sq;
  metal_params.seq_len_kv = Skv;
  metal_params.group_size = params_.group_size;
  metal_params.attn_scale = params_.attn_scale;

  auto lib = d.get_library("turboquant", current_binary_dir());

  // Single-pass kernel — correct and simple.
  // Split-K kernels are defined in the .metal file but need a Metal fence/barrier
  // between phases which requires further MLX integration work.
  auto kernel = d.get_kernel("sdpa_int4_kernel", lib);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(Q_arr, 0);
  compute_encoder.set_input_array(K_quant, 1);
  compute_encoder.set_input_array(K_scales, 2);
  compute_encoder.set_input_array(K_zeros, 3);
  compute_encoder.set_input_array(V_quant, 4);
  compute_encoder.set_input_array(V_scales, 5);
  compute_encoder.set_input_array(V_zeros, 6);
  compute_encoder.set_output_array(output, 7);
  compute_encoder.set_bytes(metal_params, 8);

  int total_q = Hq * Sq;
  MTL::Size grid_dims = MTL::Size(total_q, B, 1);
  MTL::Size group_dims = MTL::Size(32, 1, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

#else

void SdpaInt4::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("SdpaInt4 has no GPU implementation (Metal not available).");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Equivalence
///////////////////////////////////////////////////////////////////////////////

bool SdpaInt4::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const SdpaInt4&>(other);
  return params_.head_dim == o.params_.head_dim &&
         params_.num_q_heads == o.params_.num_q_heads &&
         params_.num_kv_heads == o.params_.num_kv_heads &&
         params_.group_size == o.params_.group_size &&
         params_.attn_scale == o.params_.attn_scale;
}

///////////////////////////////////////////////////////////////////////////////
// SdpaInt4Partial — Split-K Phase 1
///////////////////////////////////////////////////////////////////////////////

void SdpaInt4Partial::eval_cpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("SdpaInt4Partial is GPU-only.");
}

#ifdef _METAL_
void SdpaInt4Partial::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& Q_arr = inputs[0];
  auto& K_quant = inputs[1];
  auto& K_scales = inputs[2];
  auto& K_zeros = inputs[3];
  auto& V_quant = inputs[4];
  auto& V_scales = inputs[5];
  auto& V_zeros = inputs[6];
  auto& partial_out = outputs[0];
  auto& partial_max = outputs[1];
  auto& partial_sum_exp = outputs[2];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  partial_out.set_data(mx::allocator::malloc(partial_out.nbytes()));
  partial_max.set_data(mx::allocator::malloc(partial_max.nbytes()));
  partial_sum_exp.set_data(mx::allocator::malloc(partial_sum_exp.nbytes()));

  int B = Q_arr.shape(0);
  int Hq = params_.num_q_heads;
  int Sq = Q_arr.shape(2);
  int Skv = K_quant.shape(2);

  struct MetalParams {
    int num_q_heads; int num_kv_heads; int seq_len_q; int seq_len_kv;
    int group_size; float attn_scale;
  };
  MetalParams mp{params_.num_q_heads, params_.num_kv_heads, Sq, Skv,
                 params_.group_size, params_.attn_scale};

  struct SplitKParams { int num_chunks; int chunk_size; };
  SplitKParams sk{num_chunks_, chunk_size_};

  auto lib = d.get_library("turboquant", current_binary_dir());
  auto kernel = d.get_kernel("sdpa_int4_partial", lib);
  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(Q_arr, 0);
  enc.set_input_array(K_quant, 1);
  enc.set_input_array(K_scales, 2);
  enc.set_input_array(K_zeros, 3);
  enc.set_input_array(V_quant, 4);
  enc.set_input_array(V_scales, 5);
  enc.set_input_array(V_zeros, 6);
  enc.set_output_array(partial_out, 7);
  enc.set_output_array(partial_max, 8);
  enc.set_output_array(partial_sum_exp, 9);
  enc.set_bytes(mp, 10);
  enc.set_bytes(sk, 11);

  int total_q = Hq * Sq;
  MTL::Size grid_dims = MTL::Size(total_q * num_chunks_, B, 1);
  MTL::Size group_dims = MTL::Size(32, 1, 1);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}
#else
void SdpaInt4Partial::eval_gpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("No Metal.");
}
#endif

bool SdpaInt4Partial::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const SdpaInt4Partial&>(other);
  return params_.head_dim == o.params_.head_dim &&
         num_chunks_ == o.num_chunks_ && chunk_size_ == o.chunk_size_;
}

///////////////////////////////////////////////////////////////////////////////
// SdpaInt4Reduce — Split-K Phase 2
///////////////////////////////////////////////////////////////////////////////

void SdpaInt4Reduce::eval_cpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("SdpaInt4Reduce is GPU-only.");
}

#ifdef _METAL_
void SdpaInt4Reduce::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& partial_out = inputs[0];
  auto& partial_max = inputs[1];
  auto& partial_sum_exp = inputs[2];
  auto& output = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  output.set_data(mx::allocator::malloc(output.nbytes()));

  // Recover B, Hq, Sq from output shape
  int B = output.shape(0);
  int Hq = params_.num_q_heads;
  int Sq = output.shape(2);

  struct MetalParams {
    int num_q_heads; int num_kv_heads; int seq_len_q; int seq_len_kv;
    int group_size; float attn_scale;
  };
  MetalParams mp{params_.num_q_heads, params_.num_kv_heads, Sq, 0,
                 params_.group_size, params_.attn_scale};

  struct SplitKParams { int num_chunks; int chunk_size; };
  SplitKParams sk{num_chunks_, 0};

  auto lib = d.get_library("turboquant", current_binary_dir());
  auto kernel = d.get_kernel("sdpa_int4_reduce", lib);
  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(partial_out, 0);
  enc.set_input_array(partial_max, 1);
  enc.set_input_array(partial_sum_exp, 2);
  enc.set_output_array(output, 3);
  enc.set_bytes(mp, 4);
  enc.set_bytes(sk, 5);

  int total_q = Hq * Sq;
  MTL::Size grid_dims = MTL::Size(total_q, B, 1);
  MTL::Size group_dims = MTL::Size(32, 1, 1);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}
#else
void SdpaInt4Reduce::eval_gpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("No Metal.");
}
#endif

bool SdpaInt4Reduce::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const SdpaInt4Reduce&>(other);
  return params_.head_dim == o.params_.head_dim && num_chunks_ == o.num_chunks_;
}

}  // namespace turboquant
