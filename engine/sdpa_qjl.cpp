#include <dlfcn.h>
#include <filesystem>
#include <iostream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "engine/sdpa_qjl.h"

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

mx::array sdpa_qjl(
    const mx::array& Q_sketches,
    const mx::array& Q_norms,
    const mx::array& K_sketches,
    const mx::array& K_norms,
    const mx::array& V_quant,
    const mx::array& V_scales,
    const mx::array& V_zeros,
    const SdpaQJLParams& params,
    mx::StreamOrDevice s) {
  auto stream = to_stream(s);

  if (Q_sketches.ndim() != 4) {
    throw std::invalid_argument("sdpa_qjl: Q_sketches must be 4D [B, Hq, Sq, m/32]");
  }
  if (K_sketches.ndim() != 4) {
    throw std::invalid_argument("sdpa_qjl: K_sketches must be 4D [B, Hkv, Skv, m/32]");
  }
  if (V_quant.ndim() != 4) {
    throw std::invalid_argument("sdpa_qjl: V_quant must be 4D [B, Hkv, Skv, D/2]");
  }

  int B = Q_sketches.shape(0);
  int Hq = Q_sketches.shape(1);
  int Sq = Q_sketches.shape(2);
  int D = params.head_dim;
  int Skv = K_sketches.shape(2);

  if (Hq != params.num_q_heads) {
    throw std::invalid_argument("sdpa_qjl: Q_sketches heads must equal num_q_heads");
  }
  if (params.num_q_heads % params.num_kv_heads != 0) {
    throw std::invalid_argument("sdpa_qjl: num_q_heads must be divisible by num_kv_heads");
  }

  const int SPLIT_K_THRESHOLD = 512;
  const int CHUNK_SIZE = 512;

  if (Skv <= SPLIT_K_THRESHOLD) {
    return mx::array(
        {B, Hq, Sq, D},
        mx::float16,
        std::make_shared<SdpaQJL>(stream, params),
        {Q_sketches, Q_norms, K_sketches, K_norms,
         V_quant, V_scales, V_zeros});
  }

  // Split-K path
  int num_chunks = (Skv + CHUNK_SIZE - 1) / CHUNK_SIZE;

  auto partials = mx::array::make_arrays(
      {{B * Hq * Sq * num_chunks * D}, {B * Hq * Sq * num_chunks}, {B * Hq * Sq * num_chunks}},
      {mx::float32, mx::float32, mx::float32},
      std::make_shared<SdpaQJLPartial>(stream, params, num_chunks, CHUNK_SIZE),
      {Q_sketches, Q_norms, K_sketches, K_norms,
       V_quant, V_scales, V_zeros});

  return mx::array(
      {B, Hq, Sq, D},
      mx::float16,
      std::make_shared<SdpaQJLReduce>(stream, params, num_chunks),
      {partials[0], partials[1], partials[2]});
}

///////////////////////////////////////////////////////////////////////////////
// CPU eval -- not implemented (GPU-only kernel)
///////////////////////////////////////////////////////////////////////////////

void SdpaQJL::eval_cpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("SdpaQJL is GPU-only. Use a Metal-capable device.");
}

///////////////////////////////////////////////////////////////////////////////
// GPU eval (Metal)
///////////////////////////////////////////////////////////////////////////////

#ifdef _METAL_

void SdpaQJL::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& Q_sk = inputs[0];
  auto& Q_norms = inputs[1];
  auto& K_sk = inputs[2];
  auto& K_norms = inputs[3];
  auto& V_quant = inputs[4];
  auto& V_scales = inputs[5];
  auto& V_zeros = inputs[6];
  auto& output = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  output.set_data(mx::allocator::malloc(output.nbytes()));

  int B = Q_sk.shape(0);
  int Hq = params_.num_q_heads;
  int Sq = Q_sk.shape(2);
  int Skv = K_sk.shape(2);

  // Metal-side params struct must match the Metal shader layout exactly
  struct MetalParams {
    int num_q_heads;
    int num_kv_heads;
    int seq_len_q;
    int seq_len_kv;
    int sketch_dim;
    int sketch_words;
    int v_group_size;
    float attn_scale;
  };

  MetalParams mp;
  mp.num_q_heads = params_.num_q_heads;
  mp.num_kv_heads = params_.num_kv_heads;
  mp.seq_len_q = Sq;
  mp.seq_len_kv = Skv;
  mp.sketch_dim = params_.sketch_dim;
  mp.sketch_words = params_.sketch_dim / 32;
  mp.v_group_size = params_.v_group_size;
  mp.attn_scale = params_.attn_scale;

  auto lib = d.get_library("turboquant", current_binary_dir());
  auto kernel = d.get_kernel("sdpa_qjl_kernel", lib);
  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(Q_sk, 0);
  enc.set_input_array(Q_norms, 1);
  enc.set_input_array(K_sk, 2);
  enc.set_input_array(K_norms, 3);
  enc.set_input_array(V_quant, 4);
  enc.set_input_array(V_scales, 5);
  enc.set_input_array(V_zeros, 6);
  enc.set_output_array(output, 7);
  enc.set_bytes(mp, 8);

  int total_q = Hq * Sq;
  MTL::Size grid_dims = MTL::Size(total_q, B, 1);
  MTL::Size group_dims = MTL::Size(32, 1, 1);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

#else

void SdpaQJL::eval_gpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("SdpaQJL: Metal not available.");
}

#endif

bool SdpaQJL::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const SdpaQJL&>(other);
  return params_.head_dim == o.params_.head_dim &&
         params_.sketch_dim == o.params_.sketch_dim &&
         params_.num_q_heads == o.params_.num_q_heads &&
         params_.num_kv_heads == o.params_.num_kv_heads &&
         params_.v_group_size == o.params_.v_group_size &&
         params_.attn_scale == o.params_.attn_scale;
}

///////////////////////////////////////////////////////////////////////////////
// SdpaQJLPartial -- Split-K Phase 1
///////////////////////////////////////////////////////////////////////////////

void SdpaQJLPartial::eval_cpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("SdpaQJLPartial is GPU-only.");
}

#ifdef _METAL_
void SdpaQJLPartial::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& Q_sk = inputs[0];
  auto& Q_norms_arr = inputs[1];
  auto& K_sk = inputs[2];
  auto& K_norms_arr = inputs[3];
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

  int B = Q_sk.shape(0);
  int Hq = params_.num_q_heads;
  int Sq = Q_sk.shape(2);
  int Skv = K_sk.shape(2);

  struct MetalParams {
    int num_q_heads; int num_kv_heads; int seq_len_q; int seq_len_kv;
    int sketch_dim; int sketch_words; int v_group_size; float attn_scale;
  };
  MetalParams mp{params_.num_q_heads, params_.num_kv_heads, Sq, Skv,
                 params_.sketch_dim, params_.sketch_dim / 32,
                 params_.v_group_size, params_.attn_scale};

  struct SplitKParams { int num_chunks; int chunk_size; };
  SplitKParams sk{num_chunks_, chunk_size_};

  auto lib = d.get_library("turboquant", current_binary_dir());
  auto kernel = d.get_kernel("sdpa_qjl_partial", lib);
  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(Q_sk, 0);
  enc.set_input_array(Q_norms_arr, 1);
  enc.set_input_array(K_sk, 2);
  enc.set_input_array(K_norms_arr, 3);
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
void SdpaQJLPartial::eval_gpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("No Metal.");
}
#endif

bool SdpaQJLPartial::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const SdpaQJLPartial&>(other);
  return params_.head_dim == o.params_.head_dim &&
         params_.sketch_dim == o.params_.sketch_dim &&
         num_chunks_ == o.num_chunks_ && chunk_size_ == o.chunk_size_;
}

///////////////////////////////////////////////////////////////////////////////
// SdpaQJLReduce -- Split-K Phase 2
///////////////////////////////////////////////////////////////////////////////

void SdpaQJLReduce::eval_cpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("SdpaQJLReduce is GPU-only.");
}

#ifdef _METAL_
void SdpaQJLReduce::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& partial_out = inputs[0];
  auto& partial_max = inputs[1];
  auto& partial_sum_exp = inputs[2];
  auto& output = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  output.set_data(mx::allocator::malloc(output.nbytes()));

  int B = output.shape(0);
  int Hq = params_.num_q_heads;
  int Sq = output.shape(2);

  struct MetalParams {
    int num_q_heads; int num_kv_heads; int seq_len_q; int seq_len_kv;
    int sketch_dim; int sketch_words; int v_group_size; float attn_scale;
  };
  MetalParams mp{params_.num_q_heads, params_.num_kv_heads, Sq, 0,
                 params_.sketch_dim, params_.sketch_dim / 32,
                 params_.v_group_size, params_.attn_scale};

  struct SplitKParams { int num_chunks; int chunk_size; };
  SplitKParams sk{num_chunks_, 0};

  auto lib = d.get_library("turboquant", current_binary_dir());
  auto kernel = d.get_kernel("sdpa_qjl_reduce", lib);
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
void SdpaQJLReduce::eval_gpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("No Metal.");
}
#endif

bool SdpaQJLReduce::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const SdpaQJLReduce&>(other);
  return params_.head_dim == o.params_.head_dim &&
         params_.sketch_dim == o.params_.sketch_dim &&
         num_chunks_ == o.num_chunks_;
}

}  // namespace turboquant
