#include <dlfcn.h>
#include <filesystem>
#include <iostream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "engine/sdpa_polar.h"

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

mx::array sdpa_polar(
    const mx::array& Q,
    const mx::array& K_angles, const mx::array& K_norms,
    const mx::array& V_angles, const mx::array& V_norms,
    const SdpaPolarParams& params,
    mx::StreamOrDevice s) {
  auto stream = to_stream(s);

  if (Q.ndim() != 4) {
    throw std::invalid_argument("sdpa_polar: Q must be 4D [B, Hq, Sq, D]");
  }
  if (K_angles.ndim() != 4) {
    throw std::invalid_argument("sdpa_polar: K_angles must be 4D [B, Hkv, Skv, D-1]");
  }

  int B = Q.shape(0);
  int Hq = Q.shape(1);
  int Sq = Q.shape(2);
  int D = Q.shape(3);
  int Skv = K_angles.shape(2);
  int n_angles = D - 1;

  if (D != params.head_dim) {
    throw std::invalid_argument("sdpa_polar: Q last dim must equal head_dim");
  }
  if (Hq != params.num_q_heads) {
    throw std::invalid_argument("sdpa_polar: Q heads must equal num_q_heads");
  }
  if (params.num_q_heads % params.num_kv_heads != 0) {
    throw std::invalid_argument("sdpa_polar: num_q_heads must be divisible by num_kv_heads");
  }
  if (K_angles.shape(3) != n_angles) {
    throw std::invalid_argument("sdpa_polar: K_angles last dim must be D-1");
  }

  auto Q_f16 = mx::astype(Q, mx::float16, s);

  return mx::array(
      {B, Hq, Sq, D},
      mx::float16,
      std::make_shared<SdpaPolar>(stream, params),
      {Q_f16, K_angles, K_norms, V_angles, V_norms});
}

///////////////////////////////////////////////////////////////////////////////
// CPU eval — not implemented (GPU-only kernel)
///////////////////////////////////////////////////////////////////////////////

void SdpaPolar::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error(
      "SdpaPolar is GPU-only. Use a Metal-capable device.");
}

///////////////////////////////////////////////////////////////////////////////
// GPU eval (Metal)
///////////////////////////////////////////////////////////////////////////////

#ifdef _METAL_

void SdpaPolar::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& Q_arr = inputs[0];
  auto& K_angles = inputs[1];
  auto& K_norms = inputs[2];
  auto& V_angles = inputs[3];
  auto& V_norms = inputs[4];
  auto& output = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  output.set_data(mx::allocator::malloc(output.nbytes()));

  int B = Q_arr.shape(0);
  int Hq = params_.num_q_heads;
  int Sq = Q_arr.shape(2);
  int Skv = K_angles.shape(2);

  // Must match SdpaPolarParams layout in the Metal shader
  struct MetalParams {
    int num_q_heads;
    int num_kv_heads;
    int seq_len_q;
    int seq_len_kv;
    int bits;
    float attn_scale;
  };

  MetalParams metal_params;
  metal_params.num_q_heads = params_.num_q_heads;
  metal_params.num_kv_heads = params_.num_kv_heads;
  metal_params.seq_len_q = Sq;
  metal_params.seq_len_kv = Skv;
  metal_params.bits = params_.bits;
  metal_params.attn_scale = params_.attn_scale;

  auto lib = d.get_library("turboquant", current_binary_dir());
  auto kernel = d.get_kernel("sdpa_polar_kernel", lib);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(Q_arr, 0);
  compute_encoder.set_input_array(K_angles, 1);
  compute_encoder.set_input_array(K_norms, 2);
  compute_encoder.set_input_array(V_angles, 3);
  compute_encoder.set_input_array(V_norms, 4);
  compute_encoder.set_output_array(output, 5);
  compute_encoder.set_bytes(metal_params, 6);

  int total_q = Hq * Sq;
  MTL::Size grid_dims = MTL::Size(total_q, B, 1);
  MTL::Size group_dims = MTL::Size(32, 1, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

#else

void SdpaPolar::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("SdpaPolar has no GPU implementation (Metal not available).");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Equivalence
///////////////////////////////////////////////////////////////////////////////

bool SdpaPolar::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const SdpaPolar&>(other);
  return params_.head_dim == o.params_.head_dim &&
         params_.num_q_heads == o.params_.num_q_heads &&
         params_.num_kv_heads == o.params_.num_kv_heads &&
         params_.bits == o.params_.bits &&
         params_.attn_scale == o.params_.attn_scale;
}

}  // namespace turboquant
