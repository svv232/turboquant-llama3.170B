#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "engine/polarquant.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace turboquant {

// Find directory containing this binary (for locating .metallib)
std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

// Total number of quantized angles for head_dim D with recursive pairing:
// Level 0: D/2 pairs → D/2 angles, D/2 radii
// Level 1: D/4 pairs → D/4 angles, D/4 radii
// ...
// Level log2(D)-1: 1 pair → 1 angle, 1 radius (the norm)
// Total angles = D/2 + D/4 + ... + 1 = D - 1
static int total_angles(int head_dim) {
  return head_dim - 1;
}

// ---- CPU Encode Implementation ----
static void polarquant_encode_cpu(
    const float* input,      // [B, H, S, D] contiguous
    uint8_t* angles_out,     // [B, H, S, D-1] packed nibbles
    float* norms_out,        // [B, H, S, 1]
    int batch, int heads, int seq_len, int head_dim,
    int bits) {
  const int D = head_dim;
  const int n_angles = total_angles(D);
  const int n_bins = 1 << bits;  // 16 for 4-bit
  const float bin_width = (2.0f * M_PI) / n_bins;

  // Temp buffers for recursive decomposition of one vector
  std::vector<float> radii_cur(D);
  std::vector<float> radii_next(D / 2);

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < heads; h++) {
      for (int s = 0; s < seq_len; s++) {
        const float* vec = input + ((b * heads + h) * seq_len + s) * D;
        uint8_t* ang = angles_out + ((b * heads + h) * seq_len + s) * n_angles;
        float* nrm = norms_out + ((b * heads + h) * seq_len + s);

        // Copy input to radii buffer for level 0
        for (int i = 0; i < D; i++) {
          radii_cur[i] = vec[i];
        }

        int angle_offset = 0;
        int cur_dim = D;

        // Recursive polar decomposition
        while (cur_dim > 1) {
          int n_pairs = cur_dim / 2;
          for (int p = 0; p < n_pairs; p++) {
            float x0 = radii_cur[2 * p];
            float x1 = radii_cur[2 * p + 1];

            // Polar decomposition
            float r = std::sqrt(x0 * x0 + x1 * x1);
            float theta = std::atan2(x1, x0);  // [-π, π)

            // Quantize angle to n_bins levels
            // Map [-π, π) → [0, n_bins)
            float normalized = (theta + M_PI) / bin_width;
            int bin = static_cast<int>(std::floor(normalized));
            if (bin >= n_bins) bin = n_bins - 1;
            if (bin < 0) bin = 0;

            ang[angle_offset + p] = static_cast<uint8_t>(bin);
            radii_next[p] = r;
          }

          angle_offset += n_pairs;
          cur_dim = n_pairs;

          // Swap buffers
          for (int i = 0; i < cur_dim; i++) {
            radii_cur[i] = radii_next[i];
          }
        }

        // Final radius is the vector norm
        *nrm = radii_cur[0];
      }
    }
  }
}

// ---- CPU Decode Implementation ----
static void polarquant_decode_cpu(
    const uint8_t* angles_in,  // [B, H, S, D-1]
    const float* norms_in,     // [B, H, S, 1]
    float* output,             // [B, H, S, D]
    int batch, int heads, int seq_len, int head_dim,
    int bits) {
  const int D = head_dim;
  const int n_angles = total_angles(D);
  const int n_bins = 1 << bits;
  const float bin_width = (2.0f * M_PI) / n_bins;

  // Compute number of recursion levels
  int n_levels = 0;
  {
    int d = D;
    while (d > 1) { d /= 2; n_levels++; }
  }

  // Precompute angle offsets for each level
  std::vector<int> level_offset(n_levels);
  std::vector<int> level_pairs(n_levels);
  {
    int off = 0;
    int d = D;
    for (int l = 0; l < n_levels; l++) {
      level_pairs[l] = d / 2;
      level_offset[l] = off;
      off += d / 2;
      d /= 2;
    }
  }

  // Temp buffer for reconstruction
  std::vector<float> radii(D);

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < heads; h++) {
      for (int s = 0; s < seq_len; s++) {
        const uint8_t* ang = angles_in + ((b * heads + h) * seq_len + s) * n_angles;
        float norm = norms_in[(b * heads + h) * seq_len + s];
        float* out = output + ((b * heads + h) * seq_len + s) * D;

        // Start from the root: radii[0] = norm
        radii[0] = norm;

        // Reconstruct from deepest level back to level 0
        for (int l = n_levels - 1; l >= 0; l--) {
          int n_pairs = level_pairs[l];
          int off = level_offset[l];

          // Expand: each radius → pair of values using dequantized angle
          // Process in reverse to avoid overwriting
          for (int p = n_pairs - 1; p >= 0; p--) {
            float r = radii[p];
            uint8_t bin = ang[off + p];

            // Dequantize: use bin center
            float theta = (static_cast<float>(bin) + 0.5f) * bin_width - M_PI;

            radii[2 * p] = r * std::cos(theta);
            radii[2 * p + 1] = r * std::sin(theta);
          }
        }

        // Copy reconstructed vector to output
        for (int i = 0; i < D; i++) {
          out[i] = radii[i];
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// MLX Primitive: PolarQuantEncode
///////////////////////////////////////////////////////////////////////////////

std::vector<mx::array> polarquant_encode(
    const mx::array& input,
    const PolarQuantParams& params,
    mx::StreamOrDevice s) {
  auto stream = to_stream(s);

  if (input.ndim() != 4) {
    throw std::invalid_argument("polarquant_encode expects 4D input [B, H, S, D]");
  }
  if (input.shape(-1) != params.head_dim) {
    throw std::invalid_argument("Last dim must equal head_dim");
  }

  int B = input.shape(0);
  int H = input.shape(1);
  int S = input.shape(2);
  int D = params.head_dim;
  int n_angles = total_angles(D);

  auto casted = mx::astype(input, mx::float32, s);

  // Use make_arrays for multi-output primitive
  return mx::array::make_arrays(
      {{B, H, S, n_angles}, {B, H, S, 1}},
      {mx::uint8, mx::float32},
      std::make_shared<PolarQuantEncode>(stream, params),
      {casted});
}

mx::array polarquant_decode(
    const mx::array& angles,
    const mx::array& norms,
    const PolarQuantParams& params,
    mx::StreamOrDevice s) {
  auto stream = to_stream(s);

  int B = angles.shape(0);
  int H = angles.shape(1);
  int S = angles.shape(2);
  int D = params.head_dim;

  return mx::array(
      {B, H, S, D},
      mx::float32,
      std::make_shared<PolarQuantDecode>(stream, params),
      {angles, norms});
}

///////////////////////////////////////////////////////////////////////////////
// CPU eval
///////////////////////////////////////////////////////////////////////////////

void PolarQuantEncode::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& input = inputs[0];
  auto& angles = outputs[0];
  auto& norms = outputs[1];

  int B = input.shape(0);
  int H = input.shape(1);
  int S = input.shape(2);
  int D = params_.head_dim;
  int n_angles = total_angles(D);

  angles.set_data(mx::allocator::malloc(angles.nbytes()));
  norms.set_data(mx::allocator::malloc(norms.nbytes()));

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(input);
  encoder.set_output_array(angles);
  encoder.set_output_array(norms);

  encoder.dispatch([in_ptr = input.data<float>(),
                    ang_ptr = angles.data<uint8_t>(),
                    nrm_ptr = norms.data<float>(),
                    B, H, S, D,
                    bits = params_.bits]() {
    polarquant_encode_cpu(in_ptr, ang_ptr, nrm_ptr, B, H, S, D, bits);
  });
}

void PolarQuantDecode::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& angles = inputs[0];
  auto& norms = inputs[1];
  auto& output = outputs[0];

  int B = angles.shape(0);
  int H = angles.shape(1);
  int S = angles.shape(2);
  int D = params_.head_dim;

  output.set_data(mx::allocator::malloc(output.nbytes()));

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(angles);
  encoder.set_input_array(norms);
  encoder.set_output_array(output);

  encoder.dispatch([ang_ptr = angles.data<uint8_t>(),
                    nrm_ptr = norms.data<float>(),
                    out_ptr = output.data<float>(),
                    B, H, S, D,
                    bits = params_.bits]() {
    polarquant_decode_cpu(ang_ptr, nrm_ptr, out_ptr, B, H, S, D, bits);
  });
}

///////////////////////////////////////////////////////////////////////////////
// GPU eval (Metal)
///////////////////////////////////////////////////////////////////////////////

#ifdef _METAL_

void PolarQuantEncode::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& input = inputs[0];
  auto& angles = outputs[0];
  auto& norms = outputs[1];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  angles.set_data(mx::allocator::malloc(angles.nbytes()));
  norms.set_data(mx::allocator::malloc(norms.nbytes()));

  int B = input.shape(0);
  int H = input.shape(1);
  int S = input.shape(2);
  int D = params_.head_dim;
  int n_angles = total_angles(D);
  int total_vecs = B * H * S;

  auto lib = d.get_library("turboquant", current_binary_dir());
  auto kernel = d.get_kernel("polarquant_encode_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(input, 0);
  compute_encoder.set_output_array(angles, 1);
  compute_encoder.set_output_array(norms, 2);
  compute_encoder.set_bytes(D, 3);
  compute_encoder.set_bytes(params_.bits, 4);
  compute_encoder.set_bytes(total_vecs, 5);

  // One thread per vector
  size_t tgp_size = std::min(
      static_cast<size_t>(total_vecs),
      kernel->maxTotalThreadsPerThreadgroup());
  MTL::Size grid_dims = MTL::Size(total_vecs, 1, 1);
  MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void PolarQuantDecode::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& angles = inputs[0];
  auto& norms = inputs[1];
  auto& output = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  output.set_data(mx::allocator::malloc(output.nbytes()));

  int B = angles.shape(0);
  int H = angles.shape(1);
  int S = angles.shape(2);
  int D = params_.head_dim;
  int total_vecs = B * H * S;

  auto lib = d.get_library("turboquant", current_binary_dir());
  auto kernel = d.get_kernel("polarquant_decode_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(angles, 0);
  compute_encoder.set_input_array(norms, 1);
  compute_encoder.set_output_array(output, 2);
  compute_encoder.set_bytes(D, 3);
  compute_encoder.set_bytes(params_.bits, 4);
  compute_encoder.set_bytes(total_vecs, 5);

  size_t tgp_size = std::min(
      static_cast<size_t>(total_vecs),
      kernel->maxTotalThreadsPerThreadgroup());
  MTL::Size grid_dims = MTL::Size(total_vecs, 1, 1);
  MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

#else

void PolarQuantEncode::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("PolarQuantEncode has no GPU implementation.");
}

void PolarQuantDecode::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("PolarQuantDecode has no GPU implementation.");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Equivalence
///////////////////////////////////////////////////////////////////////////////

bool PolarQuantEncode::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const PolarQuantEncode&>(other);
  return params_.head_dim == o.params_.head_dim &&
         params_.bits == o.params_.bits;
}

bool PolarQuantDecode::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const PolarQuantDecode&>(other);
  return params_.head_dim == o.params_.head_dim &&
         params_.bits == o.params_.bits;
}

}  // namespace turboquant
