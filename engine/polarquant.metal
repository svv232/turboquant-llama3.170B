#include <metal_stdlib>
using namespace metal;

// PolarQuant encode kernel
// One thread per vector: takes a float32 vector of dim D,
// recursively decomposes into polar coordinates, quantizes angles to n_bins levels.
//
// For head_dim=128:
//   Level 0: 64 pairs → 64 angles, 64 radii
//   Level 1: 32 pairs → 32 angles, 32 radii
//   ...
//   Level 6: 1 pair → 1 angle, 1 radius (norm)
//   Total: 127 angles + 1 norm

[[kernel]] void polarquant_encode_kernel(
    device const float* input    [[buffer(0)]],  // [total_vecs, D]
    device uchar* angles_out     [[buffer(1)]],  // [total_vecs, D-1]
    device float* norms_out      [[buffer(2)]],  // [total_vecs]
    constant int& D              [[buffer(3)]],
    constant int& bits           [[buffer(4)]],
    constant int& total_vecs     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {

  if (tid >= uint(total_vecs)) return;

  const int n_angles = D - 1;
  const int n_bins = 1 << bits;
  const float bin_width = (2.0f * M_PI_F) / float(n_bins);

  // Each thread processes one vector
  device const float* vec = input + tid * D;
  device uchar* ang = angles_out + tid * n_angles;

  // Use threadgroup memory would be better for large D,
  // but for D=128 we can use thread-local arrays
  // Metal doesn't support VLAs, so use fixed size for D=128
  float radii_a[128];
  float radii_b[64];

  // Copy input
  for (int i = 0; i < D; i++) {
    radii_a[i] = vec[i];
  }

  int angle_offset = 0;
  int cur_dim = D;

  // Recursive polar decomposition
  // We alternate between radii_a and radii_b as source/dest
  bool use_a = true;

  while (cur_dim > 1) {
    int n_pairs = cur_dim / 2;

    for (int p = 0; p < n_pairs; p++) {
      float x0, x1;
      if (use_a) {
        x0 = radii_a[2 * p];
        x1 = radii_a[2 * p + 1];
      } else {
        x0 = radii_b[2 * p];
        x1 = radii_b[2 * p + 1];
      }

      float r = sqrt(x0 * x0 + x1 * x1);
      float theta = atan2(x1, x0);

      // Quantize: map [-π, π) → [0, n_bins)
      float normalized = (theta + M_PI_F) / bin_width;
      int bin = int(floor(normalized));
      bin = clamp(bin, 0, n_bins - 1);

      ang[angle_offset + p] = uchar(bin);

      if (use_a) {
        radii_b[p] = r;
      } else {
        radii_a[p] = r;
      }
    }

    angle_offset += n_pairs;
    cur_dim = n_pairs;
    use_a = !use_a;
  }

  // Final radius is the norm
  norms_out[tid] = use_a ? radii_a[0] : radii_b[0];
}

// PolarQuant decode kernel
// Reverse: from quantized angles + norm, reconstruct the vector
[[kernel]] void polarquant_decode_kernel(
    device const uchar* angles_in  [[buffer(0)]],  // [total_vecs, D-1]
    device const float* norms_in   [[buffer(1)]],  // [total_vecs]
    device float* output           [[buffer(2)]],  // [total_vecs, D]
    constant int& D                [[buffer(3)]],
    constant int& bits             [[buffer(4)]],
    constant int& total_vecs       [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {

  if (tid >= uint(total_vecs)) return;

  const int n_angles = D - 1;
  const int n_bins = 1 << bits;
  const float bin_width = (2.0f * M_PI_F) / float(n_bins);

  device const uchar* ang = angles_in + tid * n_angles;
  device float* out = output + tid * D;

  // Compute recursion level info
  // For D=128: levels = 7
  // level_pairs[l] = D >> (l+1), level_offset[l] = sum of pairs before level l
  int level_offset[8];  // max 8 levels for D=256
  int level_pairs[8];
  int n_levels = 0;
  {
    int off = 0;
    int d = D;
    while (d > 1) {
      level_pairs[n_levels] = d / 2;
      level_offset[n_levels] = off;
      off += d / 2;
      d /= 2;
      n_levels++;
    }
  }

  // Start from the root norm
  float radii[128];
  radii[0] = norms_in[tid];

  // Reconstruct from deepest level to level 0
  for (int l = n_levels - 1; l >= 0; l--) {
    int n_pairs_l = level_pairs[l];
    int off = level_offset[l];

    // Expand in reverse order to avoid overwriting
    for (int p = n_pairs_l - 1; p >= 0; p--) {
      float r = radii[p];
      uchar bin = ang[off + p];

      // Dequantize: use bin center
      float theta = (float(bin) + 0.5f) * bin_width - M_PI_F;

      radii[2 * p] = r * cos(theta);
      radii[2 * p + 1] = r * sin(theta);
    }
  }

  // Write output
  for (int i = 0; i < D; i++) {
    out[i] = radii[i];
  }
}
