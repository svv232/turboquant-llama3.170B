#include <metal_stdlib>
using namespace metal;

// Fused Scaled Dot-Product Attention with PolarQuant KV Cache
//
// Instead of dequantizing PolarQuant → FP16 → standard SDPA, we fuse
// the PolarQuant decode directly into the attention loop.
//
// PolarQuant encoding (D=128, 5-bit):
//   127 quantized angles (uint8, one per byte) + 1 float32 norm per vector
//   Decode: start from norm, expand 1→2→4→...→128 using cos/sin of dequantized angles
//
// Kernel structure mirrors sdpa_int4_kernel:
//   - Each threadgroup = one (batch, q_head, q_pos)
//   - 32 threads per simdgroup, each handles 4 elements of head_dim=128
//   - Online softmax over blocks of BN=32 KV positions
//   - Pre-scaled query for fused QK dot product

constant constexpr int BN = 32;        // inner block for online softmax
constant constexpr int D = 128;        // head_dim
constant constexpr int SIMD_SIZE = 32; // Apple GPU SIMD width
constant constexpr int ELEMS_PER_THREAD = D / SIMD_SIZE;  // 4
constant constexpr int N_ANGLES = D - 1;  // 127 angles per vector
constant constexpr int N_LEVELS = 7;      // log2(128) = 7

struct SdpaPolarParams {
  int num_q_heads;
  int num_kv_heads;
  int seq_len_q;
  int seq_len_kv;
  int bits;
  float attn_scale;
};

// Precomputed cos/sin lookup table for 5-bit quantization (32 bins)
// Angle for bin b = (b + 0.5) * (2*pi/n_bins) - pi
// These are generated at compile time for 5-bit (32 bins).
// For other bit widths, we compute at runtime from the bin index.

// Decode a single PolarQuant vector into thread-local elements.
// Each thread in the simdgroup decodes its assigned 4 elements.
//
// The decode is hierarchical: we start from the single norm value and
// expand level by level. However, each thread only needs the final
// 4 elements assigned to it. We decode the full tree path from root
// to those 4 leaf elements.
//
// Level structure for D=128:
//   Level 6: 1 pair  → angles[126]           offset=126, pairs=1
//   Level 5: 2 pairs → angles[124..125]      offset=124, pairs=2
//   Level 4: 4 pairs → angles[120..123]      offset=120, pairs=4
//   Level 3: 8 pairs → angles[112..119]      offset=112, pairs=8
//   Level 2: 16 pairs → angles[96..111]      offset=96,  pairs=16
//   Level 1: 32 pairs → angles[64..95]       offset=64,  pairs=32
//   Level 0: 64 pairs → angles[0..63]        offset=0,   pairs=64

// Precomputed level offsets for D=128 (compile-time constants)
// Level L has D>>(L+1) pairs. Offset = sum of pairs at levels 0..L-1.
constant constexpr int LEVEL_OFFSET[N_LEVELS] = {0, 64, 96, 112, 120, 124, 126};

// Decode PolarQuant vector into 4 elements for this thread.
// We trace from root (level 6) down to the 4 leaf elements at level 0.
//
// Thread lane_id owns elements [lane_id*4 .. lane_id*4+3].
// At level 0, those correspond to pairs (lane_id*2) and (lane_id*2+1).
// At level 1, they all descend from pair (lane_id).
// At level 2, from pair (lane_id/2). And so on.
//
// Strategy: decode top-down, tracking only the radius that leads
// to this thread's elements. At each level, we split using the angle
// for the relevant pair.
inline void decode_polar_vector(
    device const uchar* angles,  // [N_ANGLES] for this vector
    float norm,
    uint lane_id,
    int n_bins,
    float bin_width,
    thread float* out  // [4] output elements
) {
  // Top-down decode from root to this thread's 4 elements.
  //
  // This thread owns final vector elements [lane_id*4 .. lane_id*4+3].
  // At level L, the ancestor VALUE index for this thread is: lane_id >> (L-1)
  // The ancestor PAIR index at level L is: lane_id >> L
  // The child side (cos=0, sin=1) is: (lane_id >> (L-1)) & 1
  //
  // We descend from level 6 (root) to level 2, tracking a single radius.
  // Then expand level 1 into 2 values, and level 0 into 4 values.
  float radius = norm;

  // Decode from level 6 (root) down to level 2 (inclusive)
  // After this, we have the radius at the level-2 node that is ancestor
  // of all 4 of this thread's elements.
  // Level-2 ancestor index = leaf_pair_start >> 2 = lane_id / 2  (but using pairs)
  // Actually: at level 2, there are 16 pairs. This thread's 4 elements
  // come from level-2 pair index = lane_id / 2.

  // Wait, let me think about this more carefully.
  // At level L, there are LEVEL_PAIRS[L] = D >> (L+1) pairs.
  // Level 0: 64 pairs → 128 values (the final vector)
  // Level 1: 32 pairs → 64 values (radii from level 0)
  // Level 2: 16 pairs → 32 values (radii from level 1)
  // ...
  // Level 6: 1 pair → 2 values (radii from level 5), and the root norm feeds into this.
  //
  // This thread owns elements [lane_id*4 .. lane_id*4+3].
  // At level 0, elements (2i, 2i+1) come from pair i with radius = level-1 value i.
  // So this thread's elements come from level-0 pairs (lane_id*2) and (lane_id*2+1).
  // Those two pairs' radii are level-1 values (lane_id*2) and (lane_id*2+1).
  // Those come from level-1 pair (lane_id), with radius = level-2 value (lane_id).
  // That radius is level-2 value (lane_id), from level-2 pair (lane_id/2).
  // And so on up.
  //
  // So at level L (L >= 2), the ancestor node index is:
  //   ancestor = lane_id >> (L - 1)   ... hmm, let me just use the pair-based indexing.
  //
  // At level L, we need to know which VALUE (not pair) is the ancestor.
  // Level 1 value: lane_id  (since level-1 has 32 values = SIMD_SIZE)
  // Level 2 value: lane_id / 2
  // Level L value (L >= 1): lane_id >> (L-1)
  // Level L pair: floor(value_at_L / 2) = lane_id >> L
  //
  // When descending from level L to level L-1:
  //   parent_value = ancestor at level L = lane_id >> (L-1)
  //   parent_pair = parent_value / 2 = lane_id >> L
  //   child_side = parent_value & 1  (0=cos child, 1=sin child)
  //   angle = angles[LEVEL_OFFSET[L] + parent_pair]
  //
  // After applying cos/sin, we get the value at level L-1 for ancestor lane_id >> (L-2).
  //
  // We want to stop at level 1, which gives us 1 value per thread.
  // Then from level 1 value, we expand to 2 level-0 values (via level-1 pair = lane_id),
  // and from each level-0 value, we expand to 2 final elements.
  //
  // Actually: from level 1 we have radius_at_level1[lane_id].
  // Level-1 pair (lane_id) has angle angles[LEVEL_OFFSET[1] + lane_id].
  //   cos child → level-0 value (lane_id*2)   = radius for level-0 pair (lane_id*2)/2 wait no.
  //
  // Let me be very precise:
  // At level L, pair p has:
  //   input radius = "value p" from the radii at that level
  //   angle = angles[LEVEL_OFFSET[L] + p]
  //   output: value (2p) at level L-1 = radius * cos(angle)
  //           value (2p+1) at level L-1 = radius * sin(angle)
  //
  // So from level 1, pair (lane_id):
  //   input = radius_level1[lane_id]  (what we computed by descending from root)
  //   angle = angles[64 + lane_id]  (LEVEL_OFFSET[1]=64)
  //   r0 = radius * cos(theta)  → level-0 value (lane_id*2)
  //   r1 = radius * sin(theta)  → level-0 value (lane_id*2+1)
  //
  // Then from level 0, pairs (lane_id*2) and (lane_id*2+1):
  //   Pair (lane_id*2): input = r0, angle = angles[lane_id*2]
  //     out[0] = r0 * cos(theta0), out[1] = r0 * sin(theta0)
  //   Pair (lane_id*2+1): input = r1, angle = angles[lane_id*2+1]
  //     out[2] = r1 * cos(theta1), out[3] = r1 * sin(theta1)

  // OK now implement cleanly:
  radius = norm;

  // Descend from level 6 to level 2 (4 steps)
  for (int L = N_LEVELS - 1; L >= 2; L--) {
    int ancestor_value = lane_id >> (L - 1);
    int pair_idx = ancestor_value >> 1;  // = lane_id >> L
    int child_side = ancestor_value & 1;

    uchar bin = angles[LEVEL_OFFSET[L] + pair_idx];
    float theta = (float(bin) + 0.5f) * bin_width - M_PI_F;

    radius = child_side ? (radius * sin(theta)) : (radius * cos(theta));
  }

  // Now radius = level-1 value at index (lane_id).
  // Expand via level-1 pair (lane_id):
  uchar bin1 = angles[LEVEL_OFFSET[1] + lane_id];
  float theta1 = (float(bin1) + 0.5f) * bin_width - M_PI_F;
  float r0 = radius * cos(theta1);  // level-0 value at (lane_id*2)
  float r1 = radius * sin(theta1);  // level-0 value at (lane_id*2+1)

  // Expand via level-0 pairs:
  uchar bin0a = angles[LEVEL_OFFSET[0] + lane_id * 2];
  float theta0a = (float(bin0a) + 0.5f) * bin_width - M_PI_F;
  out[0] = r0 * cos(theta0a);
  out[1] = r0 * sin(theta0a);

  uchar bin0b = angles[LEVEL_OFFSET[0] + lane_id * 2 + 1];
  float theta0b = (float(bin0b) + 0.5f) * bin_width - M_PI_F;
  out[2] = r1 * cos(theta0b);
  out[3] = r1 * sin(theta0b);
}


// ============================================================
// Single-pass fused PolarQuant SDPA kernel
// ============================================================

[[kernel]] void sdpa_polar_kernel(
    device const half* Q              [[buffer(0)]],   // [B, Hq, Sq, D]
    device const uchar* K_angles      [[buffer(1)]],   // [B, Hkv, Skv, N_ANGLES]
    device const float* K_norms       [[buffer(2)]],   // [B, Hkv, Skv]
    device const uchar* V_angles      [[buffer(3)]],   // [B, Hkv, Skv, N_ANGLES]
    device const float* V_norms       [[buffer(4)]],   // [B, Hkv, Skv]
    device half* output               [[buffer(5)]],   // [B, Hq, Sq, D]
    constant SdpaPolarParams& params  [[buffer(6)]],
    uint2 group_pos  [[threadgroup_position_in_grid]],
    uint lane_id     [[thread_index_in_simdgroup]]) {

  const int Hq = params.num_q_heads;
  const int Hkv = params.num_kv_heads;
  const int Sq = params.seq_len_q;
  const int Skv = params.seq_len_kv;
  const int bits = params.bits;
  const float scale = params.attn_scale;
  const int gqa_ratio = Hq / Hkv;
  const int n_bins = 1 << bits;
  const float bin_width = (2.0f * M_PI_F) / float(n_bins);

  int batch_idx = group_pos.y;
  int flat_qh = group_pos.x;
  int q_head = flat_qh / Sq;
  int q_pos = flat_qh % Sq;
  int kv_head = q_head / gqa_ratio;

  // Load and pre-scale query
  int q_offset = ((batch_idx * Hq + q_head) * Sq + q_pos) * D;
  float q_local[ELEMS_PER_THREAD];
  for (int e = 0; e < ELEMS_PER_THREAD; e++) {
    q_local[e] = float(Q[q_offset + lane_id * ELEMS_PER_THREAD + e]) * scale;
  }

  float max_score = -INFINITY;
  float sum_exp = 0.0f;
  float out_accum[ELEMS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

  int kv_base = (batch_idx * Hkv + kv_head) * Skv;
  int num_blocks = (Skv + BN - 1) / BN;

  for (int blk = 0; blk < num_blocks; blk++) {
    int block_start = blk * BN;
    int block_end = min(block_start + BN, Skv);
    int block_len = block_end - block_start;

    // Phase 1: Decode K vectors and compute QK dot products
    float scores[BN];
    for (int bn = 0; bn < block_len; bn++) {
      int kv_idx = kv_base + block_start + bn;
      device const uchar* k_ang = K_angles + kv_idx * N_ANGLES;
      float k_norm = K_norms[kv_idx];

      float k_decoded[ELEMS_PER_THREAD];
      decode_polar_vector(k_ang, k_norm, lane_id, n_bins, bin_width, k_decoded);

      float dot = 0.0f;
      for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        dot += q_local[e] * k_decoded[e];
      }
      scores[bn] = simd_sum(dot);
    }

    // Phase 2: Online softmax
    float block_max = -INFINITY;
    for (int bn = 0; bn < block_len; bn++) block_max = max(block_max, scores[bn]);
    float new_max = max(max_score, block_max);
    float correction = exp(max_score - new_max);
    sum_exp *= correction;
    for (int e = 0; e < ELEMS_PER_THREAD; e++) out_accum[e] *= correction;

    float block_sum_exp = 0.0f;
    float exp_scores[BN];
    for (int bn = 0; bn < block_len; bn++) {
      float es = exp(scores[bn] - new_max);
      exp_scores[bn] = es;
      block_sum_exp += es;
    }
    sum_exp += block_sum_exp;
    max_score = new_max;

    // Phase 3: Decode V vectors and accumulate weighted V
    for (int bn = 0; bn < block_len; bn++) {
      int kv_idx = kv_base + block_start + bn;
      device const uchar* v_ang = V_angles + kv_idx * N_ANGLES;
      float v_norm = V_norms[kv_idx];

      float v_decoded[ELEMS_PER_THREAD];
      decode_polar_vector(v_ang, v_norm, lane_id, n_bins, bin_width, v_decoded);

      float w = exp_scores[bn];
      for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        out_accum[e] += w * v_decoded[e];
      }
    }
  }

  float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
  int out_offset = ((batch_idx * Hq + q_head) * Sq + q_pos) * D;
  for (int e = 0; e < ELEMS_PER_THREAD; e++) {
    output[out_offset + lane_id * ELEMS_PER_THREAD + e] = half(out_accum[e] * inv_sum);
  }
}
