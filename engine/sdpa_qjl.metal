#include <metal_stdlib>
using namespace metal;

// Fused QJL SDPA: 1-bit sketch scoring + int4 V dequant + online softmax
//
// Instead of full Q*K dot products, uses QJL sketch XNOR+popcount to estimate
// attention scores, then accumulates int4-quantized V with online softmax.
// Pure bitwise scoring is dramatically faster than float multiply-add.

constant constexpr int BN = 32;        // KV tokens per block for online softmax
constant constexpr int D = 128;        // head_dim
constant constexpr int D_HALF = 64;    // packed int4 width in bytes
constant constexpr int SIMD_SIZE = 32; // Apple GPU SIMD width
constant constexpr int ELEMS_PER_THREAD = D / SIMD_SIZE;  // 4

struct SdpaQJLParams {
  int num_q_heads;
  int num_kv_heads;
  int seq_len_q;
  int seq_len_kv;
  int sketch_dim;      // m (number of projection bits)
  int sketch_words;    // m / 32 (number of uint32 words per sketch)
  int v_group_size;    // int4 V quantization group size
  float attn_scale;    // 1/sqrt(head_dim)
};

// ============================================================
// Single-pass kernel
// ============================================================
// Grid:  [Hq * Sq, B, 1]
// Group: [SIMD_SIZE, 1, 1]
//
// Each SIMD group handles one (batch, q_head, q_pos) tuple.
// Phase 1: QJL score via XNOR + popcount on packed sketches
// Phase 2: Online softmax
// Phase 3: Int4 V dequant and weighted accumulation

[[kernel]] void sdpa_qjl_kernel(
    device const uint* Q_sketches      [[buffer(0)]],   // [B, Hq, Sq, sketch_words]
    device const half* Q_norms         [[buffer(1)]],   // [B, Hq, Sq]
    device const uint* K_sketches      [[buffer(2)]],   // [B, Hkv, Skv, sketch_words]
    device const half* K_norms         [[buffer(3)]],   // [B, Hkv, Skv]
    device const uchar* V_quant        [[buffer(4)]],   // [B, Hkv, Skv, D/2]
    device const half* V_scales        [[buffer(5)]],   // [B, Hkv, Skv, D/v_group]
    device const half* V_zeros         [[buffer(6)]],   // [B, Hkv, Skv, D/v_group]
    device half* output                [[buffer(7)]],   // [B, Hq, Sq, D]
    constant SdpaQJLParams& params     [[buffer(8)]],
    uint2 group_pos  [[threadgroup_position_in_grid]],
    uint lane_id     [[thread_index_in_simdgroup]]) {

  const int Hq = params.num_q_heads;
  const int Hkv = params.num_kv_heads;
  const int Sq = params.seq_len_q;
  const int Skv = params.seq_len_kv;
  const int sw = params.sketch_words;    // m / 32
  const int m = params.sketch_dim;
  const int gs = params.v_group_size;
  const float scale = params.attn_scale;
  const int gqa_ratio = Hq / Hkv;

  int batch_idx = group_pos.y;
  int flat_qh = group_pos.x;
  int q_head = flat_qh / Sq;
  int q_pos = flat_qh % Sq;
  int kv_head = q_head / gqa_ratio;

  // Load Q sketch words: each thread loads a subset of sketch words
  // For sw sketch words across SIMD_SIZE=32 threads:
  //   each thread handles ceil(sw / SIMD_SIZE) words
  int q_sketch_base = ((batch_idx * Hq + q_head) * Sq + q_pos) * sw;

  // Store Q sketch in registers. Max sketch_dim=768 -> 24 words.
  // Each thread loads its assigned words.
  // We'll iterate over all words; each thread loads words[lane_id + t*SIMD_SIZE]
  int words_per_thread = (sw + SIMD_SIZE - 1) / SIMD_SIZE;
  uint q_sk[24]; // max 768/32 = 24 words
  for (int t = 0; t < words_per_thread; t++) {
    int w = lane_id + t * SIMD_SIZE;
    if (w < sw) {
      q_sk[t] = Q_sketches[q_sketch_base + w];
    } else {
      q_sk[t] = 0;
    }
  }

  // Load Q norm (scalar, same for all lanes)
  float q_norm = float(Q_norms[((batch_idx * Hq + q_head) * Sq + q_pos)]);

  float max_score = -INFINITY;
  float sum_exp = 0.0f;
  float out_accum[ELEMS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

  int kv_base = (batch_idx * Hkv + kv_head) * Skv;
  int groups_per_vec = D / gs;
  int start_elem = lane_id * ELEMS_PER_THREAD;
  int num_blocks = (Skv + BN - 1) / BN;

  float inv_m = 1.0f / float(m);

  for (int blk = 0; blk < num_blocks; blk++) {
    int block_start = blk * BN;
    int block_end = min(block_start + BN, Skv);
    int block_len = block_end - block_start;

    // Phase 1: QJL scores via XNOR + popcount
    float scores[BN];
    for (int bn = 0; bn < block_len; bn++) {
      int kv_idx = kv_base + block_start + bn;
      int k_sketch_base = kv_idx * sw;

      // Each thread computes popcount on its assigned sketch words
      int local_matching = 0;
      for (int t = 0; t < words_per_thread; t++) {
        int w = lane_id + t * SIMD_SIZE;
        if (w < sw) {
          uint k_word = K_sketches[k_sketch_base + w];
          uint xnor_val = ~(q_sk[t] ^ k_word);
          local_matching += popcount(xnor_val);
        }
      }

      // Reduce matching count across SIMD group
      int total_matching = simd_sum(local_matching);

      // Score = (||q|| * ||k|| / m) * (2 * total_matching - m) * attn_scale
      float k_norm = float(K_norms[kv_idx]);
      float hamming_agreement = float(2 * total_matching - m);
      scores[bn] = (q_norm * k_norm * inv_m) * hamming_agreement * scale;
    }

    // Phase 2: Online softmax (same pattern as sdpa_int4)
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

    // Phase 3: Int4 V dequant and weighted accumulation
    for (int bn = 0; bn < block_len; bn++) {
      int kv_vec_offset = kv_base + block_start + bn;
      int group_idx = start_elem / gs;
      float v_scale = float(V_scales[kv_vec_offset * groups_per_vec + group_idx]);
      float v_zero = float(V_zeros[kv_vec_offset * groups_per_vec + group_idx]);
      device const uchar* v_packed = V_quant + kv_vec_offset * D_HALF;
      uint16_t raw16 = *reinterpret_cast<device const uint16_t*>(v_packed + start_elem / 2);

      float w = exp_scores[bn];
      out_accum[0] += w * ((float(raw16 & 0xF) - v_zero) * v_scale);
      out_accum[1] += w * ((float((raw16 >> 4) & 0xF) - v_zero) * v_scale);
      out_accum[2] += w * ((float((raw16 >> 8) & 0xF) - v_zero) * v_scale);
      out_accum[3] += w * ((float((raw16 >> 12) & 0xF) - v_zero) * v_scale);
    }
  }

  float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
  int out_offset = ((batch_idx * Hq + q_head) * Sq + q_pos) * D;
  for (int e = 0; e < ELEMS_PER_THREAD; e++) {
    output[out_offset + lane_id * ELEMS_PER_THREAD + e] = half(out_accum[e] * inv_sum);
  }
}

// ============================================================
// Split-K Phase 1: Partial attention over a KV chunk
// ============================================================
// Grid:  [Hq * Sq * num_chunks, B, 1]
// Group: [SIMD_SIZE, 1, 1]

struct SplitKParams {
  int num_chunks;
  int chunk_size;
};

[[kernel]] void sdpa_qjl_partial(
    device const uint* Q_sketches      [[buffer(0)]],
    device const half* Q_norms         [[buffer(1)]],
    device const uint* K_sketches      [[buffer(2)]],
    device const half* K_norms         [[buffer(3)]],
    device const uchar* V_quant        [[buffer(4)]],
    device const half* V_scales        [[buffer(5)]],
    device const half* V_zeros         [[buffer(6)]],
    device float* partial_out          [[buffer(7)]],   // [B, Hq, Sq, num_chunks, D]
    device float* partial_max          [[buffer(8)]],   // [B, Hq, Sq, num_chunks]
    device float* partial_sum_exp      [[buffer(9)]],   // [B, Hq, Sq, num_chunks]
    constant SdpaQJLParams& params     [[buffer(10)]],
    constant SplitKParams& sk_params   [[buffer(11)]],
    uint2 group_pos  [[threadgroup_position_in_grid]],
    uint lane_id     [[thread_index_in_simdgroup]]) {

  const int Hq = params.num_q_heads;
  const int Hkv = params.num_kv_heads;
  const int Sq = params.seq_len_q;
  const int Skv = params.seq_len_kv;
  const int sw = params.sketch_words;
  const int m = params.sketch_dim;
  const int gs = params.v_group_size;
  const float scale = params.attn_scale;
  const int gqa_ratio = Hq / Hkv;
  const int num_chunks = sk_params.num_chunks;
  const int chunk_size = sk_params.chunk_size;

  int batch_idx = group_pos.y;
  int flat_idx = group_pos.x;
  int chunk_idx = flat_idx % num_chunks;
  int flat_qh = flat_idx / num_chunks;
  int q_head = flat_qh / Sq;
  int q_pos = flat_qh % Sq;
  int kv_head = q_head / gqa_ratio;

  int chunk_start = chunk_idx * chunk_size;
  int chunk_end = min(chunk_start + chunk_size, Skv);
  if (chunk_start >= Skv) return;

  // Load Q sketch
  int q_sketch_base = ((batch_idx * Hq + q_head) * Sq + q_pos) * sw;
  int words_per_thread = (sw + SIMD_SIZE - 1) / SIMD_SIZE;
  uint q_sk[24];
  for (int t = 0; t < words_per_thread; t++) {
    int w = lane_id + t * SIMD_SIZE;
    if (w < sw) {
      q_sk[t] = Q_sketches[q_sketch_base + w];
    } else {
      q_sk[t] = 0;
    }
  }

  float q_norm = float(Q_norms[((batch_idx * Hq + q_head) * Sq + q_pos)]);

  float max_score = -INFINITY;
  float sum_exp = 0.0f;
  float out_accum[ELEMS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

  int kv_base = (batch_idx * Hkv + kv_head) * Skv;
  int groups_per_vec = D / gs;
  int start_elem = lane_id * ELEMS_PER_THREAD;
  float inv_m = 1.0f / float(m);

  int num_blocks = (chunk_end - chunk_start + BN - 1) / BN;

  for (int blk = 0; blk < num_blocks; blk++) {
    int block_start = chunk_start + blk * BN;
    int block_end = min(block_start + BN, chunk_end);
    int block_len = block_end - block_start;

    float scores[BN];
    for (int bn = 0; bn < block_len; bn++) {
      int kv_idx = kv_base + block_start + bn;
      int k_sketch_base = kv_idx * sw;

      int local_matching = 0;
      for (int t = 0; t < words_per_thread; t++) {
        int w = lane_id + t * SIMD_SIZE;
        if (w < sw) {
          uint k_word = K_sketches[k_sketch_base + w];
          uint xnor_val = ~(q_sk[t] ^ k_word);
          local_matching += popcount(xnor_val);
        }
      }
      int total_matching = simd_sum(local_matching);

      float k_norm = float(K_norms[kv_idx]);
      float hamming_agreement = float(2 * total_matching - m);
      scores[bn] = (q_norm * k_norm * inv_m) * hamming_agreement * scale;
    }

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

    for (int bn = 0; bn < block_len; bn++) {
      int kv_vec_offset = kv_base + block_start + bn;
      int group_idx = start_elem / gs;
      float v_scale = float(V_scales[kv_vec_offset * groups_per_vec + group_idx]);
      float v_zero = float(V_zeros[kv_vec_offset * groups_per_vec + group_idx]);
      device const uchar* v_packed = V_quant + kv_vec_offset * D_HALF;
      uint16_t raw16 = *reinterpret_cast<device const uint16_t*>(v_packed + start_elem / 2);

      float w = exp_scores[bn];
      out_accum[0] += w * ((float(raw16 & 0xF) - v_zero) * v_scale);
      out_accum[1] += w * ((float((raw16 >> 4) & 0xF) - v_zero) * v_scale);
      out_accum[2] += w * ((float((raw16 >> 8) & 0xF) - v_zero) * v_scale);
      out_accum[3] += w * ((float((raw16 >> 12) & 0xF) - v_zero) * v_scale);
    }
  }

  // Write partial results
  int po_base = (((batch_idx * Hq + q_head) * Sq + q_pos) * num_chunks + chunk_idx) * D;
  for (int e = 0; e < ELEMS_PER_THREAD; e++) {
    partial_out[po_base + lane_id * ELEMS_PER_THREAD + e] = out_accum[e];
  }

  if (lane_id == 0) {
    int ps_base = ((batch_idx * Hq + q_head) * Sq + q_pos) * num_chunks + chunk_idx;
    partial_max[ps_base] = max_score;
    partial_sum_exp[ps_base] = sum_exp;
  }
}

// ============================================================
// Split-K Phase 2: Reduce partial results
// ============================================================
// Grid:  [Hq * Sq, B, 1]
// Group: [SIMD_SIZE, 1, 1]

[[kernel]] void sdpa_qjl_reduce(
    device const float* partial_out       [[buffer(0)]],
    device const float* partial_max       [[buffer(1)]],
    device const float* partial_sum_exp   [[buffer(2)]],
    device half* output                   [[buffer(3)]],
    constant SdpaQJLParams& params        [[buffer(4)]],
    constant SplitKParams& sk_params      [[buffer(5)]],
    uint2 group_pos  [[threadgroup_position_in_grid]],
    uint lane_id     [[thread_index_in_simdgroup]]) {

  const int Hq = params.num_q_heads;
  const int Sq = params.seq_len_q;
  const int num_chunks = sk_params.num_chunks;

  int batch_idx = group_pos.y;
  int flat_qh = group_pos.x;
  int q_head = flat_qh / Sq;
  int q_pos = flat_qh % Sq;

  int ps_base = ((batch_idx * Hq + q_head) * Sq + q_pos) * num_chunks;

  float global_max = -INFINITY;
  for (int c = 0; c < num_chunks; c++) {
    global_max = max(global_max, partial_max[ps_base + c]);
  }

  float total_sum_exp = 0.0f;
  float combined_out[ELEMS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

  int po_base = (((batch_idx * Hq + q_head) * Sq + q_pos) * num_chunks) * D;

  for (int c = 0; c < num_chunks; c++) {
    float correction = exp(partial_max[ps_base + c] - global_max);
    float corrected_sum = partial_sum_exp[ps_base + c] * correction;
    total_sum_exp += corrected_sum;

    int chunk_out_base = po_base + c * D;
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
      combined_out[e] += partial_out[chunk_out_base + lane_id * ELEMS_PER_THREAD + e] * correction;
    }
  }

  float inv_sum = (total_sum_exp > 0.0f) ? (1.0f / total_sum_exp) : 0.0f;
  int out_offset = ((batch_idx * Hq + q_head) * Sq + q_pos) * D;
  for (int e = 0; e < ELEMS_PER_THREAD; e++) {
    output[out_offset + lane_id * ELEMS_PER_THREAD + e] = half(combined_out[e] * inv_sum);
  }
}
