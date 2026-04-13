#include <metal_stdlib>
using namespace metal;

// Fused Scaled Dot-Product Attention with Int4 KV Cache — v3 (Split-K)
//
// Two-phase approach:
//   Phase 1: sdpa_int4_partial — each threadgroup handles a chunk of the KV sequence,
//            producing partial output, partial log-sum-exp, and partial max.
//   Phase 2: sdpa_int4_reduce  — combine partial results across chunks using
//            the online softmax correction.
//
// For short sequences (Skv <= CHUNK_SIZE), we fall back to the single-pass kernel.

constant constexpr int BN = 32;        // inner block size for online softmax
constant constexpr int D = 128;        // head_dim
constant constexpr int D_HALF = 64;    // packed int4 width in bytes
constant constexpr int SIMD_SIZE = 32; // Apple GPU SIMD width
constant constexpr int ELEMS_PER_THREAD = D / SIMD_SIZE;  // 4

struct SdpaInt4Params {
  int num_q_heads;
  int num_kv_heads;
  int seq_len_q;
  int seq_len_kv;
  int group_size;
  float attn_scale;
};

// Split-K params: how many chunks to divide KV into
struct SplitKParams {
  int num_chunks;     // number of KV chunks
  int chunk_size;     // tokens per chunk (last chunk may be smaller)
};

// ============================================================
// Single-pass kernel (for short sequences or fallback)
// ============================================================

[[kernel]] void sdpa_int4_kernel(
    device const half* Q             [[buffer(0)]],
    device const uchar* K_quant      [[buffer(1)]],
    device const half* K_scales      [[buffer(2)]],
    device const half* K_zeros       [[buffer(3)]],
    device const uchar* V_quant      [[buffer(4)]],
    device const half* V_scales      [[buffer(5)]],
    device const half* V_zeros       [[buffer(6)]],
    device half* output              [[buffer(7)]],
    constant SdpaInt4Params& params  [[buffer(8)]],
    uint2 group_pos  [[threadgroup_position_in_grid]],
    uint lane_id     [[thread_index_in_simdgroup]]) {

  const int Hq = params.num_q_heads;
  const int Hkv = params.num_kv_heads;
  const int Sq = params.seq_len_q;
  const int Skv = params.seq_len_kv;
  const int gs = params.group_size;
  const float scale = params.attn_scale;
  const int gqa_ratio = Hq / Hkv;

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
  int groups_per_vec = D / gs;
  int start_elem = lane_id * ELEMS_PER_THREAD;
  int num_blocks = (Skv + BN - 1) / BN;

  for (int blk = 0; blk < num_blocks; blk++) {
    int block_start = blk * BN;
    int block_end = min(block_start + BN, Skv);
    int block_len = block_end - block_start;

    // Phase 1: QK scores
    float scores[BN];
    for (int bn = 0; bn < block_len; bn++) {
      int kv_vec_offset = kv_base + block_start + bn;
      int group_idx = start_elem / gs;
      float k_scale = float(K_scales[kv_vec_offset * groups_per_vec + group_idx]);
      float k_zero = float(K_zeros[kv_vec_offset * groups_per_vec + group_idx]);
      device const uchar* k_packed = K_quant + kv_vec_offset * D_HALF;
      uint16_t raw16 = *reinterpret_cast<device const uint16_t*>(k_packed + start_elem / 2);

      float dot = 0.0f;
      dot += q_local[0] * ((float(raw16 & 0xF) - k_zero) * k_scale);
      dot += q_local[1] * ((float((raw16 >> 4) & 0xF) - k_zero) * k_scale);
      dot += q_local[2] * ((float((raw16 >> 8) & 0xF) - k_zero) * k_scale);
      dot += q_local[3] * ((float((raw16 >> 12) & 0xF) - k_zero) * k_scale);
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

    // Phase 3: Weighted V accumulation
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
// Grid:  threadgroups [Hq * Sq * num_chunks, B, 1]
// Group: [SIMD_SIZE, 1, 1]
//
// Outputs per (query, chunk):
//   partial_out:     [B, Hq, Sq, num_chunks, D] float32
//   partial_max:     [B, Hq, Sq, num_chunks] float32
//   partial_sum_exp: [B, Hq, Sq, num_chunks] float32

[[kernel]] void sdpa_int4_partial(
    device const half* Q              [[buffer(0)]],
    device const uchar* K_quant       [[buffer(1)]],
    device const half* K_scales       [[buffer(2)]],
    device const half* K_zeros        [[buffer(3)]],
    device const uchar* V_quant       [[buffer(4)]],
    device const half* V_scales       [[buffer(5)]],
    device const half* V_zeros        [[buffer(6)]],
    device float* partial_out         [[buffer(7)]],   // [B, Hq, Sq, num_chunks, D]
    device float* partial_max         [[buffer(8)]],   // [B, Hq, Sq, num_chunks]
    device float* partial_sum_exp     [[buffer(9)]],   // [B, Hq, Sq, num_chunks]
    constant SdpaInt4Params& params   [[buffer(10)]],
    constant SplitKParams& sk_params  [[buffer(11)]],
    uint2 group_pos  [[threadgroup_position_in_grid]],
    uint lane_id     [[thread_index_in_simdgroup]]) {

  const int Hq = params.num_q_heads;
  const int Hkv = params.num_kv_heads;
  const int Sq = params.seq_len_q;
  const int Skv = params.seq_len_kv;
  const int gs = params.group_size;
  const float scale = params.attn_scale;
  const int gqa_ratio = Hq / Hkv;
  const int num_chunks = sk_params.num_chunks;
  const int chunk_size = sk_params.chunk_size;

  int batch_idx = group_pos.y;
  int flat_idx = group_pos.x;  // in [0, Hq * Sq * num_chunks)
  int chunk_idx = flat_idx % num_chunks;
  int flat_qh = flat_idx / num_chunks;
  int q_head = flat_qh / Sq;
  int q_pos = flat_qh % Sq;
  int kv_head = q_head / gqa_ratio;

  // This chunk's KV range
  int chunk_start = chunk_idx * chunk_size;
  int chunk_end = min(chunk_start + chunk_size, Skv);
  if (chunk_start >= Skv) return;  // past end

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
  int groups_per_vec = D / gs;
  int start_elem = lane_id * ELEMS_PER_THREAD;

  // Process this chunk in blocks of BN
  int num_blocks = (chunk_end - chunk_start + BN - 1) / BN;

  for (int blk = 0; blk < num_blocks; blk++) {
    int block_start = chunk_start + blk * BN;
    int block_end = min(block_start + BN, chunk_end);
    int block_len = block_end - block_start;

    float scores[BN];
    for (int bn = 0; bn < block_len; bn++) {
      int kv_vec_offset = kv_base + block_start + bn;
      int group_idx = start_elem / gs;
      float k_scale = float(K_scales[kv_vec_offset * groups_per_vec + group_idx]);
      float k_zero = float(K_zeros[kv_vec_offset * groups_per_vec + group_idx]);
      device const uchar* k_packed = K_quant + kv_vec_offset * D_HALF;
      uint16_t raw16 = *reinterpret_cast<device const uint16_t*>(k_packed + start_elem / 2);

      float dot = 0.0f;
      dot += q_local[0] * ((float(raw16 & 0xF) - k_zero) * k_scale);
      dot += q_local[1] * ((float((raw16 >> 4) & 0xF) - k_zero) * k_scale);
      dot += q_local[2] * ((float((raw16 >> 8) & 0xF) - k_zero) * k_scale);
      dot += q_local[3] * ((float((raw16 >> 12) & 0xF) - k_zero) * k_scale);
      scores[bn] = simd_sum(dot);
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
  // partial_out: [B, Hq, Sq, num_chunks, D]
  int po_base = (((batch_idx * Hq + q_head) * Sq + q_pos) * num_chunks + chunk_idx) * D;
  for (int e = 0; e < ELEMS_PER_THREAD; e++) {
    partial_out[po_base + lane_id * ELEMS_PER_THREAD + e] = out_accum[e];
  }

  // Only lane 0 writes scalar partials (all lanes have same value after simd_sum)
  if (lane_id == 0) {
    int ps_base = ((batch_idx * Hq + q_head) * Sq + q_pos) * num_chunks + chunk_idx;
    partial_max[ps_base] = max_score;
    partial_sum_exp[ps_base] = sum_exp;
  }
}

// ============================================================
// Split-K Phase 2: Reduce partial results
// ============================================================
// Grid:  threadgroups [Hq * Sq, B, 1]
// Group: [SIMD_SIZE, 1, 1]
//
// Each threadgroup combines num_chunks partial results for one query position.

[[kernel]] void sdpa_int4_reduce(
    device const float* partial_out       [[buffer(0)]],  // [B, Hq, Sq, num_chunks, D]
    device const float* partial_max       [[buffer(1)]],  // [B, Hq, Sq, num_chunks]
    device const float* partial_sum_exp   [[buffer(2)]],  // [B, Hq, Sq, num_chunks]
    device half* output                   [[buffer(3)]],   // [B, Hq, Sq, D]
    constant SdpaInt4Params& params       [[buffer(4)]],
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

  // Find global max across chunks
  float global_max = -INFINITY;
  for (int c = 0; c < num_chunks; c++) {
    global_max = max(global_max, partial_max[ps_base + c]);
  }

  // Combine: for each chunk, correct its sum_exp and output by exp(chunk_max - global_max)
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

  // Normalize and write final output
  float inv_sum = (total_sum_exp > 0.0f) ? (1.0f / total_sum_exp) : 0.0f;
  int out_offset = ((batch_idx * Hq + q_head) * Sq + q_pos) * D;
  for (int e = 0; e < ELEMS_PER_THREAD; e++) {
    output[out_offset + lane_id * ELEMS_PER_THREAD + e] = half(combined_out[e] * inv_sum);
  }
}
