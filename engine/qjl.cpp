#include <cmath>

#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

#include "engine/qjl.h"

namespace turboquant {

mx::array generate_projection_matrix(int sketch_dim, int head_dim, int seed,
                                      mx::StreamOrDevice s) {
  // Generate random {-1, +1} matrix using uniform random and sign
  // Use key-based seeding for reproducibility
  mx::random::seed(seed);
  auto rand = mx::random::uniform(0.0f, 1.0f, {sketch_dim, head_dim}, mx::float32,
                                   std::nullopt, s);
  // sign: >= 0.5 → +1, < 0.5 → -1
  auto signs = mx::subtract(
      mx::multiply(mx::astype(mx::greater_equal(rand, mx::array(0.5f), s), mx::float32, s),
                    mx::array(2.0f), s),
      mx::array(1.0f), s);
  return mx::astype(signs, mx::int8, s);
}

std::pair<mx::array, mx::array> qjl_encode(
    const mx::array& keys,       // [B, H, S, D] float16
    const mx::array& proj_matrix, // [m, D] int8
    mx::StreamOrDevice s) {
  int B = keys.shape(0);
  int H = keys.shape(1);
  int S = keys.shape(2);
  int D = keys.shape(3);
  int m = proj_matrix.shape(0);

  // Compute norms: ||k|| for each key vector
  auto keys_f32 = mx::astype(keys, mx::float32, s);
  auto norms = mx::sqrt(mx::sum(mx::multiply(keys_f32, keys_f32, s), -1, false, s), s);
  // norms: [B, H, S]

  // Project: R · k^T for all keys
  // keys: [B, H, S, D], proj: [m, D]
  // Result: [B, H, S, m]
  auto proj_f32 = mx::astype(proj_matrix, mx::float32, s);
  auto projected = mx::matmul(keys_f32, mx::transpose(proj_f32, s), s);
  // projected: [B, H, S, m]

  // Sign quantization: >= 0 → 1, < 0 → 0
  auto bits = mx::astype(mx::greater_equal(projected, mx::array(0.0f), s), mx::uint8, s);

  // Pack bits into uint32: 32 bits per uint32
  // Reshape to [B, H, S, m/32, 32] then pack
  int packed_dim = m / 32;
  auto reshaped = mx::reshape(bits, {B * H * S, packed_dim, 32}, s);

  // Manual bit packing: multiply each bit by its positional weight and sum
  // bit_weights = [1, 2, 4, 8, ..., 2^31]
  std::vector<uint32_t> weights(32);
  for (int i = 0; i < 32; i++) weights[i] = 1u << i;
  auto bit_weights = mx::array(weights.data(), {1, 1, 32}, mx::uint32);
  auto packed = mx::sum(mx::multiply(mx::astype(reshaped, mx::uint32, s), bit_weights, s), -1, false, s);
  // packed: [B*H*S, m/32]
  packed = mx::reshape(packed, {B, H, S, packed_dim}, s);

  return {packed, mx::astype(norms, mx::float16, s)};
}

mx::array qjl_scores(
    const mx::array& Q,            // [B, Hq, Sq, D] float16
    const mx::array& K_sketches,   // [B, Hkv, Skv, m/32] uint32
    const mx::array& K_norms,      // [B, Hkv, Skv] float16
    const mx::array& proj_matrix,  // [m, D] int8
    const QJLParams& params,
    mx::StreamOrDevice s) {
  int B = Q.shape(0);
  int Hq = params.num_q_heads;
  int Hkv = params.num_kv_heads;
  int Sq = Q.shape(2);
  int Skv = K_sketches.shape(2);
  int m = params.sketch_dim;
  int D = params.head_dim;
  int gqa = Hq / Hkv;

  // Step 1: Compute Q sketches on the fly
  auto Q_f32 = mx::astype(Q, mx::float32, s);
  auto Q_norms = mx::sqrt(mx::sum(mx::multiply(Q_f32, Q_f32, s), -1, false, s), s);
  // Q_norms: [B, Hq, Sq]

  auto proj_f32 = mx::astype(proj_matrix, mx::float32, s);
  auto Q_projected = mx::matmul(Q_f32, mx::transpose(proj_f32, s), s);
  // Q_projected: [B, Hq, Sq, m]
  auto Q_bits = mx::astype(mx::greater_equal(Q_projected, mx::array(0.0f), s), mx::uint8, s);

  // Pack Q bits
  int packed_dim = m / 32;
  auto Q_reshaped = mx::reshape(Q_bits, {B * Hq * Sq, packed_dim, 32}, s);
  std::vector<uint32_t> weights(32);
  for (int i = 0; i < 32; i++) weights[i] = 1u << i;
  auto bit_weights = mx::array(weights.data(), {1, 1, 32}, mx::uint32);
  auto Q_packed = mx::sum(mx::multiply(mx::astype(Q_reshaped, mx::uint32, s), bit_weights, s), -1, false, s);
  Q_packed = mx::reshape(Q_packed, {B, Hq, Sq, packed_dim}, s);

  // Step 2: Compute XNOR popcount between Q and K sketches
  // XNOR(a, b) = ~(a ^ b), popcount gives matching bits
  // For GQA: expand K sketches to match Q heads
  auto K_sk = K_sketches;  // [B, Hkv, Skv, packed_dim]
  if (gqa > 1) {
    K_sk = mx::reshape(
        mx::repeat(mx::reshape(K_sk, {B, Hkv, 1, Skv, packed_dim}, s), gqa, 2, s),
        {B, Hq, Skv, packed_dim}, s);
  }

  // For each (q, k) pair, compute popcount(XNOR(q_sketch, k_sketch))
  // Q_packed: [B, Hq, Sq, packed_dim], K_sk: [B, Hq, Skv, packed_dim]
  // We need [B, Hq, Sq, Skv] scores

  // Approach: for each packed uint32, compute XNOR and count matching bits
  // Total matching bits = m - hamming_distance
  // Inner product estimate = (||q|| * ||k|| / m) * (2 * matching_bits - m)

  // Expand dims for broadcasting: Q[..., Sq, 1, packed] x K[..., 1, Skv, packed]
  auto Q_exp = mx::reshape(Q_packed, {B, Hq, Sq, 1, packed_dim}, s);
  auto K_exp = mx::reshape(K_sk, {B, Hq, 1, Skv, packed_dim}, s);

  // XOR to get differing bits, then count
  auto xor_result = mx::bitwise_xor(Q_exp, K_exp, s);  // [B, Hq, Sq, Skv, packed_dim]

  // Popcount: count set bits in each uint32
  // MLX doesn't have popcount, so use the bit-counting trick:
  // For each uint32, unpack to bytes and use lookup or arithmetic
  // Simpler: convert to float and use the Hamming weight formula
  // Actually, let's use a different approach: unpack bits and sum
  // For now, use a simple approach that works: cast to larger type and count

  // Hamming weight via bit manipulation (works on uint32):
  // v = v - ((v >> 1) & 0x55555555)
  // v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
  // v = ((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101 >> 24

  // MLX doesn't support all these bitwise ops easily, so let's use a simpler approach:
  // Unpack each uint32 to 32 bits and sum
  // This is memory-intensive but correct for validation

  // For validation, compute in Python-style: sum of matching bits
  // xor_result has 1s where bits differ
  // We need popcount(xor_result) = number of differing bits per uint32
  // hamming_distance = sum of popcounts across packed_dim
  // matching_bits = m - hamming_distance

  // Use byte-level counting: split uint32 into 4 bytes, use lookup
  auto xor_flat = mx::reshape(xor_result, {B * Hq * Sq * Skv, packed_dim}, s);

  // Extract bytes
  auto byte0 = mx::bitwise_and(xor_flat, mx::array(0xFFu, mx::uint32), s);
  auto byte1 = mx::bitwise_and(mx::right_shift(xor_flat, mx::array(8u, mx::uint32), s),
                                mx::array(0xFFu, mx::uint32), s);
  auto byte2 = mx::bitwise_and(mx::right_shift(xor_flat, mx::array(16u, mx::uint32), s),
                                mx::array(0xFFu, mx::uint32), s);
  auto byte3 = mx::right_shift(xor_flat, mx::array(24u, mx::uint32), s);

  // Build popcount lookup table for bytes (0-255)
  std::vector<uint32_t> popcount_lut(256);
  for (int i = 0; i < 256; i++) {
    int cnt = 0;
    int v = i;
    while (v) { cnt += v & 1; v >>= 1; }
    popcount_lut[i] = cnt;
  }
  auto lut = mx::array(popcount_lut.data(), {256}, mx::uint32);

  // Lookup popcount for each byte
  auto pc0 = mx::take(lut, mx::astype(byte0, mx::int32, s), 0, s);
  auto pc1 = mx::take(lut, mx::astype(byte1, mx::int32, s), 0, s);
  auto pc2 = mx::take(lut, mx::astype(byte2, mx::int32, s), 0, s);
  auto pc3 = mx::take(lut, mx::astype(byte3, mx::int32, s), 0, s);

  // Total differing bits per uint32
  auto diff_bits = mx::add(mx::add(pc0, pc1, s), mx::add(pc2, pc3, s), s);
  // Sum across packed_dim to get total Hamming distance
  auto hamming_dist = mx::sum(diff_bits, -1, false, s);
  // hamming_dist: [B*Hq*Sq*Skv]

  auto hamming_f32 = mx::astype(hamming_dist, mx::float32, s);
  auto matching = mx::subtract(mx::array(float(m)), hamming_f32, s);
  // Inner product estimate: (||q|| * ||k|| / m) * (2*matching - m)
  auto agreement = mx::subtract(mx::multiply(mx::array(2.0f), matching, s),
                                 mx::array(float(m)), s);
  agreement = mx::reshape(agreement, {B, Hq, Sq, Skv}, s);

  // Scale by norms
  auto Q_norms_exp = mx::reshape(Q_norms, {B, Hq, Sq, 1}, s);
  // K_norms: [B, Hkv, Skv] → expand for GQA
  auto K_norms_f32 = mx::astype(K_norms, mx::float32, s);
  if (gqa > 1) {
    K_norms_f32 = mx::reshape(
        mx::repeat(mx::reshape(K_norms_f32, {B, Hkv, 1, Skv}, s), gqa, 2, s),
        {B, Hq, Skv}, s);
  }
  auto K_norms_exp = mx::reshape(K_norms_f32, {B, Hq, 1, Skv}, s);

  auto norm_product = mx::multiply(Q_norms_exp, K_norms_exp, s);
  auto scores = mx::multiply(
      mx::divide(norm_product, mx::array(float(m)), s),
      agreement, s);

  // Apply attention scale
  scores = mx::multiply(scores, mx::array(params.attn_scale), s);

  return scores;
}

}  // namespace turboquant
