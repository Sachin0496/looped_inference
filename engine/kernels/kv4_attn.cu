// Latent decode attention over a compressed cache: FP4 (E2M1) latent c[t] with one E4M3 scale per 16 values,
// FP8 (E4M3) RoPE key kr[t, g] with one fp16 scale per (t, g), fp16 key scale ks[t, g]. The reference packer and
// the dequantized math are in engine/ref/kv4_attn_ref.py.
//
// Math (per sequence b, 16 query heads h, group g = h / 4, softmax scale 1/16):
//   score(h, t) = 1/16 * ( ks[t, g] * dot(qa[h], c[t]) + krs[t, g] * dot(qr[h], kr_code[t, g]) )
//   o_lat[h]    = sum_t softmax_t(score) * c[t]        (fp32, R dims per head, unnormalized partials per split)
//
// Layout of one split block (grid = (NS splits, B sequences), 128 threads, 32-token tiles):
//   stage:  the tile's packed latent bytes, scale codes (decoded to fp32), rope key codes, ks, krs -> shared memory.
//   score:  warp g owns group g (heads 4g..4g+3). Lane t owns token t of the tile, so each warp reads every
//           token's latent once and dequantizes it in registers (E2M1 via a 16-entry shared LUT, E4M3 scale and
//           rope-key codes via manual bit-level E4M3 decode). Heads' q vectors are broadcast loads (all lanes read
//           the same address). Tile max and sum are warp reductions; P = exp(s - m) goes to shared memory.
//   output: thread tid owns R/128 latent dims for all 16 heads (fp32 accumulators in registers). It re-reads the tile
//           from shared memory, dequantizes its dims, and accumulates P * c. Running accumulators are rescaled by
//           alpha = exp(m_old - m_new) before each tile (online softmax).
//   Each latent byte is read from DRAM once per call; all 16 heads consume the same dequantized c[t].
// Combine kernel: per (sequence, head), o = sum_p w_p * O_p with w_p = exp(m_p - M) / sum_p exp(m_p - M) l_p.
//
// Shared-memory bank layout: rows of the latent tile are padded to 8 mod 16 bytes so lanes reading 8-byte words
// from different tokens hit different banks; scale and P rows are padded likewise.

#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>
#include <stdint.h>
#include <algorithm>

namespace kv4 {

constexpr int H = 16;              // query heads
constexpr int G = 4;               // KV groups; head h uses group h / 4
constexpr int DR = 64;             // RoPE dims
constexpr int TT = 32;             // tokens per tile; equals the warp size (score phase: lane = token)
constexpr int THREADS = 128;       // 4 warps; warp g owns group g in the score phase
constexpr int PSTR = 20;           // P row stride in floats (80 B: 16 B aligned, conflict-free 16 B stores)
constexpr int MAX_NS = 128;        // max splits per call
constexpr int TARGET_BLOCKS = 80;  // split blocks per call (2 per SM on a 40-SM T4)
constexpr float SM_SCALE = 0.0625f;  // 1 / sqrt(256)

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

// Manual FP8 E4M3 decode (bias 7, S.EEEE.MMM, no infinities). Exact for every code except the NaN codes.
__device__ __forceinline__ float e4m3_f32(uint32_t b) {
  const uint32_t sign = (b & 0x80u) << 24;
  const uint32_t e = (b >> 3) & 0xFu;
  const uint32_t m = b & 0x7u;
  const float normal = __uint_as_float(((e + 120u) << 23) | (m << 20));  // 2^(e-7) * (1 + m/8)
  const float sub = (float)m * 0.001953125f;                              // m * 2^-9 (subnormal, e == 0)
  const float v = (e == 0u) ? sub : normal;
  return __uint_as_float(__float_as_uint(v) | sign);
}

template <int R>
__global__ void __launch_bounds__(THREADS)
split_kernel(const float* __restrict__ qa,      // [B, H, R]
             const float* __restrict__ qr,      // [B, H, DR]
             const uint8_t* __restrict__ cq,    // [B, S, R/2]
             const uint8_t* __restrict__ cs,    // [B, S, R/16]
             const uint8_t* __restrict__ kr,    // [B, S, G, DR]
             const __half* __restrict__ krs,    // [B, S, G]
             const __half* __restrict__ ks,     // [B, S, G]
             float* __restrict__ opart,         // [B, NS, H, R]
             float* __restrict__ mpart,         // [B, NS, H]
             float* __restrict__ lpart,         // [B, NS, H]
             int S, int tps) {
  constexpr int RB = R / 2;          // packed latent bytes per token
  constexpr int NB = R / 16;         // scale blocks per token
  constexpr int CSTR = RB + 8;       // padded latent row stride in bytes
  constexpr int DPT = R / THREADS;   // latent dims owned by each thread in the output phase
  static_assert(DPT == 2 || DPT == 4, "R must be 256 or 512");

  __shared__ __align__(16) uint8_t s_cq[TT * CSTR];
  __shared__ float s_scf[NB][TT + 1];          // decoded E4M3 scale of c, [block][token]
  __shared__ uint32_t s_kr[G][16][TT + 1];     // rope key codes, 4 per word, [group][word][token]
  __shared__ float s_ks[G][TT];
  __shared__ float s_krs[G][TT];
  __shared__ __align__(16) float s_P[TT][PSTR];
  __shared__ float s_alpha[H];
  __shared__ float s_lut[16];
  __shared__ float s_m[H], s_l[H];

  const int split = blockIdx.x, b = blockIdx.y;
  const int NSPL = gridDim.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tile0 = split * tps;
  const int tok_end = imin(S, (tile0 + tps) * TT);
  const size_t tokbase = (size_t)b * S;

  if (tid < 16) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};  // E2M1 magnitudes
    s_lut[tid] = (tid & 8) ? -mag[tid & 7] : mag[tid & 7];
  }

  float m_run[4], l_run[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) { m_run[i] = -1e30f; l_run[i] = 0.f; }
  float acc[H][DPT];
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int i = 0; i < DPT; ++i) acc[h][i] = 0.f;

  for (int tile = tile0; tile < tile0 + tps; ++tile) {
    const int t0 = tile * TT;

    // ---- stage the tile (zeros past the end of the cache) ----
    for (int i = tid; i < TT * (RB / 8); i += THREADS) {
      const int t = i / (RB / 8), u = i % (RB / 8);
      const int tk = t0 + t;
      uint2 v = make_uint2(0u, 0u);
      if (tk < tok_end) v = reinterpret_cast<const uint2*>(cq + (tokbase + tk) * RB)[u];
      *reinterpret_cast<uint2*>(&s_cq[t * CSTR + 8 * u]) = v;
    }
    for (int i = tid; i < TT * NB; i += THREADS) {
      const int t = i % TT, j = i / TT;
      const int tk = t0 + t;
      s_scf[j][t] = (tk < tok_end) ? e4m3_f32(cs[(tokbase + tk) * NB + j]) : 0.f;
    }
    for (int i = tid; i < TT * G * 16; i += THREADS) {
      const int t = i / (G * 16), w = i % (G * 16);   // w = g * 16 + k
      const int tk = t0 + t;
      uint32_t v = 0u;
      if (tk < tok_end) v = reinterpret_cast<const uint32_t*>(kr)[(tokbase + tk) * (G * 16) + w];
      s_kr[w / 16][w % 16][t] = v;
    }
    for (int i = tid; i < TT * G; i += THREADS) {
      const int t = i % TT, g = i / TT;
      const int tk = t0 + t;
      float a = 0.f, c = 0.f;
      if (tk < tok_end) {
        a = __half2float(ks[(tokbase + tk) * G + g]);
        c = __half2float(krs[(tokbase + tk) * G + g]);
      }
      s_ks[g][t] = a;
      s_krs[g][t] = c;
    }
    __syncthreads();

    // ---- score phase: lane = token, warp = group ----
    const int t = lane;
    const bool valid = (t0 + t) < tok_end;
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const float* qbase = qa + ((size_t)b * H + 4 * warp) * R;
#pragma unroll 1
    for (int j = 0; j < NB; ++j) {
      const uint2 pk = *reinterpret_cast<const uint2*>(&s_cq[t * CSTR + 8 * j]);
      const float sc = s_scf[j][t];
      float v[16];
      const uint32_t wd[2] = {pk.x, pk.y};
#pragma unroll
      for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int bt = 0; bt < 4; ++bt) {
          const uint32_t byte = (wd[w] >> (8 * bt)) & 0xFFu;
          v[8 * w + 2 * bt] = s_lut[byte & 0xFu] * sc;
          v[8 * w + 2 * bt + 1] = s_lut[byte >> 4] * sc;
        }
      }
#pragma unroll
      for (int hh = 0; hh < 4; ++hh) {
        const float4* qp = reinterpret_cast<const float4*>(qbase + (size_t)hh * R + 16 * j);
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const float4 qq = __ldg(qp + q4);
          s[hh] = fmaf(qq.x, v[4 * q4], s[hh]);
          s[hh] = fmaf(qq.y, v[4 * q4 + 1], s[hh]);
          s[hh] = fmaf(qq.z, v[4 * q4 + 2], s[hh]);
          s[hh] = fmaf(qq.w, v[4 * q4 + 3], s[hh]);
        }
      }
    }

    // rope term: this group's 64 key codes against the group's 4 rope queries
    float rp[4] = {0.f, 0.f, 0.f, 0.f};
    const float* qrb = qr + ((size_t)b * H + 4 * warp) * DR;
#pragma unroll 1
    for (int k = 0; k < 16; ++k) {
      const uint32_t wd = s_kr[warp][k][t];
      const float k0 = e4m3_f32(wd & 0xFFu);
      const float k1 = e4m3_f32((wd >> 8) & 0xFFu);
      const float k2 = e4m3_f32((wd >> 16) & 0xFFu);
      const float k3 = e4m3_f32(wd >> 24);
#pragma unroll
      for (int hh = 0; hh < 4; ++hh) {
        const float4 qq = __ldg(reinterpret_cast<const float4*>(qrb + hh * DR) + k);
        rp[hh] = fmaf(qq.x, k0, rp[hh]);
        rp[hh] = fmaf(qq.y, k1, rp[hh]);
        rp[hh] = fmaf(qq.z, k2, rp[hh]);
        rp[hh] = fmaf(qq.w, k3, rp[hh]);
      }
    }

    const float ksv = s_ks[warp][t], krv = s_krs[warp][t];
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
      const float x = SM_SCALE * (ksv * s[hh] + krv * rp[hh]);
      s[hh] = valid ? x : -CUDART_INF_F;
    }

    // ---- tile softmax statistics per head (warp reductions over the 32 tokens) ----
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
      float mt = s[hh];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
      const float mn = fmaxf(m_run[hh], mt);
      const float alpha = expf(m_run[hh] - mn);
      const float pv = expf(s[hh] - mn);
      float ls = pv;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, o);
      l_run[hh] = l_run[hh] * alpha + ls;
      m_run[hh] = mn;
      s[hh] = pv;
      if (lane == 0) s_alpha[4 * warp + hh] = alpha;
    }
    *reinterpret_cast<float4*>(&s_P[t][4 * warp]) = make_float4(s[0], s[1], s[2], s[3]);
    __syncthreads();

    // ---- output phase: thread owns DPT latent dims for all 16 heads ----
#pragma unroll
    for (int h = 0; h < H; ++h) {
      const float a = s_alpha[h];
#pragma unroll
      for (int i = 0; i < DPT; ++i) acc[h][i] *= a;
    }
    const int d0 = tid * DPT;
#pragma unroll 1
    for (int tt = 0; tt < TT; ++tt) {
      const float scv = s_scf[d0 / 16][tt];
      float v[DPT];
#pragma unroll
      for (int i = 0; i < DPT / 2; ++i) {
        const uint32_t byte = s_cq[tt * CSTR + d0 / 2 + i];
        v[2 * i] = s_lut[byte & 0xFu] * scv;
        v[2 * i + 1] = s_lut[byte >> 4] * scv;
      }
      const float4* pp = reinterpret_cast<const float4*>(&s_P[tt][0]);
      float pvv[H];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float4 x = pp[q];
        pvv[4 * q] = x.x;
        pvv[4 * q + 1] = x.y;
        pvv[4 * q + 2] = x.z;
        pvv[4 * q + 3] = x.w;
      }
#pragma unroll
      for (int h = 0; h < H; ++h)
#pragma unroll
        for (int i = 0; i < DPT; ++i) acc[h][i] = fmaf(pvv[h], v[i], acc[h][i]);
    }
    __syncthreads();
  }

  // ---- write this split's partials ----
  if (lane == 0) {
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
      s_m[4 * warp + hh] = m_run[hh];
      s_l[4 * warp + hh] = l_run[hh];
    }
  }
  __syncthreads();
  const size_t pidx = (size_t)b * NSPL + split;
  if (tid < H) {
    mpart[pidx * H + tid] = s_m[tid];
    lpart[pidx * H + tid] = s_l[tid];
  }
  const int d0 = tid * DPT;
#pragma unroll
  for (int h = 0; h < H; ++h)
#pragma unroll
    for (int i = 0; i < DPT; ++i) opart[(pidx * H + h) * R + d0 + i] = acc[h][i];
}

template <int R>
__global__ void __launch_bounds__(256)
combine_kernel(const float* __restrict__ opart, const float* __restrict__ mpart,
               const float* __restrict__ lpart, float* __restrict__ out, int NS) {
  __shared__ float s_m[MAX_NS], s_l[MAX_NS], s_w[MAX_NS];
  const int b = blockIdx.x / H, h = blockIdx.x % H;
  for (int p = threadIdx.x; p < NS; p += blockDim.x) {
    s_m[p] = mpart[((size_t)b * NS + p) * H + h];
    s_l[p] = lpart[((size_t)b * NS + p) * H + h];
  }
  __syncthreads();
  float M = -CUDART_INF_F;
  for (int p = 0; p < NS; ++p) M = fmaxf(M, s_m[p]);
  float den = 0.f;
  for (int p = 0; p < NS; ++p) den = fmaf(expf(s_m[p] - M), s_l[p], den);
  for (int p = threadIdx.x; p < NS; p += blockDim.x) s_w[p] = expf(s_m[p] - M) / den;
  __syncthreads();
  for (int d = threadIdx.x; d < R; d += blockDim.x) {
    float a = 0.f;
    for (int p = 0; p < NS; ++p)
      a = fmaf(s_w[p], opart[(((size_t)b * NS + p) * H + h) * R + d], a);
    out[((size_t)b * H + h) * R + d] = a;
  }
}

// ===========================================================================================================
// v2: tensor-core path (kv4_attn_tc). Same math, split-KV structure and combine kernel as v1.
// The 16 query heads are the M dimension of WMMA 16x16x16 fp16 tiles. Each split block has 128 threads and
// walks 32-token tiles. Per tile:
//   1. Packed latent bytes are dequantized once to fp16 in shared memory. E2M1 x E4M3 products are exact in fp16,
//      and E4M3 rope codes are exact in fp16. The next tile's bytes are loaded into registers before the current
//      tile's math, so DRAM reads overlap compute.
//   2. nope  S[16 x 32]   = Qh[16 x R] . C^T       WMMA, fp32 accumulate. Warps 0,1 take one 16-token block each.
//      rope  RP[16 x 128] = Qrh[16 x 64] . KR^T    WMMA. The 4 groups are 8 column blocks. Warps 2,3 take 4 each.
//   3. CUDA cores: score(h,t) = SM * (ks[t,g] * qs[h] * S[h,t] + krs[t,g] * qrs[h] * RP[h, g*32+t]), then the online
//      softmax. P = fp16(exp(score - m)). The running sum uses the fp16-rounded P, so numerator and sum agree.
//   4. O[16 x R] += P[16 x 32] . C[32 x R]         WMMA, fp32 accumulate. O stays in WMMA fragments. When the
//      running max moves, each warp rescales its own fragments through a 16x16 fp32 scratch (no fragment layout
//      assumptions).
// Q_abs and q_rope are divided by their per-(sequence, head) max-abs value (q_prep_kernel) before the fp16 rounding,
// so the MMA operands lie in [-1, 1]. The scales are applied in fp32 to the scores.
// Shared memory is about 64 KB per block on a T4, so one block runs per SM.
// ===========================================================================================================
namespace tc {
using namespace nvcuda;

constexpr int TC_KRS = 72;     // fp16 row stride of the decoded rope key tile (64 + 8)
constexpr int TC_NPSTR = 36;   // fp32 row stride of the nope score tile (32 + 4)
constexpr int TC_RPSTR = 132;  // fp32 row stride of the rope score tile (128 + 4)
constexpr int TC_PSTR = 40;    // fp16 row stride of P (32 + 8)

__host__ __device__ constexpr int al128(int x) { return (x + 127) & ~127; }

template <int R>
struct TcLayout {
  static constexpr int CSTR = R + 8;                        // fp16 row stride of the latent tile
  static constexpr int C_BYTES = TT * CSTR * 2;
  static constexpr int KR_OFF = al128(C_BYTES);
  static constexpr int KR_BYTES = G * TT * TC_KRS * 2;
  static constexpr int RP_OFF = KR_OFF + al128(KR_BYTES);
  static constexpr int RP_BYTES = H * TC_RPSTR * 4;         // also the per-warp rescale scratch (aliased)
  static constexpr int NP_OFF = RP_OFF + al128(RP_BYTES);
  static constexpr int NP_BYTES = H * TC_NPSTR * 4;
  static constexpr int P_OFF = NP_OFF + al128(NP_BYTES);
  static constexpr int P_BYTES = H * TC_PSTR * 2;
  static constexpr int KS_OFF = P_OFF + al128(P_BYTES);
  static constexpr int KS_BYTES = TT * G * 2;               // fp16 [token][group]
  static constexpr int KRS_OFF = KS_OFF + KS_BYTES;
  static constexpr int AL_OFF = KRS_OFF + KS_BYTES;
  static constexpr int AL_BYTES = H * 4;
  static constexpr int LUT_OFF = AL_OFF + AL_BYTES;
  static constexpr int TOTAL = LUT_OFF + 32;                // E2M1 table, 16 fp16
  static_assert(TOTAL <= 64512, "shared memory per block must leave room for the 1 KB system reservation");
};

// Global-to-register staging of one tile (every thread holds a fixed slice of the tile).
template <int R>
struct TcStage {
  static constexpr int CPR = R / 32;               // uint4 per token of latent bytes (R/2 bytes)
  static constexpr int NCPT = TT * CPR / THREADS;  // latent uint4 per thread
  uint4 c[NCPT];
  uint32_t sc[NCPT];   // E4M3 scale bytes of blocks 2u (low byte) and 2u+1 (high byte)
  uint4 kr[4];         // rope key codes: TT tokens x 16 uint4
  __half ks, krs;      // [token][group] for thread (token = tid/4, group = tid%4)
};

__device__ __forceinline__ __half e4m3_h(uint32_t b) { return __float2half_rn(e4m3_f32(b)); }

template <int R>
__device__ __forceinline__ void tc_stage_load(TcStage<R>& st, const uint8_t* __restrict__ cq,
                                              const uint8_t* __restrict__ cs, const uint8_t* __restrict__ kr,
                                              const __half* __restrict__ ks, const __half* __restrict__ krs,
                                              size_t tokbase, int t0, int tok_end, int tid) {
  constexpr int CPR = TcStage<R>::CPR, NCPT = TcStage<R>::NCPT, NB = R / 16;
  const uint4* cq4 = reinterpret_cast<const uint4*>(cq);
  const uint4* kr4 = reinterpret_cast<const uint4*>(kr);
#pragma unroll
  for (int k = 0; k < NCPT; ++k) {
    const int i = tid + THREADS * k, t = i / CPR, u = i % CPR, tok = t0 + t;
    if (tok < tok_end) {
      const size_t tk = tokbase + tok;
      st.c[k] = cq4[tk * CPR + u];
      st.sc[k] = *reinterpret_cast<const uint16_t*>(cs + tk * NB + 2 * u);
    } else {
      st.c[k] = make_uint4(0u, 0u, 0u, 0u);
      st.sc[k] = 0u;
    }
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int i = tid + THREADS * k, t = i >> 4, q = i & 15, tok = t0 + t;
    st.kr[k] = (tok < tok_end) ? kr4[(tokbase + tok) * 16 + q] : make_uint4(0u, 0u, 0u, 0u);
  }
  const int t = tid >> 2, g = tid & 3, tok = t0 + t;
  if (tok < tok_end) {
    st.ks = ks[(tokbase + tok) * G + g];
    st.krs = krs[(tokbase + tok) * G + g];
  } else {
    st.ks = __float2half_rn(0.f);
    st.krs = __float2half_rn(0.f);
  }
}

// Decode a staged tile into shared memory (fp16). Exact: see the header comment.
template <int R>
__device__ __forceinline__ void tc_stage_store(const TcStage<R>& st, __half* __restrict__ s_c,
                                               __half* __restrict__ s_kr, __half* __restrict__ s_ks,
                                               __half* __restrict__ s_krs, const __half* __restrict__ s_lut,
                                               int tid) {
  constexpr int CPR = TcStage<R>::CPR, NCPT = TcStage<R>::NCPT, CSTR = R + 8;
#pragma unroll
  for (int k = 0; k < NCPT; ++k) {
    const int i = tid + THREADS * k, t = i / CPR, u = i % CPR;
    const uint32_t w[4] = {st.c[k].x, st.c[k].y, st.c[k].z, st.c[k].w};
    const __half2 s0 = __half2half2(e4m3_h(st.sc[k] & 0xFFu));
    const __half2 s1 = __half2half2(e4m3_h(st.sc[k] >> 8));
    __half* dst = s_c + t * CSTR + 32 * u;
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
      for (int bt = 0; bt < 4; ++bt) {
        const uint32_t byte = (w[wi] >> (8 * bt)) & 0xFFu;
        const int kb = 4 * wi + bt;  // byte inside the 16-byte chunk: values 32u + 2kb, 32u + 2kb + 1
        const __half2 v = __hmul2(__halves2half2(s_lut[byte & 15u], s_lut[byte >> 4]), kb < 8 ? s0 : s1);
        *reinterpret_cast<__half2*>(dst + 2 * kb) = v;
      }
    }
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int i = tid + THREADS * k, t = i >> 4, q = i & 15, grp = q >> 2, d0 = (q & 3) * 16;
    const uint32_t w[4] = {st.kr[k].x, st.kr[k].y, st.kr[k].z, st.kr[k].w};
    __half* dst = s_kr + (grp * TT + t) * TC_KRS + d0;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint32_t word = w[p >> 1];
      const uint32_t b0 = (word >> (16 * (p & 1))) & 0xFFu;
      const uint32_t b1 = (word >> (16 * (p & 1) + 8)) & 0xFFu;
      *reinterpret_cast<__half2*>(dst + 2 * p) = __halves2half2(e4m3_h(b0), e4m3_h(b1));
    }
  }
  s_ks[tid] = st.ks;    // index = token * 4 + group, and tid = token * 4 + group
  s_krs[tid] = st.krs;
}

// Query prescale: x[row, :] / max|x[row, :]| in fp16, with the max kept per row. One block per row.
template <int W>
__global__ void __launch_bounds__(128) q_prep_kernel(const float* __restrict__ x, __half* __restrict__ xh,
                                                     float* __restrict__ xs) {
  __shared__ float red[4];
  const int row = blockIdx.x, tid = threadIdx.x;
  const float* xr = x + (size_t)row * W;
  float amax = 0.f;
  for (int d = tid; d < W; d += 128) amax = fmaxf(amax, fabsf(xr[d]));
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((tid & 31) == 0) red[tid >> 5] = amax;
  __syncthreads();
  amax = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  const float s = amax > 0.f ? amax : 1.f;
  if (tid == 0) xs[row] = s;
  for (int d = tid; d < W; d += 128) xh[(size_t)row * W + d] = __float2half_rn(xr[d] / s);
}

template <int R>
__global__ void __launch_bounds__(THREADS, 1)
split_tc_kernel(const __half* __restrict__ qh,    // [B, H, R]   Q_abs / qs, fp16
                const float* __restrict__ qs,     // [B, H]
                const __half* __restrict__ qrh,   // [B, H, DR]  q_rope / qrs, fp16
                const float* __restrict__ qrs,    // [B, H]
                const uint8_t* __restrict__ cq, const uint8_t* __restrict__ cs,
                const uint8_t* __restrict__ kr, const __half* __restrict__ krs, const __half* __restrict__ ks,
                float* __restrict__ opart, float* __restrict__ mpart, float* __restrict__ lpart, int S, int tps) {
  using L = TcLayout<R>;
  constexpr int CSTR = L::CSTR, NTW = R / 64;   // 16-column output blocks per warp
  static_assert(R == 256 || R == 512, "R must be 256 or 512");

  extern __shared__ __align__(128) unsigned char smem[];
  __half* s_c = reinterpret_cast<__half*>(smem);
  __half* s_kr = reinterpret_cast<__half*>(smem + L::KR_OFF);
  float* s_rp = reinterpret_cast<float*>(smem + L::RP_OFF);
  float* s_np = reinterpret_cast<float*>(smem + L::NP_OFF);
  __half* s_p = reinterpret_cast<__half*>(smem + L::P_OFF);
  __half* s_ks = reinterpret_cast<__half*>(smem + L::KS_OFF);
  __half* s_krs = reinterpret_cast<__half*>(smem + L::KRS_OFF);
  float* s_alpha = reinterpret_cast<float*>(smem + L::AL_OFF);
  __half* s_lut = reinterpret_cast<__half*>(smem + L::LUT_OFF);

  const int split = blockIdx.x, b = blockIdx.y, NSPL = gridDim.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tile0 = split * tps;
  const int tile_end = tile0 + tps;
  const int tok_end = imin(S, tile_end * TT);
  const size_t tokbase = (size_t)b * S;
  const int h = tid >> 3, sub = tid & 7, g = h >> 2;   // softmax thread: head h, tokens sub*4 .. sub*4+3
  const float qsh = qs[b * H + h], qrsh = qrs[b * H + h];

  if (tid < 16) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};  // E2M1 magnitudes
    s_lut[tid] = __float2half_rn((tid & 8) ? -mag[tid & 7] : mag[tid & 7]);
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[NTW];
#pragma unroll
  for (int j = 0; j < NTW; ++j) wmma::fill_fragment(o[j], 0.f);
  float m_run = -1e30f, l_run = 0.f;

  TcStage<R> st;
  tc_stage_load<R>(st, cq, cs, kr, ks, krs, tokbase, tile0 * TT, tok_end, tid);
  __syncthreads();  // lut
  tc_stage_store<R>(st, s_c, s_kr, s_ks, s_krs, s_lut, tid);
  __syncthreads();

  for (int tile = tile0; tile < tile_end; ++tile) {
    const int t0 = tile * TT;
    const bool more = tile + 1 < tile_end;
    if (more) tc_stage_load<R>(st, cq, cs, kr, ks, krs, tokbase, t0 + TT, tok_end, tid);

    // ---- scores ----
    if (warp < 2) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
      const __half* qb = qh + (size_t)b * H * R;
#pragma unroll 4
      for (int kk = 0; kk < R / 16; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
        wmma::load_matrix_sync(a, qb + kk * 16, R);
        wmma::load_matrix_sync(bf, s_c + warp * 16 * CSTR + kk * 16, CSTR);
        wmma::mma_sync(acc, a, bf, acc);
      }
      wmma::store_matrix_sync(s_np + warp * 16, acc, TC_NPSTR, wmma::mem_row_major);
    } else {
      const __half* qrb = qrh + (size_t)b * H * DR;
#pragma unroll 1
      for (int q = 0; q < 4; ++q) {
        const int nb = (warp - 2) * 4 + q;   // column block: group nb / 2, tokens (nb % 2) * 16 ..
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
#pragma unroll
        for (int kk = 0; kk < DR / 16; ++kk) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
          wmma::load_matrix_sync(a, qrb + kk * 16, DR);
          wmma::load_matrix_sync(bf, s_kr + nb * 16 * TC_KRS + kk * 16, TC_KRS);
          wmma::mma_sync(acc, a, bf, acc);
        }
        wmma::store_matrix_sync(s_rp + nb * 16, acc, TC_RPSTR, wmma::mem_row_major);
      }
    }
    __syncthreads();

    // ---- scale, mask, online softmax statistics; P to fp16 smem ----
    {
      float x[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int t = sub * 4 + j;
        const float nope = s_np[h * TC_NPSTR + t];
        const float rope = s_rp[h * TC_RPSTR + g * TT + t];
        const float ksv = __half2float(s_ks[t * G + g]);
        const float krv = __half2float(s_krs[t * G + g]);
        const float v = SM_SCALE * (ksv * qsh * nope + krv * qrsh * rope);
        x[j] = (t0 + t < tok_end) ? v : -CUDART_INF_F;
      }
      float mt = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
#pragma unroll
      for (int msk = 1; msk < 8; msk <<= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, msk));
      const float mn = fmaxf(m_run, mt);
      const float alpha = expf(m_run - mn);
      __half ph[4];
      float ps = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ph[j] = __float2half_rn(expf(x[j] - mn));
        ps += __half2float(ph[j]);
      }
#pragma unroll
      for (int msk = 1; msk < 8; msk <<= 1) ps += __shfl_xor_sync(0xffffffffu, ps, msk);
      l_run = l_run * alpha + ps;
      m_run = mn;
      uint2 pk;
      const __half2 p01 = __halves2half2(ph[0], ph[1]), p23 = __halves2half2(ph[2], ph[3]);
      pk.x = *reinterpret_cast<const uint32_t*>(&p01);
      pk.y = *reinterpret_cast<const uint32_t*>(&p23);
      *reinterpret_cast<uint2*>(s_p + h * TC_PSTR + sub * 4) = pk;
      if (sub == 0) s_alpha[h] = alpha;
    }
    __syncthreads();

    // ---- rescale O fragments by alpha when the running max moved (per warp, through its scratch) ----
    {
      const bool need = (lane < H) && (s_alpha[lane] != 1.f);
      if (__any_sync(0xffffffffu, need)) {
        float* scr = s_rp + warp * 256;
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
          wmma::store_matrix_sync(scr, o[j], 16, wmma::mem_row_major);
          __syncwarp();
#pragma unroll
          for (int e = lane; e < 256; e += 32) scr[e] *= s_alpha[e >> 4];
          __syncwarp();
          wmma::load_matrix_sync(o[j], scr, 16, wmma::mem_row_major);
        }
      }
    }

    // ---- O += P C ----
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::load_matrix_sync(a, s_p + kk * 16, TC_PSTR);
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
        wmma::load_matrix_sync(bf, s_c + kk * 16 * CSTR + (warp * NTW + j) * 16, CSTR);
        wmma::mma_sync(o[j], a, bf, o[j]);
      }
    }
    __syncthreads();
    if (more) {
      tc_stage_store<R>(st, s_c, s_kr, s_ks, s_krs, s_lut, tid);
      __syncthreads();
    }
  }

  // ---- write this split's partials (same layout as v1) ----
  const size_t pidx = (size_t)b * NSPL + split;
  float* ob = opart + pidx * H * R;
#pragma unroll
  for (int j = 0; j < NTW; ++j) wmma::store_matrix_sync(ob + (warp * NTW + j) * 16, o[j], R, wmma::mem_row_major);
  if (sub == 0) {
    mpart[pidx * H + h] = m_run;
    lpart[pidx * H + h] = l_run;
  }
}

template <int R>
void set_tc_smem_once() {
  static bool done = false;
  if (!done) {
    cudaFuncSetAttribute(split_tc_kernel<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, TcLayout<R>::TOTAL);
    done = true;
  }
}

}  // namespace tc

}  // namespace kv4

#ifdef KV4_TORCH_EXTENSION
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// qa [B, 16, R] fp32 | qr [B, 16, 64] fp32 | cq [B, S, R/2] u8 | cs [B, S, R/16] u8 | kr [B, S, 4, 64] u8
// krs [B, S, 4] fp16 | ks [B, S, 4] fp16  ->  o_lat [B, 16, R] fp32.   R in {256, 512}.
torch::Tensor kv4_attn(torch::Tensor qa, torch::Tensor qr, torch::Tensor cq, torch::Tensor cs, torch::Tensor kr,
                       torch::Tensor krs, torch::Tensor ks) {
  using namespace kv4;
  TORCH_CHECK(qa.is_cuda() && qa.scalar_type() == torch::kFloat && qa.dim() == 3 && qa.is_contiguous());
  const int B = qa.size(0), R = qa.size(2);
  TORCH_CHECK(qa.size(1) == H && (R == 256 || R == 512), "qa must be [B, 16, 256|512]");
  TORCH_CHECK(qr.is_cuda() && qr.scalar_type() == torch::kFloat && qr.is_contiguous() && qr.dim() == 3 &&
              qr.size(0) == B && qr.size(1) == H && qr.size(2) == DR);
  TORCH_CHECK(cq.is_cuda() && cq.scalar_type() == torch::kUInt8 && cq.is_contiguous() && cq.dim() == 3 &&
              cq.size(0) == B && cq.size(2) == R / 2);
  const int S = cq.size(1);
  TORCH_CHECK(cs.is_cuda() && cs.scalar_type() == torch::kUInt8 && cs.is_contiguous() && cs.dim() == 3 &&
              cs.size(0) == B && cs.size(1) == S && cs.size(2) == R / 16);
  TORCH_CHECK(kr.is_cuda() && kr.scalar_type() == torch::kUInt8 && kr.is_contiguous() && kr.dim() == 4 &&
              kr.size(0) == B && kr.size(1) == S && kr.size(2) == G && kr.size(3) == DR);
  TORCH_CHECK(krs.is_cuda() && krs.scalar_type() == torch::kHalf && krs.is_contiguous() && krs.dim() == 3 &&
              krs.size(0) == B && krs.size(1) == S && krs.size(2) == G);
  TORCH_CHECK(ks.is_cuda() && ks.scalar_type() == torch::kHalf && ks.is_contiguous() && ks.dim() == 3 &&
              ks.size(0) == B && ks.size(1) == S && ks.size(2) == G);

  // split the sequence so that the call has about TARGET_BLOCKS blocks in total
  const int ntiles = (S + TT - 1) / TT;
  int NS = std::min(ntiles, std::max(1, TARGET_BLOCKS / B));
  const int tps = (ntiles + NS - 1) / NS;
  NS = (ntiles + tps - 1) / tps;
  TORCH_CHECK(NS <= MAX_NS, "too many splits");

  auto opts = qa.options();
  auto opart = torch::empty({B, NS, H, R}, opts);
  auto mpart = torch::empty({B, NS, H}, opts);
  auto lpart = torch::empty({B, NS, H}, opts);
  auto out = torch::empty({B, H, R}, opts);

  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
  const float* qap = qa.data_ptr<float>();
  const float* qrp = qr.data_ptr<float>();
  const uint8_t* cqp = cq.data_ptr<uint8_t>();
  const uint8_t* csp = cs.data_ptr<uint8_t>();
  const uint8_t* krp = kr.data_ptr<uint8_t>();
  const __half* krsp = reinterpret_cast<const __half*>(krs.data_ptr<at::Half>());
  const __half* ksp = reinterpret_cast<const __half*>(ks.data_ptr<at::Half>());
  float* op = opart.data_ptr<float>();
  float* mp = mpart.data_ptr<float>();
  float* lp = lpart.data_ptr<float>();
  float* outp = out.data_ptr<float>();

  dim3 grid(NS, B);
  if (R == 256) {
    split_kernel<256><<<grid, THREADS, 0, st>>>(qap, qrp, cqp, csp, krp, krsp, ksp, op, mp, lp, S, tps);
    combine_kernel<256><<<B * H, 256, 0, st>>>(op, mp, lp, outp, NS);
  } else {
    split_kernel<512><<<grid, THREADS, 0, st>>>(qap, qrp, cqp, csp, krp, krsp, ksp, op, mp, lp, S, tps);
    combine_kernel<512><<<B * H, 256, 0, st>>>(op, mp, lp, outp, NS);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// v2: same contract as kv4_attn. Tensor-core path (WMMA fp16 operands, fp32 accumulate); see namespace tc above.
torch::Tensor kv4_attn_tc(torch::Tensor qa, torch::Tensor qr, torch::Tensor cq, torch::Tensor cs, torch::Tensor kr,
                          torch::Tensor krs, torch::Tensor ks) {
  using namespace kv4;
  TORCH_CHECK(qa.is_cuda() && qa.scalar_type() == torch::kFloat && qa.dim() == 3 && qa.is_contiguous());
  const int B = qa.size(0), R = qa.size(2);
  TORCH_CHECK(qa.size(1) == H && (R == 256 || R == 512), "qa must be [B, 16, 256|512]");
  TORCH_CHECK(qr.is_cuda() && qr.scalar_type() == torch::kFloat && qr.is_contiguous() && qr.dim() == 3 &&
              qr.size(0) == B && qr.size(1) == H && qr.size(2) == DR);
  TORCH_CHECK(cq.is_cuda() && cq.scalar_type() == torch::kUInt8 && cq.is_contiguous() && cq.dim() == 3 &&
              cq.size(0) == B && cq.size(2) == R / 2);
  const int S = cq.size(1);
  TORCH_CHECK(cs.is_cuda() && cs.scalar_type() == torch::kUInt8 && cs.is_contiguous() && cs.dim() == 3 &&
              cs.size(0) == B && cs.size(1) == S && cs.size(2) == R / 16);
  TORCH_CHECK(kr.is_cuda() && kr.scalar_type() == torch::kUInt8 && kr.is_contiguous() && kr.dim() == 4 &&
              kr.size(0) == B && kr.size(1) == S && kr.size(2) == G && kr.size(3) == DR);
  TORCH_CHECK(krs.is_cuda() && krs.scalar_type() == torch::kHalf && krs.is_contiguous() && krs.dim() == 3 &&
              krs.size(0) == B && krs.size(1) == S && krs.size(2) == G);
  TORCH_CHECK(ks.is_cuda() && ks.scalar_type() == torch::kHalf && ks.is_contiguous() && ks.dim() == 3 &&
              ks.size(0) == B && ks.size(1) == S && ks.size(2) == G);

  // one split block per SM (the kernel uses ~64 KB shared memory)
  int dev = 0, nsm = 40;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const int ntiles = (S + TT - 1) / TT;
  int NS = std::min(ntiles, std::max(1, nsm / B));
  const int tps = (ntiles + NS - 1) / NS;
  NS = (ntiles + tps - 1) / tps;
  TORCH_CHECK(NS <= MAX_NS, "too many splits");

  auto opts = qa.options();
  auto hopts = opts.dtype(torch::kHalf);
  auto qh = torch::empty({B, H, R}, hopts);
  auto qrh = torch::empty({B, H, DR}, hopts);
  auto qsc = torch::empty({B, H}, opts);
  auto qrsc = torch::empty({B, H}, opts);
  auto opart = torch::empty({B, NS, H, R}, opts);
  auto mpart = torch::empty({B, NS, H}, opts);
  auto lpart = torch::empty({B, NS, H}, opts);
  auto out = torch::empty({B, H, R}, opts);

  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
  const float* qap = qa.data_ptr<float>();
  const float* qrp = qr.data_ptr<float>();
  __half* qhp = reinterpret_cast<__half*>(qh.data_ptr<at::Half>());
  __half* qrhp = reinterpret_cast<__half*>(qrh.data_ptr<at::Half>());
  float* qsp = qsc.data_ptr<float>();
  float* qrsp = qrsc.data_ptr<float>();
  const uint8_t* cqp = cq.data_ptr<uint8_t>();
  const uint8_t* csp = cs.data_ptr<uint8_t>();
  const uint8_t* krp = kr.data_ptr<uint8_t>();
  const __half* krsp = reinterpret_cast<const __half*>(krs.data_ptr<at::Half>());
  const __half* ksp = reinterpret_cast<const __half*>(ks.data_ptr<at::Half>());
  float* op = opart.data_ptr<float>();
  float* mp = mpart.data_ptr<float>();
  float* lp = lpart.data_ptr<float>();
  float* outp = out.data_ptr<float>();

  dim3 grid(NS, B);
  // query prescale first (Q_abs and q_rope in fp16, per-(sequence, head) scales), then the split kernel, then combine
  if (R == 256) tc::q_prep_kernel<256><<<B * H, 128, 0, st>>>(qap, qhp, qsp);
  else tc::q_prep_kernel<512><<<B * H, 128, 0, st>>>(qap, qhp, qsp);
  tc::q_prep_kernel<DR><<<B * H, 128, 0, st>>>(qrp, qrhp, qrsp);
  if (R == 256) {
    tc::set_tc_smem_once<256>();
    tc::split_tc_kernel<256><<<grid, THREADS, tc::TcLayout<256>::TOTAL, st>>>(
        qhp, qsp, qrhp, qrsp, cqp, csp, krp, krsp, ksp, op, mp, lp, S, tps);
    combine_kernel<256><<<B * H, 256, 0, st>>>(op, mp, lp, outp, NS);
  } else {
    tc::set_tc_smem_once<512>();
    tc::split_tc_kernel<512><<<grid, THREADS, tc::TcLayout<512>::TOTAL, st>>>(
        qhp, qsp, qrhp, qrsp, cqp, csp, krp, krsp, ksp, op, mp, lp, S, tps);
    combine_kernel<512><<<B * H, 256, 0, st>>>(op, mp, lp, outp, NS);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kv4_attn", &kv4_attn, "latent decode attention over an FP4 compressed cache (split-KV)");
  m.def("kv4_attn_tc", &kv4_attn_tc, "tensor-core (WMMA fp16) latent decode attention, split-KV");
}
#endif
