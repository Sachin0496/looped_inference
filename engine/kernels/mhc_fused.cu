// Fused mHC step for SenseNova-Loop decode/prefill.
//
// Replaces, in one launch per (layer, sub-block):
//   residual = merge(residual, branch_prev, post_prev, mix_prev)   (optional)
//   branch, post, mix = MHCRouter(residual)                         (RMS-normalize, 8-way GEMV, sigmoids, Sinkhorn)
//   out = Qwen3_5RMSNorm(branch)                                    ((1 + w) convention)
//
// Numerics follow engine/ref/mhc_ref.py: fp32 math, with bf16 rounding exactly where
// PyTorch rounds (bf16 matmuls accumulate in fp32 and round once; elementwise bf16 ops round per op).
// One CUDA block per token; each thread keeps its slice of the S x D residual in registers.

#include <cuda_bf16.h>
#include <stdint.h>

namespace mhc {

__device__ __forceinline__ float rnd(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template <int THREADS>
__device__ __forceinline__ float block_sum(float v, float* smem) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float t = (threadIdx.x < THREADS / 32) ? smem[threadIdx.x] : 0.f;
  if (warp == 0) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

// Reduce NV values at once (one shared row per value).
template <int THREADS, int NV>
__device__ __forceinline__ void block_sum_vec(float (&v)[NV], float* smem /* NV * THREADS/32 */) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int NW = THREADS / 32;
  #pragma unroll
  for (int i = 0; i < NV; ++i) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v[i] += __shfl_xor_sync(0xffffffffu, v[i], o);
  }
  __syncthreads();
  if (lane == 0) {
    #pragma unroll
    for (int i = 0; i < NV; ++i) smem[i * NW + warp] = v[i];
  }
  __syncthreads();
  if (warp == 0) {
    #pragma unroll
    for (int i = 0; i < NV; ++i) {
      float t = lane < NW ? smem[i * NW + lane] : 0.f;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
      if (lane == 0) smem[i * NW] = t;
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < NV; ++i) v[i] = smem[i * NW];
}

// Accurate expf, not __expf: pre is rounded to bf16 next, and a 1-ulp fp32 error can flip that rounding.
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

template <int S>
__device__ void sinkhorn_log(float (&m)[S][S], int iterations);

// Sinkhorn on an S x S matrix, same order as the reference (rows, then columns). The reference works in
// log space; dividing by row/column sums of exp(logits - max) is the same map with ~10x fewer
// transcendental ops on this serial path. Falls back to log space if any entry would underflow.
template <int S>
__device__ void sinkhorn(float (&m)[S][S], int iterations) {
  float mx = m[0][0];
  #pragma unroll
  for (int r = 0; r < S; ++r)
    #pragma unroll
    for (int c = 0; c < S; ++c) mx = fmaxf(mx, m[r][c]);
  float e[S][S], mn = 1.f;
  #pragma unroll
  for (int r = 0; r < S; ++r)
    #pragma unroll
    for (int c = 0; c < S; ++c) { e[r][c] = expf(m[r][c] - mx); mn = fminf(mn, e[r][c]); }
  if (mn < 1e-30f) { sinkhorn_log<S>(m, iterations); return; }
  for (int it = 0; it < iterations; ++it) {
    #pragma unroll
    for (int r = 0; r < S; ++r) {
      float s = 0.f;
      #pragma unroll
      for (int c = 0; c < S; ++c) s += e[r][c];
      #pragma unroll
      for (int c = 0; c < S; ++c) e[r][c] = e[r][c] / s;
    }
    #pragma unroll
    for (int c = 0; c < S; ++c) {
      float s = 0.f;
      #pragma unroll
      for (int r = 0; r < S; ++r) s += e[r][c];
      #pragma unroll
      for (int r = 0; r < S; ++r) e[r][c] = e[r][c] / s;
    }
  }
  #pragma unroll
  for (int r = 0; r < S; ++r)
    #pragma unroll
    for (int c = 0; c < S; ++c) m[r][c] = e[r][c];
}

// Log-space Sinkhorn, exactly the reference's sequence of operations.
template <int S>
__device__ void sinkhorn_log(float (&m)[S][S], int iterations) {
  for (int it = 0; it < iterations; ++it) {
    #pragma unroll
    for (int r = 0; r < S; ++r) {
      float mx = m[r][0];
      #pragma unroll
      for (int c = 1; c < S; ++c) mx = fmaxf(mx, m[r][c]);
      float s = 0.f;
      #pragma unroll
      for (int c = 0; c < S; ++c) s += expf(m[r][c] - mx);
      const float lse = mx + logf(s);
      #pragma unroll
      for (int c = 0; c < S; ++c) m[r][c] -= lse;
    }
    #pragma unroll
    for (int c = 0; c < S; ++c) {
      float mx = m[0][c];
      #pragma unroll
      for (int r = 1; r < S; ++r) mx = fmaxf(mx, m[r][c]);
      float s = 0.f;
      #pragma unroll
      for (int r = 0; r < S; ++r) s += expf(m[r][c] - mx);
      const float lse = mx + logf(s);
      #pragma unroll
      for (int r = 0; r < S; ++r) m[r][c] -= lse;
    }
  }
  #pragma unroll
  for (int r = 0; r < S; ++r)
    #pragma unroll
    for (int c = 0; c < S; ++c) m[r][c] = expf(m[r][c]);
}

union Pack8 { uint4 u; __nv_bfloat16 h[8]; };

// S streams, D hidden, THREADS per block. Requires D % (8 * THREADS) == 0.
template <int S, int D, int THREADS>
__global__ void __launch_bounds__(THREADS)
mhc_step_kernel(__nv_bfloat16* __restrict__ residual,          // [T, S, D], merged result written back
                const __nv_bfloat16* __restrict__ branch_prev,  // [T, D] or nullptr (no merge)
                const float* __restrict__ post_prev,            // [T, S]
                const float* __restrict__ mix_prev,             // [T, S, S]
                const float* __restrict__ W,                    // [2S + S*S, S*D]
                const float* __restrict__ bias,                 // [2S + S*S]
                const __nv_bfloat16* __restrict__ norm_w,       // [D]
                __nv_bfloat16* __restrict__ out,                // [T, D]
                float* __restrict__ post_out,                   // [T, S]
                float* __restrict__ mix_out,                    // [T, S, S]
                int iterations, float pre_scale, float router_eps, float norm_eps) {
  constexpr int O = 2 * S + S * S;
  constexpr int CH = D / (8 * THREADS);  // 8-wide chunks per thread per stream
  static_assert(D % (8 * THREADS) == 0, "D must be a multiple of 8*THREADS");
  __shared__ float smem[O * (THREADS / 32)];
  __shared__ float s_pre[S];

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  __nv_bfloat16* res_t = residual + (size_t)t * S * D;

  float r[S][CH][8];
  #pragma unroll
  for (int s = 0; s < S; ++s)
    #pragma unroll
    for (int c = 0; c < CH; ++c) {
      Pack8 p;
      p.u = reinterpret_cast<const uint4*>(res_t + s * D)[c * THREADS + tid];
      #pragma unroll
      for (int e = 0; e < 8; ++e) r[s][c][e] = __bfloat162float(p.h[e]);
    }

  // ---- merge: r_i = rnd( rnd(sum_j rnd(mix_ij) r_j) + rnd(rnd(post_i) * b) ) ----
  if (branch_prev != nullptr) {
    float mx[S][S], po[S];
    #pragma unroll
    for (int i = 0; i < S; ++i) {
      po[i] = rnd(post_prev[t * S + i]);
      #pragma unroll
      for (int j = 0; j < S; ++j) mx[i][j] = rnd(mix_prev[(t * S + i) * S + j]);
    }
    #pragma unroll
    for (int c = 0; c < CH; ++c) {
      Pack8 bp;
      bp.u = reinterpret_cast<const uint4*>(branch_prev + (size_t)t * D)[c * THREADS + tid];
      float nr[S][8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float b = __bfloat162float(bp.h[e]);
        #pragma unroll
        for (int i = 0; i < S; ++i) {
          float acc = 0.f;
          #pragma unroll
          for (int j = 0; j < S; ++j) acc = fmaf(mx[i][j], r[j][c][e], acc);
          nr[i][e] = rnd(rnd(acc) + rnd(po[i] * b));
        }
      }
      #pragma unroll
      for (int i = 0; i < S; ++i) {
        Pack8 p;
        #pragma unroll
        for (int e = 0; e < 8; ++e) { r[i][c][e] = nr[i][e]; p.h[e] = __float2bfloat16_rn(nr[i][e]); }
        reinterpret_cast<uint4*>(res_t + i * D)[c * THREADS + tid] = p.u;
      }
    }
  }

  // ---- router RMS over the flattened S*D vector ----
  float ss = 0.f;
  #pragma unroll
  for (int s = 0; s < S; ++s)
    #pragma unroll
    for (int c = 0; c < CH; ++c)
      #pragma unroll
      for (int e = 0; e < 8; ++e) ss = fmaf(r[s][c][e], r[s][c][e], ss);
  ss = block_sum<THREADS>(ss, smem);
  const float inv = rsqrtf(ss / (float)(S * D) + router_eps);

  // ---- O dot products against fp32 router weights ----
  float acc[O];
  #pragma unroll
  for (int o = 0; o < O; ++o) acc[o] = 0.f;
  #pragma unroll
  for (int s = 0; s < S; ++s)
    #pragma unroll
    for (int c = 0; c < CH; ++c) {
      const int k = s * D + (c * THREADS + tid) * 8;
      float nv[8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) nv[e] = r[s][c][e] * inv;
      #pragma unroll
      for (int o = 0; o < O; ++o) {
        const float4 w0 = *reinterpret_cast<const float4*>(W + (size_t)o * S * D + k);
        const float4 w1 = *reinterpret_cast<const float4*>(W + (size_t)o * S * D + k + 4);
        acc[o] = fmaf(nv[0], w0.x, acc[o]); acc[o] = fmaf(nv[1], w0.y, acc[o]);
        acc[o] = fmaf(nv[2], w0.z, acc[o]); acc[o] = fmaf(nv[3], w0.w, acc[o]);
        acc[o] = fmaf(nv[4], w1.x, acc[o]); acc[o] = fmaf(nv[5], w1.y, acc[o]);
        acc[o] = fmaf(nv[6], w1.z, acc[o]); acc[o] = fmaf(nv[7], w1.w, acc[o]);
      }
    }
  block_sum_vec<THREADS, O>(acc, smem);

  if (tid == 0) {
    float l[O];
    #pragma unroll
    for (int o = 0; o < O; ++o) l[o] = acc[o] + bias[o];
    #pragma unroll
    for (int s = 0; s < S; ++s) {
      s_pre[s] = rnd(sigmoidf_(l[s]));                     // pre.to(bf16)
      post_out[t * S + s] = 2.f * sigmoidf_(l[S + s]);
    }
    float m[S][S];
    #pragma unroll
    for (int i = 0; i < S; ++i)
      #pragma unroll
      for (int j = 0; j < S; ++j) m[i][j] = l[2 * S + i * S + j];
    sinkhorn<S>(m, iterations);
    #pragma unroll
    for (int i = 0; i < S; ++i)
      #pragma unroll
      for (int j = 0; j < S; ++j) mix_out[(t * S + i) * S + j] = m[i][j];
  }
  __syncthreads();
  float pre[S];
  #pragma unroll
  for (int s = 0; s < S; ++s) pre[s] = s_pre[s];

  // ---- branch = rnd(sum_s pre_s r_s) [* pre_scale], then RMSNorm with (1 + w) ----
  float b[CH][8];
  float ss2 = 0.f;
  #pragma unroll
  for (int c = 0; c < CH; ++c)
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      float v = 0.f;
      #pragma unroll
      for (int s = 0; s < S; ++s) v = fmaf(pre[s], r[s][c][e], v);
      v = rnd(v);
      if (pre_scale != 1.f) v = rnd(v * pre_scale);
      b[c][e] = v;
      ss2 = fmaf(v, v, ss2);
    }
  ss2 = block_sum<THREADS>(ss2, smem);
  const float inv2 = rsqrtf(ss2 / (float)D + norm_eps);
  #pragma unroll
  for (int c = 0; c < CH; ++c) {
    Pack8 wp, op;
    wp.u = reinterpret_cast<const uint4*>(norm_w)[c * THREADS + tid];
    #pragma unroll
    for (int e = 0; e < 8; ++e)
      op.h[e] = __float2bfloat16_rn(b[c][e] * inv2 * (1.f + __bfloat162float(wp.h[e])));
    reinterpret_cast<uint4*>(out + (size_t)t * D)[c * THREADS + tid] = op.u;
  }
}

// ======================= small-T (decode) path: work spread over SPLIT blocks per token ===================
// Kernel A: each block (one warp) owns D/SPLIT hidden positions across all S streams. It merges them,
// writes them back, and emits partial (sum of squares, O dot products). The last block to finish for a token
// (threadfence-reduction pattern) finalizes pre/post/mix. Logits use inv * dot(r, w) instead of
// dot(r * inv, w): same value up to fp32 rounding (~1e-7 relative).
template <int S, int D, int SPLIT>
__global__ void __launch_bounds__(32)
mhc_router_split_kernel(__nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ branch_prev,
                        const float* __restrict__ post_prev, const float* __restrict__ mix_prev,
                        const float* __restrict__ W, const float* __restrict__ bias,
                        float* __restrict__ partials,   // [T, SPLIT, 1 + O]
                        unsigned int* __restrict__ counters,  // [T], zero on entry, reset to zero on exit
                        float* __restrict__ pre_out, float* __restrict__ post_out, float* __restrict__ mix_out,
                        int iterations, float router_eps) {
  constexpr int O = 2 * S + S * S;
  constexpr int P = 1 + O;
  constexpr int DPB = D / SPLIT;          // hidden positions per block
  constexpr int CPL = DPB / (8 * 32);     // 8-wide chunks per lane per stream
  static_assert(DPB % 256 == 0, "D/SPLIT must be a multiple of 256");
  static_assert(SPLIT <= 32, "finalizer uses one lane per split");
  const int b = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
  __nv_bfloat16* res_t = residual + (size_t)t * S * D;

  float mx[S][S], po[S];
  const bool merge = branch_prev != nullptr;
  if (merge) {
    #pragma unroll
    for (int i = 0; i < S; ++i) {
      po[i] = rnd(post_prev[t * S + i]);
      #pragma unroll
      for (int j = 0; j < S; ++j) mx[i][j] = rnd(mix_prev[(t * S + i) * S + j]);
    }
  }
  float acc[P];
  #pragma unroll
  for (int p = 0; p < P; ++p) acc[p] = 0.f;

  #pragma unroll
  for (int c = 0; c < CPL; ++c) {
    const int d0 = b * DPB + (c * 32 + lane) * 8;
    float r[S][8];
    #pragma unroll
    for (int s = 0; s < S; ++s) {
      Pack8 p; p.u = *reinterpret_cast<const uint4*>(res_t + s * D + d0);
      #pragma unroll
      for (int e = 0; e < 8; ++e) r[s][e] = __bfloat162float(p.h[e]);
    }
    if (merge) {
      Pack8 bp; bp.u = *reinterpret_cast<const uint4*>(branch_prev + (size_t)t * D + d0);
      float nr[S][8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float bv = __bfloat162float(bp.h[e]);
        #pragma unroll
        for (int i = 0; i < S; ++i) {
          float a = 0.f;
          #pragma unroll
          for (int j = 0; j < S; ++j) a = fmaf(mx[i][j], r[j][e], a);
          nr[i][e] = rnd(rnd(a) + rnd(po[i] * bv));
        }
      }
      #pragma unroll
      for (int i = 0; i < S; ++i) {
        Pack8 p;
        #pragma unroll
        for (int e = 0; e < 8; ++e) { r[i][e] = nr[i][e]; p.h[e] = __float2bfloat16_rn(nr[i][e]); }
        *reinterpret_cast<uint4*>(res_t + i * D + d0) = p.u;
      }
    }
    #pragma unroll
    for (int s = 0; s < S; ++s) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) acc[0] = fmaf(r[s][e], r[s][e], acc[0]);
      #pragma unroll
      for (int o = 0; o < O; ++o) {
        const float* wp = W + (size_t)o * S * D + s * D + d0;
        const float4 w0 = *reinterpret_cast<const float4*>(wp);
        const float4 w1 = *reinterpret_cast<const float4*>(wp + 4);
        float a = acc[1 + o];
        a = fmaf(r[s][0], w0.x, a); a = fmaf(r[s][1], w0.y, a); a = fmaf(r[s][2], w0.z, a); a = fmaf(r[s][3], w0.w, a);
        a = fmaf(r[s][4], w1.x, a); a = fmaf(r[s][5], w1.y, a); a = fmaf(r[s][6], w1.z, a); a = fmaf(r[s][7], w1.w, a);
        acc[1 + o] = a;
      }
    }
  }
  #pragma unroll
  for (int p = 0; p < P; ++p)
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[p] += __shfl_xor_sync(0xffffffffu, acc[p], o);
  if (lane == 0) {
    #pragma unroll
    for (int p = 0; p < P; ++p) partials[((size_t)t * SPLIT + b) * P + p] = acc[p];
  }
  __threadfence();
  unsigned int ticket = 0;
  if (lane == 0) ticket = atomicAdd(&counters[t], 1u);
  ticket = __shfl_sync(0xffffffffu, ticket, 0);
  if (ticket != SPLIT - 1) return;

  // ---- last block for this token: finalize ----
  __threadfence();
  float v[P];
  #pragma unroll
  for (int p = 0; p < P; ++p) v[p] = lane < SPLIT ? __ldcg(&partials[((size_t)t * SPLIT + lane) * P + p]) : 0.f;
  #pragma unroll
  for (int p = 0; p < P; ++p)
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v[p] += __shfl_xor_sync(0xffffffffu, v[p], o);
  if (lane == 0) {
    const float inv = rsqrtf(v[0] / (float)(S * D) + router_eps);
    float l[O];
    #pragma unroll
    for (int o = 0; o < O; ++o) l[o] = inv * v[1 + o] + bias[o];
    #pragma unroll
    for (int s = 0; s < S; ++s) {
      pre_out[t * S + s] = rnd(sigmoidf_(l[s]));
      post_out[t * S + s] = 2.f * sigmoidf_(l[S + s]);
    }
    float m[S][S];
    #pragma unroll
    for (int i = 0; i < S; ++i)
      #pragma unroll
      for (int j = 0; j < S; ++j) m[i][j] = l[2 * S + i * S + j];
    sinkhorn<S>(m, iterations);
    #pragma unroll
    for (int i = 0; i < S; ++i)
      #pragma unroll
      for (int j = 0; j < S; ++j) mix_out[(t * S + i) * S + j] = m[i][j];
    counters[t] = 0;  // ready for the next call (keeps CUDA-graph replays valid)
  }
}

// Kernel B: branch = rnd(sum_s pre_s r_s) [* pre_scale]; RMSNorm (1 + w). Each block recomputes the full
// sum of squares from L2 (16 KB per token) and writes its own D/SPLIT slice, so no second grid reduction.
// In the full engine this prologue moves into the next GEMV, and this kernel disappears.
template <int S, int D, int SPLIT, int THREADS>
__global__ void __launch_bounds__(THREADS)
mhc_branch_norm_kernel(const __nv_bfloat16* __restrict__ residual, const float* __restrict__ pre_in,
                       const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ out,
                       float pre_scale, float norm_eps) {
  constexpr int CH = D / (8 * THREADS);
  constexpr int DPB = D / SPLIT;
  static_assert(D % (8 * THREADS) == 0 && DPB % 8 == 0, "shape");
  __shared__ float smem[THREADS / 32];
  const int b = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
  const __nv_bfloat16* res_t = residual + (size_t)t * S * D;
  float pre[S];
  #pragma unroll
  for (int s = 0; s < S; ++s) pre[s] = pre_in[t * S + s];

  auto branch8 = [&](int d0, float (&bv)[8]) {
    float r[S][8];
    #pragma unroll
    for (int s = 0; s < S; ++s) {
      Pack8 p; p.u = *reinterpret_cast<const uint4*>(res_t + s * D + d0);
      #pragma unroll
      for (int e = 0; e < 8; ++e) r[s][e] = __bfloat162float(p.h[e]);
    }
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      float v = 0.f;
      #pragma unroll
      for (int s = 0; s < S; ++s) v = fmaf(pre[s], r[s][e], v);
      v = rnd(v);
      if (pre_scale != 1.f) v = rnd(v * pre_scale);
      bv[e] = v;
    }
  };

  float ss = 0.f;
  #pragma unroll
  for (int c = 0; c < CH; ++c) {
    float bv[8];
    branch8((c * THREADS + tid) * 8, bv);
    #pragma unroll
    for (int e = 0; e < 8; ++e) ss = fmaf(bv[e], bv[e], ss);
  }
  ss = block_sum<THREADS>(ss, smem);
  const float inv = rsqrtf(ss / (float)D + norm_eps);
  for (int c = tid; c < DPB / 8; c += THREADS) {
    const int d0 = b * DPB + c * 8;
    float bv[8];
    branch8(d0, bv);
    Pack8 wp, op;
    wp.u = *reinterpret_cast<const uint4*>(norm_w + d0);
    #pragma unroll
    for (int e = 0; e < 8; ++e) op.h[e] = __float2bfloat16_rn(bv[e] * inv * (1.f + __bfloat162float(wp.h[e])));
    *reinterpret_cast<uint4*>(out + (size_t)t * D + d0) = op.u;
  }
}

}  // namespace mhc

// ---------------- host launcher (PyTorch extension) ----------------
#ifdef MHC_TORCH_EXTENSION
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

std::vector<torch::Tensor> mhc_step(torch::Tensor residual, c10::optional<torch::Tensor> branch_prev,
                                    c10::optional<torch::Tensor> post_prev, c10::optional<torch::Tensor> mix_prev,
                                    torch::Tensor W, torch::Tensor bias, torch::Tensor norm_w,
                                    int64_t iterations, double pre_scale) {
  TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == torch::kBFloat16 && residual.is_contiguous());
  TORCH_CHECK(residual.size(1) == 2 && residual.size(2) == 4096, "built for S=2, D=4096");
  TORCH_CHECK(W.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32);
  TORCH_CHECK(norm_w.scalar_type() == torch::kBFloat16);
  const int T = residual.size(0);
  auto out = torch::empty({T, 4096}, residual.options());
  auto post = torch::empty({T, 2}, residual.options().dtype(torch::kFloat32));
  auto mix = torch::empty({T, 2, 2}, residual.options().dtype(torch::kFloat32));
  const bool merge = branch_prev.has_value();
  auto stream = c10::cuda::getCurrentCUDAStream();
  mhc::mhc_step_kernel<2, 4096, 256><<<T, 256, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),
      merge ? reinterpret_cast<const __nv_bfloat16*>(branch_prev->data_ptr()) : nullptr,
      merge ? post_prev->data_ptr<float>() : nullptr,
      merge ? mix_prev->data_ptr<float>() : nullptr,
      W.data_ptr<float>(), bias.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), post.data_ptr<float>(), mix.data_ptr<float>(),
      (int)iterations, (float)pre_scale, 1e-6f, 1e-6f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {residual, out, post, mix};
}

// Decode path: split router (A) + branch/norm (B). `counters` is a persistent zeroed uint32 buffer of
// length >= T that the kernel leaves zeroed.
std::vector<torch::Tensor> mhc_step_split(torch::Tensor residual, c10::optional<torch::Tensor> branch_prev,
                                          c10::optional<torch::Tensor> post_prev, c10::optional<torch::Tensor> mix_prev,
                                          torch::Tensor W, torch::Tensor bias, torch::Tensor norm_w,
                                          torch::Tensor counters, int64_t iterations, double pre_scale) {
  TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == torch::kBFloat16 && residual.is_contiguous());
  TORCH_CHECK(residual.size(1) == 2 && residual.size(2) == 4096, "built for S=2, D=4096");
  TORCH_CHECK(counters.scalar_type() == torch::kInt32 && counters.numel() >= residual.size(0));
  constexpr int SPLIT = 16;
  const int T = residual.size(0);
  auto f32 = residual.options().dtype(torch::kFloat32);
  auto out = torch::empty({T, 4096}, residual.options());
  auto pre = torch::empty({T, 2}, f32);
  auto post = torch::empty({T, 2}, f32);
  auto mix = torch::empty({T, 2, 2}, f32);
  auto partials = torch::empty({T, SPLIT, 9}, f32);
  const bool merge = branch_prev.has_value();
  auto stream = c10::cuda::getCurrentCUDAStream();
  mhc::mhc_router_split_kernel<2, 4096, SPLIT><<<dim3(SPLIT, T), 32, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),
      merge ? reinterpret_cast<const __nv_bfloat16*>(branch_prev->data_ptr()) : nullptr,
      merge ? post_prev->data_ptr<float>() : nullptr, merge ? mix_prev->data_ptr<float>() : nullptr,
      W.data_ptr<float>(), bias.data_ptr<float>(), partials.data_ptr<float>(),
      reinterpret_cast<unsigned int*>(counters.data_ptr<int>()),
      pre.data_ptr<float>(), post.data_ptr<float>(), mix.data_ptr<float>(), (int)iterations, 1e-6f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  mhc::mhc_branch_norm_kernel<2, 4096, SPLIT, 128><<<dim3(SPLIT, T), 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()), pre.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)pre_scale, 1e-6f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {residual, out, post, mix};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mhc_step", &mhc_step, "fused mHC step, one block per token");
  m.def("mhc_step_split", &mhc_step_split, "fused mHC step, decode path split over 16 blocks per token");
}
#endif
