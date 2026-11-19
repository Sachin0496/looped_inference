// Fused 4-bit weight GEMV for decode:  y[T, N] = rmsnorm?(x)[T, K] @ dequant(W)[N, K]^T  (+ resid)
//
// Weight format (ours; the reference packer is engine/ref/w4_ref.py):
//   W[n, k] = q[n, k] * scale[n, g] + minv[n, g],  g = k / 64, q in 0..15, fp16 scale/minv.
//   qw   uint8 [N, K/2]   per row K/32 chunks of 16 bytes. Chunk c covers k = 32c .. 32c+31.
//                         byte j of chunk c = q[32c + j] (low nibble) | q[32c + 16 + j] << 4.
//                         A uint4 load = 32 weights. Word w of the uint4 (bytes 4w..4w+3) holds
//                         low-nibble quad w (k = 4w..4w+3 of the chunk) and high-nibble quad 4+w
//                         (k = 16+4w..16+4w+3). Each 16-byte load is aligned and contiguous.
//   meta fp16 [N, K/64, 2]  (scale, minv) per group.
//   Bytes per weight: 0.5 + 4/64 = 0.5625.
//
// Algorithm. 128 threads = 4 warps. A warp owns RPW consecutive output rows. Lane l loads chunks
// l, l+32, ... of each of its rows, so a warp-wide load is 512 contiguous bytes. Each weight byte is
// read from DRAM exactly once per call, for all T activation rows (register blocking over T).
// Activations are processed in K-slabs of SL elements. For each slab the block stages the (optionally
// RMS-normalized) x slab into shared memory as fp16, quad-interleaved so that lanes read consecutive
// 8-byte entries (conflict-free). The normalization needs the full-row sum of squares, which each
// block recomputes once up front (K * T fp16 reads, from L2). Dequantization is done in registers:
// q is put in the mantissa of 2^23 (exact), minus 2^23, then fma(q, scale, minv). Accumulation is fp32.
//
// Epilogue: y = fp16(acc + resid) when resid is given.

#include <cuda_fp16.h>
#include <stdint.h>

namespace w4 {

constexpr int THREADS = 128;
constexpr int CHUNK = 32;  // weights per 16-byte load

__device__ __forceinline__ __half2 as_h2(uint32_t u) {
  union { uint32_t u; __half2 h; } c;
  c.u = u;
  return c.h;
}
__device__ __forceinline__ uint32_t as_u32(__half2 h) {
  union { uint32_t u; __half2 h; } c;
  c.h = h;
  return c.u;
}

// w = q * s + m for nib in [0, 15]. (2^23 + q) is exact in fp32, so one subtract recovers q exactly.
__device__ __forceinline__ float dq(uint32_t nib, float s, float m) {
  return fmaf(__uint_as_float(nib | 0x4B000000u) - 8388608.0f, s, m);
}

// Two elements of fp16 x times inv, times (1 + w), rounded to fp16, same op order as the reference.
__device__ __forceinline__ uint32_t norm_pair(uint32_t xu, uint32_t wu, float inv) {
  const float2 xf = __half22float2(as_h2(xu));
  const float2 wf = __half22float2(as_h2(wu));
  return as_u32(__floats2half2_rn((xf.x * inv) * (1.f + wf.x), (xf.y * inv) * (1.f + wf.y)));
}

__device__ __forceinline__ float sumsq_u4(uint4 v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __half22float2(as_h2(w[i]));
    s += f.x * f.x + f.y * f.y;
  }
  return s;
}

template <int T, int RPW, int SL>
__global__ void __launch_bounds__(THREADS)
w4_gemv_kernel(const uint4* __restrict__ qw, const __half2* __restrict__ meta,
               const __half* __restrict__ x,        // [T, K]
               const __half* __restrict__ norm_w,   // [K] or nullptr
               const __half* __restrict__ resid,    // [T, N] or nullptr
               __half* __restrict__ y,              // [T, N]
               int N, int K, int norm, float eps) {
  static_assert(SL % 1024 == 0, "slab must be a multiple of 1024 elements");
  constexpr int CPS = SL / CHUNK;  // 16-byte chunks per slab
  constexpr int ITS = CPS / 32;    // chunk iterations per lane per slab
  constexpr int WARPS = THREADS / 32;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  uint2* xs = reinterpret_cast<uint2*>(smem_raw);   // [T][8 quads][CPS] x 4 fp16
  float* red = reinterpret_cast<float*>(smem_raw);  // [T][WARPS] scratch, used only before xs is written

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rowc = K / CHUNK;  // uint4 per row
  const int rowm = K / 64;     // half2 per row
  const int row0 = blockIdx.x * (WARPS * RPW) + warp * RPW;
  const int nslab = K / SL;

  uint4 wq[RPW][ITS];
  __half2 mt[RPW][ITS];
  float acc[RPW][T];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int t = 0; t < T; ++t) acc[r][t] = 0.f;

  // Packed weights for slab s. Issued before the prologue so the loads overlap it.
  auto load = [&](int s) {
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const size_t row = (size_t)(row0 + r);
#pragma unroll
      for (int it = 0; it < ITS; ++it) {
        const int cg = s * CPS + lane + 32 * it;  // global chunk index
        wq[r][it] = qw[row * rowc + cg];
        mt[r][it] = meta[row * rowm + (cg >> 1)];
      }
    }
  };
  load(0);

  float inv[T];
#pragma unroll
  for (int t = 0; t < T; ++t) inv[t] = 1.f;

  if (norm) {
    // Full-row sum of squares for each of the T rows (block-redundant, reads from L2).
    float ss[T];
#pragma unroll
    for (int t = 0; t < T; ++t) ss[t] = 0.f;
    for (int k0 = 8 * tid; k0 < K; k0 += 8 * THREADS) {
#pragma unroll
      for (int t = 0; t < T; ++t)
        ss[t] += sumsq_u4(*reinterpret_cast<const uint4*>(x + (size_t)t * K + k0));
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss[t] += __shfl_xor_sync(0xffffffffu, ss[t], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < T; ++t) red[t * WARPS + warp] = ss[t];
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) tot += red[t * WARPS + w];
      inv[t] = rsqrtf(tot / (float)K + eps);
    }
    __syncthreads();  // all reads of red done before xs (which aliases it) is written
  }

  constexpr int VPT = SL / 8 / THREADS;  // 8-element vectors per thread per row per slab
  for (int s = 0; s < nslab; ++s) {
    if (s > 0) load(s);

    // Stage x[:, s*SL : (s+1)*SL] (normalized if norm). Vector v holds elements 8v..8v+7 of the slab:
    // chunk v/4, quads 2*(v%4) and 2*(v%4)+1.
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const float iv = inv[t];
#pragma unroll
      for (int j = 0; j < VPT; ++j) {
        const int v = tid + THREADS * j;
        const int k = s * SL + 8 * v;
        uint4 hv = *reinterpret_cast<const uint4*>(x + (size_t)t * K + k);
        if (norm) {
          const uint4 wv = *reinterpret_cast<const uint4*>(norm_w + k);
          hv.x = norm_pair(hv.x, wv.x, iv);
          hv.y = norm_pair(hv.y, wv.y, iv);
          hv.z = norm_pair(hv.z, wv.z, iv);
          hv.w = norm_pair(hv.w, wv.w, iv);
        }
        const int cl = v >> 2, q0 = 2 * (v & 3);
        xs[(t * 8 + q0) * CPS + cl] = make_uint2(hv.x, hv.y);
        xs[(t * 8 + q0 + 1) * CPS + cl] = make_uint2(hv.z, hv.w);
      }
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < ITS; ++it) {
      const int cl = lane + 32 * it;  // chunk index within the slab
      // Dequantize the RPW chunks first, then load each x quad once and use it for all RPW rows.
      float wf[RPW][32];  // wf[r][k] for k = 0..31 of this chunk
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        const float2 sm = __half22float2(mt[r][it]);
        const uint32_t wd[4] = {wq[r][it].x, wq[r][it].y, wq[r][it].z, wq[r][it].w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            wf[r][4 * w + b] = dq((wd[w] >> (8 * b)) & 0xFu, sm.x, sm.y);
            wf[r][16 + 4 * w + b] = dq((wd[w] >> (8 * b + 4)) & 0xFu, sm.x, sm.y);
          }
        }
      }
#pragma unroll
      for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int qd = 0; qd < 8; ++qd) {
          const uint2 xv = xs[(t * 8 + qd) * CPS + cl];
          const float2 p0 = __half22float2(as_h2(xv.x));
          const float2 p1 = __half22float2(as_h2(xv.y));
#pragma unroll
          for (int r = 0; r < RPW; ++r) {
            float a = acc[r][t];
            a = fmaf(wf[r][4 * qd + 0], p0.x, a);
            a = fmaf(wf[r][4 * qd + 1], p0.y, a);
            a = fmaf(wf[r][4 * qd + 2], p1.x, a);
            a = fmaf(wf[r][4 * qd + 3], p1.y, a);
            acc[r][t] = a;
          }
        }
      }
    }
    __syncthreads();  // xs is rewritten by the next slab
  }

  // Warp-reduce each (row, t) and write.
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    const int row = row0 + r;
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) {
        if (resid != nullptr) v += __half2float(resid[(size_t)t * N + row]);
        y[(size_t)t * N + row] = __float2half_rn(v);
      }
    }
  }
}

template <int T, int RPW, int SL>
void launch(const uint8_t* qw, const __half2* meta, const __half* x, const __half* nw, const __half* rs,
            __half* y, int N, int K, int norm, float eps, cudaStream_t st) {
  constexpr int smem = T * SL * (int)sizeof(uint16_t);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(w4_gemv_kernel<T, RPW, SL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_set = true;
  }
  dim3 grid(N / (THREADS / 32 * RPW));
  w4_gemv_kernel<T, RPW, SL><<<grid, THREADS, smem, st>>>(reinterpret_cast<const uint4*>(qw), meta, x, nw, rs,
                                                          y, N, K, norm, eps);
}

}  // namespace w4

#ifdef W4_TORCH_EXTENSION
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

template <int T>
static void dispatch(int rpw, int slab, const uint8_t* qw, const __half2* meta, const __half* x,
                     const __half* nw, const __half* rs, __half* y, int N, int K, int norm, float eps,
                     cudaStream_t st) {
#define W4_CASE(R, S)                                                                  \
  if (rpw == R && slab == S) {                                                         \
    w4::launch<T, R, S>(qw, meta, x, nw, rs, y, N, K, norm, eps, st);                  \
    return;                                                                            \
  }
  W4_CASE(1, 2048) W4_CASE(2, 2048) W4_CASE(4, 2048) W4_CASE(1, 4096) W4_CASE(2, 4096) W4_CASE(4, 4096)
#undef W4_CASE
  TORCH_CHECK(false, "unsupported (rpw, slab) = (", rpw, ", ", slab, ")");
}

// x [T, K] fp16 | qw [N, K/2] uint8 | meta [N, K/64, 2] fp16 | norm_w [K] fp16 (if norm)
// resid [T, N] fp16 (optional) -> y [T, N] fp16.
torch::Tensor w4_gemv(torch::Tensor x, torch::Tensor qw, torch::Tensor meta, c10::optional<torch::Tensor> norm_w,
                      c10::optional<torch::Tensor> resid, double eps, int64_t rpw, int64_t slab) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf && x.dim() == 2 && x.is_contiguous());
  const int T = x.size(0), K = x.size(1);
  TORCH_CHECK(T == 1 || T == 2 || T == 4 || T == 8, "T must be 1, 2, 4 or 8");
  TORCH_CHECK(K % slab == 0 && slab % 1024 == 0, "K must be a multiple of the slab");
  TORCH_CHECK(qw.is_cuda() && qw.scalar_type() == torch::kUInt8 && qw.is_contiguous() && qw.size(1) == K / 2);
  const int N = qw.size(0);
  TORCH_CHECK(meta.is_cuda() && meta.scalar_type() == torch::kHalf && meta.is_contiguous() &&
              meta.size(0) == N && meta.size(1) == K / 64 && meta.size(2) == 2);
  TORCH_CHECK(N % (4 * rpw) == 0, "N must be a multiple of 4 * rpw");
  const bool norm = norm_w.has_value();
  if (norm) TORCH_CHECK(norm_w->is_cuda() && norm_w->scalar_type() == torch::kHalf && norm_w->numel() == K);
  const bool has_resid = resid.has_value();
  if (has_resid)
    TORCH_CHECK(resid->is_cuda() && resid->scalar_type() == torch::kHalf && resid->size(0) == T && resid->size(1) == N);

  auto y = torch::empty({T, N}, x.options());
  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
  const __half* nw = norm ? reinterpret_cast<const __half*>(norm_w->data_ptr<at::Half>()) : nullptr;
  const __half* rs = has_resid ? reinterpret_cast<const __half*>(resid->data_ptr<at::Half>()) : nullptr;
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  __half* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  const uint8_t* qp = qw.data_ptr<uint8_t>();
  const __half2* mp = reinterpret_cast<const __half2*>(meta.data_ptr<at::Half>());
  const int rp = (int)rpw, sl = (int)slab;
  switch (T) {
    case 1: dispatch<1>(rp, sl, qp, mp, xp, nw, rs, yp, N, K, norm, (float)eps, st); break;
    case 2: dispatch<2>(rp, sl, qp, mp, xp, nw, rs, yp, N, K, norm, (float)eps, st); break;
    case 4: dispatch<4>(rp, sl, qp, mp, xp, nw, rs, yp, N, K, norm, (float)eps, st); break;
    default: dispatch<8>(rp, sl, qp, mp, xp, nw, rs, yp, N, K, norm, (float)eps, st); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4_gemv", &w4_gemv, "fused 4-bit weight GEMV (+optional RMSNorm prologue, residual epilogue)");
}
#endif
