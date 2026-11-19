// Fused Gated DeltaNet decode step (one new token per sequence) for SenseNova-Loop-9B (Qwen3.5 linear attention).
//
// Replaces, in ONE launch, everything between the in_proj outputs and the out_proj input:
//   causal_conv1d_update + SiLU (q, k, v channels)       -> engine/ref/deltanet_ref.py causal_conv1d_update
//   q/k L2 norm, q / sqrt(128)                           -> l2norm, torch_recurrent_gated_delta_rule
//   g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)   (beta rounded to bf16 as PyTorch does)
//   GQA: v-head h uses q/k head h / 2
//   S <- S * exp(g);  kv = S^T k;  delta = (v - kv) * beta;  S <- S + k delta^T;  o = S^T q
//   gated RMSNorm with SiLU(z), weight NOT zero-centred                     -> rmsnorm_gated
//
// Launch: grid (16 k-heads, T sequences), block 512 threads. Block (kh, s) owns the two v-heads 2kh and
// 2kh+1 that read k-head kh. Each v-head uses 256 threads; thread (kg = lt/32, v4 = lt%32) keeps the
// 16 x 4 fp32 state tile S[kg*16 .. +16][4*v4 .. +4] in registers for the whole step. The state is read
// once and written once (2 MiB per sequence each way). The q/k/v conv channels of a block are owned by
// that block alone, so the conv state update has no cross-block race (a per-v-head grid would need the
// neighbouring v-head to read conv state that this block overwrites).
// Dependencies inside the block are reduced through shared memory (k-groups for kv and o, the warp
// shuffle for the l2norm and the gated-norm variance).
//
// Numerics: fp32 math; bf16 rounding exactly where PyTorch rounds (conv output, SiLU output, beta,
// core attention output, gated-norm intermediates, final output). State stays fp32.

#include <cuda_bf16.h>
#include <math.h>

namespace dn {

constexpr int HK = 16, HV = 32, DK = 128, DV = 128, CK = 4;
constexpr int KEY_DIM = HK * DK;          // 2048
constexpr int VALUE_DIM = HV * DV;        // 4096
constexpr int CONV_DIM = 2 * KEY_DIM + VALUE_DIM;  // 8192
constexpr int KG = 8;                     // k-groups per v-head (one warp each)
constexpr int KR = DK / KG;               // 16 state rows per thread
constexpr int NTHREADS = 512;
constexpr float EPS = 1e-6f;
constexpr float SQRT_DK = 11.313708498984761f;   // sqrt(128), as a float32 literal

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rnd(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__global__ void __launch_bounds__(NTHREADS)
deltanet_decode_kernel(const __nv_bfloat16* __restrict__ qkv,      // [T, 8192]
                       const __nv_bfloat16* __restrict__ z,        // [T, 4096]
                       const __nv_bfloat16* __restrict__ b_in,     // [T, 32]
                       const __nv_bfloat16* __restrict__ a_in,     // [T, 32]
                       __nv_bfloat16* conv_state,                  // [T, 8192, 4]  in/out
                       float* rec_state,                           // [T, 32, 128, 128] in/out
                       const float* __restrict__ A_log,            // [32]
                       const float* __restrict__ dt_bias,          // [32]
                       const __nv_bfloat16* __restrict__ conv_w,   // [8192, 4]
                       const __nv_bfloat16* __restrict__ norm_w,   // [128]
                       __nv_bfloat16* __restrict__ out) {          // [T, 4096]
  const int kh = blockIdx.x;                 // k-head 0..15; owns v-heads 2kh, 2kh+1
  const int s = blockIdx.y;                  // sequence
  const int t = threadIdx.x;
  const int warp = t >> 5, lane = t & 31;
  const int j = t >> 8;                      // local v-head (0 or 1) for the recurrence phase
  const int lt = t & 255;                    // thread index within the v-head
  const int h = 2 * kh + j;                  // global v-head
  const int kg = lt >> 5;                    // k-group == warp index within the v-head
  const int v4 = lt & 31;                    // v quad: columns 4*v4 .. 4*v4+3
  const int kbase = kg * KR;

  __shared__ __align__(16) float sh_q[DK];
  __shared__ __align__(16) float sh_k[DK];
  __shared__ __align__(16) float sh_v[2][DV];
  __shared__ float sh_g[2], sh_beta[2];
  __shared__ float sh_sq[8];
  __shared__ __align__(16) float red_kv[2][KG][DV];
  __shared__ __align__(16) float red_o[2][KG][DV];

  // (1) Issue the recurrent-state loads first; they overlap with the conv phase.
  float* S_h = rec_state + ((size_t)s * HV + h) * DK * DV;
  float4 st[KR];
#pragma unroll
  for (int kk = 0; kk < KR; ++kk)
    st[kk] = reinterpret_cast<const float4*>(S_h + (kbase + kk) * DV)[v4];

  // (2) Causal conv1d update + SiLU, one channel per thread for the block's 512 channels:
  //     t < 128: q channel kh*128 + t; t < 256: k channel kh*128 + t-128; else v channels 2kh*128 + (t-256).
  const int ch = t < 128 ? kh * DK + t
               : (t < 256 ? KEY_DIM + kh * DK + (t - 128) : 2 * KEY_DIM + kh * (2 * DV) + (t - 256));
  __nv_bfloat16* cs = conv_state + ((size_t)s * CONV_DIM + ch) * CK;
  const float x = bf2f(qkv[(size_t)s * CONV_DIM + ch]);
  const float o1 = bf2f(cs[1]), o2 = bf2f(cs[2]), o3 = bf2f(cs[3]);
  const float w0 = bf2f(conv_w[ch * CK + 0]), w1 = bf2f(conv_w[ch * CK + 1]);
  const float w2 = bf2f(conv_w[ch * CK + 2]), w3 = bf2f(conv_w[ch * CK + 3]);
  // window = [old1, old2, old3, x]; the new conv state is the same window (last 4 of cat(state, x)).
  const float conv_out = rnd(w0 * o1 + w1 * o2 + w2 * o3 + w3 * x);   // F.conv1d output, bf16
  const float act = rnd(conv_out / (1.f + expf(-conv_out)));           // SiLU, bf16
  cs[0] = f2bf(o1);
  cs[1] = f2bf(o2);
  cs[2] = f2bf(o3);
  cs[3] = f2bf(x);
  if (t < 128) sh_q[t] = act;
  else if (t < 256) sh_k[t - 128] = act;
  else sh_v[(t - 256) >> 7][(t - 256) & 127] = act;

  // Gates for the two v-heads: beta = bf16(sigmoid(b)), g = -exp(A_log) * softplus(a + dt_bias) in fp32.
  if (lt == 0) {
    const float bl = bf2f(b_in[s * HV + h]);
    const float al = bf2f(a_in[s * HV + h]);
    sh_beta[j] = rnd(1.f / (1.f + expf(-bl)));
    const float xs = al + dt_bias[h];
    const float sp = xs > 20.f ? xs : log1pf(expf(xs));                 // softplus, threshold 20
    sh_g[j] = -expf(A_log[h]) * sp;
  }
  __syncthreads();                                                      // (A)

  // (3) L2-normalise q and k (sum of squares over 128, eps added to the sum), q also / sqrt(128).
  if (t < 256) {
    const float v = t < 128 ? sh_q[t] : sh_k[t - 128];
    float ss = v * v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) sh_sq[warp] = ss;                                    // warps 0-3: q, 4-7: k
  }
  __syncthreads();                                                      // (B)
  if (t < 256) {
    const int grp = warp >> 2;
    const float tot = sh_sq[grp * 4] + sh_sq[grp * 4 + 1] + sh_sq[grp * 4 + 2] + sh_sq[grp * 4 + 3];
    const float inv = rsqrtf(tot + EPS);
    if (t < 128) sh_q[t] = (sh_q[t] * inv) / SQRT_DK;
    else sh_k[t - 128] = sh_k[t - 128] * inv;
  }
  __syncthreads();                                                      // (C)

  // (4) Decay the state, then kv = S^T k (partial over this thread's 16 rows).
  const float decay = expf(sh_g[j]);
  float4 kvp = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int kk = 0; kk < KR; ++kk) {
    st[kk].x *= decay; st[kk].y *= decay; st[kk].z *= decay; st[kk].w *= decay;
    const float kv_ = sh_k[kbase + kk];
    kvp.x += st[kk].x * kv_; kvp.y += st[kk].y * kv_; kvp.z += st[kk].z * kv_; kvp.w += st[kk].w * kv_;
  }
  *reinterpret_cast<float4*>(&red_kv[j][kg][4 * v4]) = kvp;
  __syncthreads();                                                      // (D)

  // (5) Full kv for my 4 columns, delta = (v - kv) * beta, rank-1 update, o = S^T q.
  float4 kv = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int g = 0; g < KG; ++g) {
    const float4 r = *reinterpret_cast<const float4*>(&red_kv[j][g][4 * v4]);
    kv.x += r.x; kv.y += r.y; kv.z += r.z; kv.w += r.w;
  }
  const float4 vv = *reinterpret_cast<const float4*>(&sh_v[j][4 * v4]);
  const float beta = sh_beta[j];
  const float4 dl = make_float4((vv.x - kv.x) * beta, (vv.y - kv.y) * beta,
                                (vv.z - kv.z) * beta, (vv.w - kv.w) * beta);
  float4 op = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int kk = 0; kk < KR; ++kk) {
    const float kk_ = sh_k[kbase + kk];
    const float qq = sh_q[kbase + kk];
    st[kk].x += kk_ * dl.x; st[kk].y += kk_ * dl.y; st[kk].z += kk_ * dl.z; st[kk].w += kk_ * dl.w;
    op.x += st[kk].x * qq; op.y += st[kk].y * qq; op.z += st[kk].z * qq; op.w += st[kk].w * qq;
  }
  *reinterpret_cast<float4*>(&red_o[j][kg][4 * v4]) = op;
#pragma unroll
  for (int kk = 0; kk < KR; ++kk)                                       // write back fp32 state
    reinterpret_cast<float4*>(S_h + (kbase + kk) * DV)[v4] = st[kk];
  __syncthreads();                                                      // (E)

  // (6) Complete o for my 4 columns, round to bf16 (core_attn_out), gated RMSNorm over the 128 values.
  float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int g = 0; g < KG; ++g) {
    const float4 r = *reinterpret_cast<const float4*>(&red_o[j][g][4 * v4]);
    o.x += r.x; o.y += r.y; o.z += r.z; o.w += r.w;
  }
  const float ob[4] = {rnd(o.x), rnd(o.y), rnd(o.z), rnd(o.w)};
  float ss = ob[0] * ob[0] + ob[1] * ob[1] + ob[2] * ob[2] + ob[3] * ob[3];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);  // 32 lanes = 128 values
  const float r_inv = rsqrtf(ss / (float)DV + EPS);
  if (kg == 0) {                                                        // one warp of each v-head writes
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int d = 4 * v4 + c;
      const float xn = ob[c] * r_inv;
      const float y1 = rnd(bf2f(norm_w[d]) * rnd(xn));                 // weight * bf16(xn), bf16 product
      const float zc = bf2f(z[(size_t)s * VALUE_DIM + h * DV + d]);
      const float sz = zc / (1.f + expf(-zc));                         // SiLU(z) in fp32
      out[(size_t)s * VALUE_DIM + h * DV + d] = f2bf(y1 * sz);
    }
  }
}

}  // namespace dn

// ---------------- host launcher (PyTorch extension) ----------------
#ifdef DELTANET_TORCH_EXTENSION
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

torch::Tensor deltanet_decode(torch::Tensor qkv, torch::Tensor z, torch::Tensor b, torch::Tensor a,
                              torch::Tensor conv_state, torch::Tensor rec_state, torch::Tensor A_log,
                              torch::Tensor dt_bias, torch::Tensor conv_w, torch::Tensor norm_w) {
  using namespace dn;
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == torch::kBFloat16 && qkv.is_contiguous());
  TORCH_CHECK(qkv.dim() == 2 && qkv.size(1) == CONV_DIM, "qkv must be [T, 8192]");
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kBFloat16 && z.is_contiguous() && z.size(0) == qkv.size(0) && z.size(1) == VALUE_DIM);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16 && b.is_contiguous() && b.size(0) == qkv.size(0) && b.size(1) == HV);
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && a.is_contiguous() && a.size(0) == qkv.size(0) && a.size(1) == HV);
  TORCH_CHECK(conv_state.scalar_type() == torch::kBFloat16 && conv_state.is_contiguous() &&
              conv_state.size(0) == qkv.size(0) && conv_state.size(1) == CONV_DIM && conv_state.size(2) == CK);
  TORCH_CHECK(rec_state.scalar_type() == torch::kFloat32 && rec_state.is_contiguous() &&
              rec_state.size(0) == qkv.size(0) && rec_state.size(1) == HV && rec_state.size(2) == DK && rec_state.size(3) == DV);
  TORCH_CHECK(A_log.scalar_type() == torch::kFloat32 && A_log.numel() == HV);
  TORCH_CHECK(dt_bias.scalar_type() == torch::kFloat32 && dt_bias.numel() == HV);
  TORCH_CHECK(conv_w.scalar_type() == torch::kBFloat16 && conv_w.is_contiguous() && conv_w.numel() == CONV_DIM * CK);
  TORCH_CHECK(norm_w.scalar_type() == torch::kBFloat16 && norm_w.numel() == DV);
  const int T = qkv.size(0);
  auto out = torch::empty({T, VALUE_DIM}, qkv.options());
  auto stream = c10::cuda::getCurrentCUDAStream();
  deltanet_decode_kernel<<<dim3(HK, T), NTHREADS, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(conv_state.data_ptr()), rec_state.data_ptr<float>(),
      A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(conv_w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deltanet_decode", &deltanet_decode, "fused Gated DeltaNet decode step, one block per (k-head, sequence)");
}
#endif
