"""Reference for the fused 4-bit weight GEMV (format and math; see engine/kernels/w4_gemv.cu).

Format (ours). W is [N, K], K % 64 == 0, asymmetric 4-bit, group size 64 along K:
    w[n, k] = q[n, k] * scale[n, g] + minv[n, g],  g = k // 64,  q in 0..15
    packed[n] : K/2 bytes = K/32 chunks of 16 bytes. Chunk c covers k = 32c .. 32c+31,
                byte j (0..15) = q[32c + j] | (q[32c + 16 + j] << 4)
    meta[n]   : K/64 pairs (fp16 scale, fp16 minv)            -> tensor [N, K/64, 2]
Bytes per weight = 0.5 + 4/64 = 0.5625.

Activation prologue (optional): x_n = fp16( x * rsqrt(mean(x^2) + 1e-6) * (1 + norm_w) ).
Epilogue (optional): y = fp16( acc + resid ).
"""
import torch

GROUP = 64


def pack(W, rows_per_chunk=8192):
    """W [N, K] float (any float dtype, any device) -> (packed uint8 [N, K/2], meta fp16 [N, K/64, 2])."""
    N, K = W.shape
    assert K % GROUP == 0
    G = K // GROUP
    packs, metas = [], []
    for i in range(0, N, rows_per_chunk):
        Wg = W[i:i + rows_per_chunk].float().reshape(-1, G, GROUP)
        mn = Wg.amin(-1)
        mx = Wg.amax(-1)
        s16 = ((mx - mn) / 15.0).half()
        s16 = torch.where(s16 == 0, torch.ones_like(s16), s16)  # constant group: q = 0, value = minv
        m16 = mn.half()
        q = torch.round((Wg - m16.float()[..., None]) / s16.float()[..., None]).clamp_(0, 15).to(torch.uint8)
        q = q.reshape(-1, 2 * G, 32)  # chunks of 32 weights
        packs.append((q[..., :16] | (q[..., 16:] << 4)).reshape(-1, K // 2).contiguous())
        metas.append(torch.stack([s16, m16], -1).contiguous())
    return torch.cat(packs, 0), torch.cat(metas, 0)


def dequant(packed, meta, K):
    """Reference dequant -> float32 [N, K]. Exact inverse of the layout above."""
    N = packed.shape[0]
    G = K // GROUP
    p = packed.reshape(N, 2 * G, 16)
    q = torch.cat([p & 0xF, p >> 4], -1).reshape(N, G, GROUP).float()
    s = meta[..., 0].float()[..., None]
    m = meta[..., 1].float()[..., None]
    return (q * s + m).reshape(N, K)


def dequant_torch_fp16(packed, meta, K):
    """The 'dequant-then-matmul' baseline as plain torch ops, fp16 output (what a non-fused engine does)."""
    N = packed.shape[0]
    G = K // GROUP
    p = packed.reshape(N, 2 * G, 16)
    q = torch.cat([p & 0xF, p >> 4], -1).reshape(N, G, GROUP).half()
    return (q * meta[..., 0:1] + meta[..., 1:2]).reshape(N, K)


def rmsnorm_ref(x, norm_w, eps=1e-6):
    """Qwen-style (1 + w) RMSNorm, fp32 math, fp16 result."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + norm_w.float())).half()


def gemv_ref(x, packed, meta, K, norm_w=None, resid=None, norm=False, eps=1e-6, chunk=16384):
    """fp32 reference: y[T, N] = xn[T, K] @ dequant(W)^T (+ resid). Dequant is chunked over N."""
    xn = rmsnorm_ref(x, norm_w, eps).float() if norm else x.float()
    N = packed.shape[0]
    outs = []
    for i in range(0, N, chunk):
        outs.append(xn @ dequant(packed[i:i + chunk], meta[i:i + chunk], K).T)
    y = torch.cat(outs, 1)
    if resid is not None:
        y = y + resid.float()
    return y


def make_weights(N, K, device, seed=0, rows_per_chunk=8192):
    """Random packed weights, generated and packed in row chunks (bounded memory)."""
    g = torch.Generator(device=device).manual_seed(seed)
    packs, metas = [], []
    for i in range(0, N, rows_per_chunk):
        n = min(rows_per_chunk, N - i)
        W = (torch.randn(n, K, generator=g, device=device) * 0.02 + 0.003).half()
        p, m = pack(W)
        packs.append(p)
        metas.append(m)
        del W
    return torch.cat(packs, 0), torch.cat(metas, 0)
