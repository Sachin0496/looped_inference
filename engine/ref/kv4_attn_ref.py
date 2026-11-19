"""Reference for latent decode attention over the compressed cache (kernel: engine/kernels/kv4_attn.cu).

Cache per layer and sequence, S cached tokens, latent width R (256 or 512):
  cq   uint8 [B, S, R/2]    E2M1 codes, two per byte; element 2j is the low nibble of byte j.
  cs   uint8 [B, S, R/16]   E4M3 (float8_e4m3fn bit pattern) scale per 16 latent values.
  kr   uint8 [B, S, 4, 64]  E4M3 codes of the rotated, normalized RoPE key of each KV group.
  krs  fp16  [B, S, 4]      scale of kr:  kr_value = E4M3(code) * krs[t, g].
  ks   fp16  [B, S, 4]      per-(token, group) key normalization scale, used as given.
Dequantized latent: c[t, d] = E2M1(nibble) * E4M3(cs[t, d // 16]).

Query (fp32): qa [B, 16, R] absorbed nope query, qr [B, 16, 64] RoPE query. Head h uses group g = h // 4.
  score(h, t) = (1/16) * ( ks[t, g] * dot(qa[h], c[t]) + dot(qr[h], kr[t, g]) )      (softmax over all t)
  o_lat[h]    = sum_t softmax_t(score(h, t)) * c[t]                                  (fp32, [B, 16, R])
"""
import torch

E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_VALUE = torch.tensor(list(E2M1_MAG) + [-m for m in E2M1_MAG])  # index = 4-bit code
E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # round-to-nearest bin edges
HEADS, GROUPS, DR = 16, 4, 64
SM_SCALE = 1.0 / 16.0  # 1 / sqrt(256)
E4M3_MAX = 448.0


def quant_c(c, rows=1 << 16):
    """fp32 latent [..., R] -> (cq uint8 [..., R/2], cs uint8 [..., R/16]). Chunked over rows."""
    R = c.shape[-1]
    lead = c.shape[:-1]
    flat = c.reshape(-1, R).float()
    mid = E2M1_MIDPOINTS.to(c.device)
    cqs, css = [], []
    for i in range(0, flat.shape[0], rows):
        x = flat[i:i + rows].reshape(-1, R // 16, 16)
        amax = x.abs().amax(-1)
        s = (amax / 6.0).clamp(min=2.0 ** -9, max=E4M3_MAX)  # 6 = largest E2M1 magnitude
        s8 = s.to(torch.float8_e4m3fn)
        xs = x / s8.float()[..., None]
        idx = torch.bucketize(xs.abs().contiguous(), mid)
        nib = (idx + 8 * (xs < 0).long()).to(torch.uint8).reshape(-1, R)
        cqs.append(nib[:, 0::2] | (nib[:, 1::2] << 4))
        css.append(s8.view(torch.uint8))
    cq = torch.cat(cqs).reshape(*lead, R // 2).contiguous()
    cs = torch.cat(css).reshape(*lead, R // 16).contiguous()
    return cq, cs


def dequant_c(cq, cs):
    """(cq, cs) -> fp32 latent [..., R]. Exact inverse of the layout above."""
    R = cq.shape[-1] * 2
    nib = torch.stack([cq & 0xF, cq >> 4], -1).reshape(*cq.shape[:-1], R).long()
    val = E2M1_VALUE.to(cq.device)[nib]
    sc = cs.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return val * sc


def quant_kr(kr, rows=1 << 16):
    """fp32 rope key [B, S, 4, 64] -> (codes uint8 [B, S, 4, 64], krs fp16 [B, S, 4]). One scale per (t, g)."""
    lead = kr.shape[:-1]
    flat = kr.reshape(-1, DR).float()
    codes, scales = [], []
    for i in range(0, flat.shape[0], rows):
        x = flat[i:i + rows]
        amax = x.abs().amax(-1, keepdim=True)
        s = (amax / E4M3_MAX).clamp(min=1e-4).half()
        q = (x / s.float()).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        codes.append(q.view(torch.uint8))
        scales.append(s.reshape(-1))
    return torch.cat(codes).reshape(*lead, DR).contiguous(), torch.cat(scales).reshape(lead).contiguous()


def dequant_kr(codes, krs):
    """(codes, krs) -> fp32 rope key [B, S, 4, 64]."""
    return codes.view(torch.float8_e4m3fn).float() * krs.float()[..., None]


def attn_ref(qa, qr, c, kr, ks):
    """fp32 reference of the attention math. c [B, S, R] and kr [B, S, 4, 64] are the (dequantized or original)
    cache values; ks [B, S, 4] fp16/fp32. Returns o_lat [B, 16, R] fp32."""
    B, H, R = qa.shape
    S = c.shape[1]
    qa = qa.float()
    qr = qr.float()
    c = c.float()
    kr = kr.float()
    ksf = ks.float()
    nope = torch.einsum("bhr,bsr->bhs", qa, c).view(B, GROUPS, 4, S)
    nope = nope * ksf.permute(0, 2, 1)[:, :, None, :]
    rope = torch.einsum("bgid,bsgd->bgis", qr.view(B, GROUPS, 4, DR), kr).reshape(B, GROUPS, 4, S)
    sc = SM_SCALE * (nope + rope).reshape(B, H, S)
    p = torch.softmax(sc, dim=-1)
    return torch.einsum("bhs,bsr->bhr", p, c)
