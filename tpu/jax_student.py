"""JAX port of the config-E student attention K/V path (kaggle/distill/distill.py, StudentAttn.run before ROPE_FN).

Config E: visit-0 latent c = Hadamard -> FP4 (E2M1, E4M3 scale per 16) -> Hadamard^-1, with a straight-through
estimator; rope key kr = FP8 E4M3 with one scale per row of ROT dims (STE); visit-1 of looped layers up-projects
concat([c0, c_res]). Rope "shared" mode: one kr per visit, complex per-(group, frequency) scale alpha.

Shapes: x [B,T,H]; sp leaves enc_dkv [r,H], enc_kr [ROT,H], up_k [G,NOPE,cin], up_v [G,HD,cin], alpha [G,HALF,2].
"""
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

ROT = 64           # rope dims per head (64 of 256)
HALF = ROT // 2
G = 4              # KV groups
HD = 256           # head dim
NOPE = HD - ROT    # 192
F32 = jnp.float32
HI = lax.Precision.HIGHEST  # full-precision f32 matmuls (TPU default would use bf16 passes)

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


@functools.lru_cache(maxsize=None)
def _hadamard_np(n):
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.concatenate([np.concatenate([h, h], 1), np.concatenate([h, -h], 1)], 0)
    return (h / math.sqrt(n)).astype(np.float32)


def hadamard_matrix(n):
    """Normalised Sylvester Hadamard matrix (symmetric, orthogonal): H = H^T = H^-1. n must be a power of 2."""
    return jnp.asarray(_hadamard_np(int(n)))


def _ste(x, q):
    """Forward value q, gradient of identity w.r.t. x (same arithmetic as torch's ste: x + (q - x).detach())."""
    return x + lax.stop_gradient(q - x)


def fp4_fake(x):
    """E2M1 values, one E4M3 scale per 16 along the last dim (no global scale). No gradient."""
    x = lax.stop_gradient(x).astype(F32)
    shp = x.shape
    g = x.reshape(-1, 16)
    amax = jnp.maximum(jnp.max(jnp.abs(g), axis=-1, keepdims=True), 1e-12)
    scale = (amax / 6.0).astype(jnp.float8_e4m3fn).astype(F32)
    scale = jnp.maximum(scale, 2.0 ** -9)
    y = jnp.clip(g / scale, -6.0, 6.0)
    grid = jnp.asarray(E2M1)
    # argmin returns the first minimum, i.e. ties go to the lower grid value (as torch.argmin does)
    idx = jnp.argmin(jnp.abs(jnp.abs(y)[..., None] - grid), axis=-1)
    return (grid[idx] * jnp.sign(y) * scale).reshape(shp)


def fp8_fake(x):
    """E4M3 with one scale per row (the last dim, i.e. per token and group). No gradient."""
    x = lax.stop_gradient(x).astype(F32)
    amax = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True), 1e-12)
    scale = jnp.maximum(amax / 448.0, 2.0 ** -9)
    return jnp.clip(x / scale, -448.0, 448.0).astype(jnp.float8_e4m3fn).astype(F32) * scale


def _quantize(c, kr, hadamard, qat):
    """c: [B,T,r] f32 latent, kr: [B,T,ROT] f32 rope key. Mirrors distill.py quantize()."""
    if hadamard:
        Hm = hadamard_matrix(c.shape[-1])
        c = jnp.matmul(c, Hm, precision=HI)
    if qat:
        c = _ste(c, fp4_fake(c))
        kr = _ste(kr, fp8_fake(kr))
    if hadamard:
        c = jnp.matmul(c, Hm, precision=HI)
    return c, kr


def student_kv(x, sp, k_norm_w, base_c=None, *, eps=1e-6, hadamard=True, qat=True):
    """Config-E student K and V for one (layer, visit).

    Returns (k [B,T,G,HD], v [B,T,G,HD], c [B,T,r] f32). k is post-RMSNorm (1+w), pre-RoPE, cast to x.dtype.
    c is the quantized latent of this visit (what visit 0 stores for visit 1 as base_c).
    """
    dt = x.dtype
    c = jnp.einsum("bth,rh->btr", x, sp["enc_dkv"].astype(dt), preferred_element_type=F32, precision=HI)
    kr = jnp.einsum("bth,rh->btr", x, sp["enc_kr"].astype(dt), preferred_element_type=F32, precision=HI)
    c, kr = _quantize(c, kr, hadamard, qat)
    c_in = c if base_c is None else jnp.concatenate([base_c.astype(F32), c], axis=-1)

    k_nope = jnp.einsum("btc,gnc->btgn", c_in, sp["up_k"].astype(F32), precision=HI)
    v = jnp.einsum("btc,gdc->btgd", c_in, sp["up_v"].astype(F32), precision=HI)

    a = sp["alpha"][..., 0].astype(F32)  # [G,HALF]
    b = sp["alpha"][..., 1].astype(F32)
    half = sp["alpha"].shape[-2]          # HALF (32 at real dims; derived so tiny test models work too)
    xr = kr[:, :, None, :half]           # [B,T,1,HALF]
    xi = kr[:, :, None, half:]
    rope = jnp.concatenate([a * xr - b * xi, b * xr + a * xi], axis=-1)  # [B,T,G,ROT]

    kcat = jnp.concatenate([rope, k_nope], axis=-1)  # rope dims 0..ROT-1, nope dims ROT..HD-1
    ms = jnp.mean(kcat * kcat, axis=-1, keepdims=True)
    k = kcat * lax.rsqrt(ms + eps) * (1.0 + k_norm_w.astype(F32))
    return k.astype(dt), v.astype(dt), c


def load_student_pt(path):
    """Read a distill.py student_<cfg>.pt state dict -> {(L, v): sp dict of np.float32 arrays}."""
    import torch  # only needed for reading the checkpoint

    sd = torch.load(path, map_location="cpu")
    out = {}
    for key, val in sd.items():
        L_s, name, vis = key.split(".")  # e.g. "7.enc_dkv.v1"
        k = (int(L_s), int(vis[1:]))
        out.setdefault(k, {})[name] = val.detach().cpu().float().numpy()
    need = ("enc_dkv", "enc_kr", "up_k", "up_v", "alpha")
    for k, d in out.items():
        missing = [n for n in need if n not in d]
        assert not missing, f"slot {k} missing {missing}"
    return out


def student_tree(sd_by_slot, loop_layers=(7, 11, 15, 19, 23, 27), pre=3, post=31):
    """Stack the per-layer params for lax.scan-style use: pre/post single layers, loop0/loop1 stacked over loop_layers."""
    def sp_of(L, v):
        return {k: jnp.asarray(val, dtype=jnp.float32) for k, val in sd_by_slot[(L, v)].items()}

    def stack(v):
        per = [sp_of(L, v) for L in loop_layers]
        return {k: jnp.stack([p[k] for p in per], axis=0) for k in per[0]}

    return {"pre": sp_of(pre, 0), "loop0": stack(0), "loop1": stack(1), "post": sp_of(post, 0)}


def kv_bytes_per_token_E():
    """distill.py kv_bytes_per_token for config E at real dims (r=512, r_res=128, rope shared, FP4 + FP8 QAT).

    Owner slots: 8 visit-0 latents (r=512) and 6 visit-1 residual latents (r_res=128). Each slot stores a latent
    (FP4 values r/2 bytes + E4M3 scale per 16 r/16 bytes) and its rope key (64 FP8 bytes + one 2-byte scale).
    """
    rk_real, real_rot = 64, 64
    slots = [512] * 8 + [128] * 6
    tot = 0.0
    for r in slots:
        tot += r * 0.5 + r / 16 + (rk_real * 1 + (rk_real // real_rot) * 2)
    return int(tot)
