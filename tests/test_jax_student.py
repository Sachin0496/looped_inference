"""Checks tpu/jax_student.py (config-E student K/V path) against the PyTorch reference in kaggle/distill/distill.py.

The torch reference below is copied from distill.py (hadamard, ste, fp4_fake, fp8_fake, quantize, and the k/v part of
StudentAttn.run), with the globals set to the real dims and CFG = config E. Run as a script or under pytest:
    python tests/test_jax_student.py
"""
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tpu import jax_student as js  # noqa: E402

# ---------------------------------------------------------------- torch reference (copied from distill.py)
ROT, HALF, G, HD, H = 64, 32, 4, 256, 4096
NOPE = HD - ROT
CFG = dict(name="E", r=512, res=128, share_loop=True, pair=False, rope="shared", qat=True, hadamard=True)
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
HAD = {}


def hadamard(n, device):
    key = (n, str(device))
    if key not in HAD:
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        HAD[key] = (h / np.sqrt(n)).to(device)  # symmetric and orthogonal: H = H^T = H^-1
    return HAD[key]


def ste(x, q):
    return x + (q - x).detach()


@torch.no_grad()
def fp4_fake(x):
    """E2M1 values, one E4M3 scale per 16 along the last dim (NVFP4-style, no global scale)."""
    shp = x.shape
    g = x.reshape(-1, 16).float()
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float().clamp_min(2 ** -9)
    y = (g / scale).clamp(-6, 6)
    grid = E2M1.to(y.device)
    idx = (y.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    return (grid[idx] * torch.sign(y) * scale).reshape(shp)


@torch.no_grad()
def fp8_fake(x):
    """E4M3 with one scale per row (per token and group)."""
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 448.0).clamp_min(2 ** -9)
    return ((x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()) * scale


def quantize(c, kr):
    """c: [B,T,r] latent, kr: [B,T,RK] rope keys (pre-norm). Optional Hadamard + FP4 / FP8 fake quant (STE)."""
    if CFG["hadamard"]:
        Hm = hadamard(c.shape[-1], c.device)
        c = c @ Hm
    if CFG["qat"]:
        c = ste(c, fp4_fake(c))
        kr = ste(kr, fp8_fake(kr.reshape(*kr.shape[:-1], -1, ROT)).reshape(kr.shape))
    if CFG["hadamard"]:
        c = c @ Hm
    return c, kr


def torch_rmsnorm(x, w, eps=1e-6):
    """Qwen3_5RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)


def torch_student_kv(x, p, k_norm_w, base_c=None):
    """The k/v part of StudentAttn.run (rope shared mode), ROPE_FN and attention omitted.
    p: dict of torch tensors (enc_dkv, enc_kr, up_k, up_v, alpha). Returns k [B,T,G,HD], v [B,T,G,HD], c [B,T,r]."""
    B, T, _ = x.shape
    c = F.linear(x, p["enc_dkv"]).float()
    kr = F.linear(x, p["enc_kr"]).float()
    c, kr = quantize(c, kr)
    if base_c is not None:  # residual visit: up-project from [c0 ; c_res]
        c_in = torch.cat([base_c, c], -1)
    else:
        c_in = c
    cin = c_in.shape[-1]
    k_nope = F.linear(c_in, p["up_k"].reshape(G * NOPE, cin)).float().view(B, T, G, NOPE)
    v_ = F.linear(c_in, p["up_v"].reshape(G * HD, cin)).float().view(B, T, G, HD)
    al = p["alpha"]
    a, b = al[..., 0], al[..., 1]
    kr4 = kr.reshape(B, T, 1, ROT)
    xr, xi = kr4[..., :HALF], kr4[..., HALF:]
    rope = torch.cat([a * xr - b * xi, b * xr + a * xi], -1)  # [B,T,G,ROT]
    k = torch_rmsnorm(torch.cat([rope, k_nope], -1), k_norm_w)
    return k, v_, c


# ---------------------------------------------------------------- helpers
def load_torch_slot(sd, L, v):
    return {n: sd[f"{L}.{n}.v{v}"].float() for n in ("enc_dkv", "enc_kr", "up_k", "up_v", "alpha")}


def to_jax(p):
    return {k: jnp.asarray(v.detach().cpu().numpy(), dtype=jnp.float32) for k, v in p.items()}


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def main():
    t0 = time.time()
    ckpt = ROOT / "results" / "distill_E_v1" / "student_E.pt"
    sd = torch.load(ckpt, map_location="cpu")
    B, T = 2, 24
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((B, T, H)).astype(np.float32)
    kn_w_np = (0.1 * rng.standard_normal(HD)).astype(np.float32)
    x_t, kn_w_t = torch.from_numpy(x_np), torch.from_numpy(kn_w_np)
    x_j, kn_w_j = jnp.asarray(x_np), jnp.asarray(kn_w_np)

    # ------------------------------------------------ 1. forward agreement
    print("== 1. forward: k, v, c vs torch reference (float32, B=2, T=24)")
    results = {}
    for L in (7, 3):
        visits = [0, 1] if L == 7 else [0]
        base_t = base_j = None
        for v in visits:
            pt = load_torch_slot(sd, L, v)
            kt, vt, ct = torch_student_kv(x_t, pt, kn_w_t, base_c=base_t)
            pj = to_jax(pt)
            kj, vj, cj = js.student_kv(x_j, pj, kn_w_j, base_c=base_j, eps=1e-6, hadamard=True, qat=True)
            kt, vt, ct = kt.numpy(), vt.numpy(), ct.numpy()
            kj, vj, cj = np.asarray(kj), np.asarray(vj), np.asarray(cj)
            n_c_diff = int(np.sum(np.abs(cj - ct) > 1e-5 * np.max(np.abs(ct))))
            r = dict(k=rel(kj, kt), v=rel(vj, vt), c=rel(cj, ct),
                     k_abs=float(np.max(np.abs(kj - kt))), v_abs=float(np.max(np.abs(vj - vt))),
                     c_abs=float(np.max(np.abs(cj - ct))), c_elems_differ=n_c_diff, c_size=ct.size)
            results[f"layer {L} visit {v}"] = r
            print(f"  layer {L} visit {v}: k rel {r['k']:.3e} (abs {r['k_abs']:.2e})  v rel {r['v']:.3e} "
                  f"(abs {r['v_abs']:.2e})  c rel {r['c']:.3e} (abs {r['c_abs']:.2e})  "
                  f"c elems differing {n_c_diff}/{r['c_size']}")
            assert r["k"] < 1e-4 and r["v"] < 1e-4 and r["c"] < 1e-4, f"forward mismatch at {L}.{v}: {r}"
            # visit 1 of layer 7 up-projects [c0 ; c_res] with c0 = this visit's stored c
            base_t, base_j = torch.from_numpy(ct), jnp.asarray(cj)

    # ------------------------------------------------ 2. gradients (STE) w.r.t. the five parameter tensors
    print("== 2. gradients of loss = sum(k*R1) + sum(v*R2) w.r.t. enc_dkv, enc_kr, up_k, up_v, alpha")
    grad_errs = {}
    for L in (7, 3):
        visits = [0, 1] if L == 7 else [0]
        R = {v: (rng.standard_normal((B, T, G, HD)).astype(np.float32),
                 rng.standard_normal((B, T, G, HD)).astype(np.float32)) for v in visits}
        # torch
        pts = {v: {k: t.clone().requires_grad_(True) for k, t in load_torch_slot(sd, L, v).items()} for v in visits}
        base = None
        loss_t = 0.0
        for v in visits:
            kt, vt, ct = torch_student_kv(x_t, pts[v], kn_w_t, base_c=base)
            loss_t = loss_t + (kt * torch.from_numpy(R[v][0])).sum() + (vt * torch.from_numpy(R[v][1])).sum()
            base = ct
        gt = torch.autograd.grad(loss_t, [pts[v][n] for v in visits for n in pts[v]])
        gt = dict(zip([(v, n) for v in visits for n in pts[v]], [g.numpy() for g in gt]))
        # jax
        pjs = {v: to_jax(load_torch_slot(sd, L, v)) for v in visits}

        def loss_j(params):
            base_c, total = None, 0.0
            for v in visits:
                kj, vj, cj = js.student_kv(x_j, params[v], kn_w_j, base_c=base_c)
                total = total + jnp.sum(kj * R[v][0]) + jnp.sum(vj * R[v][1])
                base_c = cj
            return total

        gj_tree = jax.grad(loss_j)(pjs)
        for v in visits:
            for n in pjs[v]:
                a, b = np.asarray(gj_tree[v][n]), gt[(v, n)]
                e = rel(a, b)
                grad_errs[f"layer {L} visit {v} {n}"] = e
                print(f"  layer {L} visit {v} {n:8s} shape {tuple(b.shape)}  max rel err {e:.3e}")
    worst_grad = max(grad_errs.values())
    print(f"  worst gradient max-rel error: {worst_grad:.3e}")
    assert worst_grad < 1e-3, f"gradient mismatch: {worst_grad}"

    # ------------------------------------------------ 3. jit
    print("== 3. jax.jit")
    pj = to_jax(load_torch_slot(sd, 7, 0))
    f_jit = jax.jit(js.student_kv, static_argnames=("eps", "hadamard", "qat"))
    kj, vj, cj = f_jit(x_j, pj, kn_w_j, None, eps=1e-6, hadamard=True, qat=True)
    kj2, vj2, cj2 = js.student_kv(x_j, pj, kn_w_j, None)
    jit_diff = max(float(jnp.max(jnp.abs(kj - kj2))), float(jnp.max(jnp.abs(vj - vj2))),
                   float(jnp.max(jnp.abs(cj - cj2))))
    print(f"  jit vs eager max abs diff {jit_diff:.3e} (k shape {kj.shape}, v shape {vj.shape}, c shape {cj.shape})")
    assert jit_diff < 1e-5
    jit_grad = jax.jit(jax.grad(lambda p: jnp.sum(js.student_kv(x_j, p, kn_w_j, None)[0] ** 2)))(pj)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jit_grad.values())
    print("  jit(grad) finite")

    # ------------------------------------------------ 4. checkpoint loading
    print("== 4. load_student_pt / student_tree")
    by_slot = js.load_student_pt(str(ckpt))
    print(f"  slots loaded: {len(by_slot)}")
    tree = js.student_tree(by_slot)
    for part, val in tree.items():
        if isinstance(val, dict):
            for n, arr in val.items():
                print(f"  {part}.{n}: {tuple(arr.shape)} {arr.dtype}")
        else:
            print(f"  {part}: keys {list(val.keys())}")
    for part in ("loop0", "loop1"):
        for n, arr in tree[part].items():
            assert arr.shape[0] == 6, (part, n, arr.shape)
    assert tree["loop0"]["enc_dkv"].shape == (6, 512, H)
    assert tree["loop1"]["enc_dkv"].shape == (6, 128, H)
    assert tree["loop1"]["up_k"].shape == (6, G, NOPE, 640)
    assert tree["pre"]["up_k"].shape == (G, NOPE, 512)
    print(f"  kv bytes per token (config E): {js.kv_bytes_per_token_E()}")
    assert js.kv_bytes_per_token_E() == 3660

    print(f"ALL CHECKS PASSED in {time.time() - t0:.1f}s")


def test_jax_student():
    main()


if __name__ == "__main__":
    main()
