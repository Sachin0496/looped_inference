"""CPU check: emulate the fused kernel's arithmetic (fp32 + bf16 rounding points) against the reference."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "engine" / "ref"))
import torch
import mhc_ref as ref

rnd = lambda x: x.to(torch.bfloat16).float()

def emulate(inp, merge, pre_scale):
    r = inp["residual"].float()                       # [T,S,D]
    if merge:
        mix, post, b = rnd(inp["mix_prev"]), rnd(inp["post_prev"]), inp["branch_prev"].float()
        r = rnd(rnd(torch.einsum("tij,tjd->tid", mix, r)) + rnd(post[..., None] * b[:, None, :]))
    T, S, D = r.shape
    flat = r.reshape(T, S * D)
    nv = flat * torch.rsqrt(flat.square().sum(-1, keepdim=True) / (S * D) + 1e-6)
    l = nv @ inp["weight"].T + inp["bias"]
    pre = rnd(torch.sigmoid(l[:, :S])); post = 2 * torch.sigmoid(l[:, S:2 * S])
    mix = ref.sinkhorn(l[:, 2 * S:].reshape(T, S, S), 20)
    br = rnd(torch.einsum("ts,tsd->td", pre, r))
    if pre_scale != 1.0: br = rnd(br * pre_scale)
    out = br * torch.rsqrt(br.square().mean(-1, keepdim=True) + 1e-6) * (1 + inp["norm_w"].to(torch.bfloat16).float())
    return r.to(torch.bfloat16), out.to(torch.bfloat16), post, mix

def ulp(a, b):
    o = lambda x: (lambda i: torch.where(i < 0, -32768 - i, i))(x.view(torch.int16).int())
    return int((o(a) - o(b)).abs().max())

worst = 0
for T in (1, 4, 16):
    for merge in (False, True):
        for ps in (1.0, 0.5):
            inp = ref.make_inputs(T, seed=T + merge)
            kw = dict(branch_prev=inp["branch_prev"], post_prev=inp["post_prev"], mix_prev=inp["mix_prev"]) if merge else {}
            rr, ro, rp, rm = ref.mhc_step(inp["residual"], inp["weight"], inp["bias"], inp["norm_w"].to(torch.bfloat16), pre_scale=ps, **kw)
            er, eo, ep, em = emulate(inp, merge, ps)
            u = (ulp(er, rr), ulp(eo, ro), float((ep - rp).abs().max()), float((em - rm).abs().max()))
            worst = max(worst, u[0], u[1])
            print(f"T={T:3d} merge={merge!s:5} pre_scale={ps}: residual {u[0]} ulp, out {u[1]} ulp, post {u[2]:.1e}, mix {u[3]:.1e}")
print("worst ulp:", worst)
assert worst <= 2
