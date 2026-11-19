"""CPU check of the fused Gated DeltaNet kernel's index mapping and rounding points.

No CUDA device is available locally, so this emulates engine/kernels/deltanet_fused.cu thread by thread
(vectorised over the 512 threads of a block): same channel ownership, same state tile per thread,
same shared-memory reductions, same bf16 rounding points. It is compared with engine/ref/deltanet_ref.py
over 64 carried decode steps. The real kernel is checked on the T4 by kaggle/deltanet_bench.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "engine" / "ref"))
import torch
import torch.nn.functional as F
import deltanet_ref as R

rnd = lambda x: x.to(torch.bfloat16).float()
EPS = 1e-6
SQRT_DK = 11.313708498984761


def emulate_block(kh, S, conv, qkv_s, z_s, b_s, a_s, A_log, dt_bias, conv_w, norm_w, out_s):
    """One block (k-head kh, one sequence). S: [32,128,128] fp32, conv: [8192,4] bf16 (both updated in place)."""
    t = torch.arange(512)
    ch = torch.where(t < 128, kh * 128 + t, torch.where(t < 256, 2048 + kh * 128 + t - 128, 4096 + kh * 256 + t - 256))
    cs = conv[ch].float()                                              # [512,4]
    x = qkv_s[ch].float()
    w = conv_w[ch].float()
    o1, o2, o3 = cs[:, 1], cs[:, 2], cs[:, 3]
    conv_out = rnd(w[:, 0] * o1 + w[:, 1] * o2 + w[:, 2] * o3 + w[:, 3] * x)
    act = rnd(conv_out / (1 + torch.exp(-conv_out)))
    conv[ch] = torch.stack([o1, o2, o3, x], -1).to(torch.bfloat16)
    sh_q, sh_k, sh_v = act[:128].clone(), act[128:256].clone(), act[256:].reshape(2, 128).clone()

    j_of_head = torch.tensor([0, 1])
    h_of = 2 * kh + j_of_head
    beta = rnd(torch.sigmoid(b_s[h_of].float()))
    xs = a_s[h_of].float() + dt_bias[h_of]
    sp = torch.where(xs > 20, xs, torch.log1p(torch.exp(xs)))
    g = -torch.exp(A_log[h_of]) * sp

    def l2(v):  # warp-style sum of squares, then rsqrt(sum + eps)
        return v * torch.rsqrt((v * v).sum() + EPS)
    sh_q = l2(sh_q) / SQRT_DK
    sh_k = l2(sh_k)

    lt = t & 255; j = t >> 8; kg = lt >> 5; v4 = lt & 31; kbase = kg * 16
    h_t = 2 * kh + j
    rows = kbase[:, None] + torch.arange(16)[None, :]                  # [512,16]
    cols = (4 * v4)[:, None] + torch.arange(4)[None, :]                # [512,4]
    st = S[h_t[:, None, None], rows[:, :, None], cols[:, None, :]].clone()   # [512,16,4] state tile in "registers"
    decay = torch.exp(g[j])
    st = st * decay[:, None, None]
    kv_part = (st * sh_k[rows][:, :, None]).sum(1)                     # [512,4]
    red_kv = torch.zeros(2, 8, 128)
    red_kv[j[:, None], kg[:, None], cols] = kv_part
    kv = red_kv[j[:, None], :, cols].sum(-1)                           # [512,4] sum over k-groups
    vv = sh_v[j[:, None], cols]
    dl = (vv - kv) * beta[j][:, None]
    st = st + sh_k[rows][:, :, None] * dl[:, None, :]
    op_part = (st * sh_q[rows][:, :, None]).sum(1)
    red_o = torch.zeros(2, 8, 128)
    red_o[j[:, None], kg[:, None], cols] = op_part
    o = red_o[j[:, None], :, cols].sum(-1)                             # [512,4]
    S[h_t[:, None, None], rows[:, :, None], cols[:, None, :]] = st     # fp32 write-back
    ob = rnd(o)
    # per-head gated RMSNorm: rows of the same head cover all 128 columns exactly once
    for jj in range(2):
        sel = (j == jj)
        head_vals = torch.zeros(128)
        head_vals[cols[sel].reshape(-1)] = ob[sel].reshape(-1)
        var = (head_vals * head_vals).sum() / 128
        xn = rnd(head_vals * torch.rsqrt(var + EPS))
        h = 2 * kh + jj
        d = torch.arange(128)
        y1 = rnd(norm_w[d].float() * xn)
        zc = z_s[h * 128 + d].float()
        out_s[h * 128 + d] = (y1 * (zc / (1 + torch.exp(-zc)))).to(torch.bfloat16)


def emulate_step(qkv, z, b, a, conv_state, rec_state, A_log, dt_bias, conv_w, norm_w):
    T = qkv.shape[0]
    out = torch.empty(T, 4096, dtype=torch.bfloat16)
    for s in range(T):
        for kh in range(16):
            emulate_block(kh, rec_state[s], conv_state[s], qkv[s], z[s], b[s], a[s],
                          A_log, dt_bias, conv_w, norm_w, out[s])
    return out


def ulp(a, b):
    o = lambda x: (lambda i: torch.where(i < 0, -32768 - i, i))(x.contiguous().view(torch.int16).int())
    return (o(a) - o(b)).abs()


if __name__ == "__main__":
    worst_ulp, worst_state, worst_abs, min_le1 = 0, 0.0, 0.0, 1.0
    for T in (1, 2):
        gen = torch.Generator().manual_seed(100 + T)
        w = R.make_weights(0)
        conv_r, rec_r = R.make_state(T, gen)
        conv_e, rec_e = conv_r.clone(), rec_r.clone()
        for step in range(64):
            inp = R.make_step_inputs(T, gen)
            o_ref = R.deltanet_decode_step(inp["qkv"], inp["z"], inp["b"], inp["a"], conv_r, rec_r,
                                           w["A_log"], w["dt_bias"], w["conv_w"], w["norm_w"])
            o_emu = emulate_step(inp["qkv"], inp["z"], inp["b"], inp["a"], conv_e, rec_e,
                                 w["A_log"], w["dt_bias"], w["conv_w"], w["norm_w"])
            u = ulp(o_emu, o_ref)
            worst_ulp = max(worst_ulp, int(u.max()))
            min_le1 = min(min_le1, float((u <= 1).float().mean()))
            worst_abs = max(worst_abs, float((o_emu.float() - o_ref.float()).abs().max() / o_ref.float().abs().max()))
        assert torch.equal(conv_e, conv_r), "conv state mismatch"
        err = float((rec_e - rec_r).abs().max()) / float(rec_r.abs().max())
        worst_state = max(worst_state, err)
        print(f"T={T} steps=64 out max_ulp={worst_ulp} state max_abs_err/max|S|={err:.2e} conv_state exact")
    print(f"out: worst max_abs/max|out| = {worst_abs:.2e} (1 bf16 ulp at max scale = {2**-7:.2e}), min frac ulp<=1 = {min_le1:.5f}")
    assert worst_abs <= 2**-7 and min_le1 >= 0.99 and worst_state < 1e-4, "emulation disagrees with reference"
    print("emulation OK")
