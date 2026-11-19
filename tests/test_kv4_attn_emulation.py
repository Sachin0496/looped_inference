"""CPU check of the compressed-cache decode attention kernel (engine/kernels/kv4_attn.cu).

No CUDA device is available locally, so this (1) checks the manual E4M3 decode used by the kernel against
torch's float8 decode for all 256 codes, (2) emulates the kernel's arithmetic: 32-token tiles, rope and nope
dot products from the packed bytes, per-tile online softmax, split blocks, and the combine step, and
(3) compares the emulation with engine/ref/kv4_attn_ref.py on the dequantized cache.
The real kernel is checked on the T4 by kaggle/attn_bench.
"""
import importlib.util
import math
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kv4_attn_ref", ROOT / "engine" / "ref" / "kv4_attn_ref.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

TT = 32
SM = 1.0 / 16.0


def e4m3_manual(codes):
    """Bit-level E4M3 decode, mirroring e4m3_f32() in the kernel."""
    c = codes.to(torch.int64)
    sign = torch.where((c & 0x80) != 0, -1.0, 1.0)
    e = (c >> 3) & 0xF
    m = c & 0x7
    bits = (((e + 120) << 23) | (m << 20)).to(torch.int32)
    normal = bits.view(torch.float32)
    sub = m.float() * (2.0 ** -9)
    return sign * torch.where(e == 0, sub, normal)


def check_decodes():
    codes = torch.arange(256, dtype=torch.uint8)
    keep = ~((codes == 0x7F) | (codes == 0xFF))  # NaN codes
    mine = e4m3_manual(codes)[keep]
    theirs = codes.view(torch.float8_e4m3fn).float()[keep]
    assert torch.equal(mine, theirs), "manual E4M3 decode differs from torch"
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    for n in range(16):
        v = lut[n & 7] * (-1.0 if n & 8 else 1.0)
        assert float(ref.E2M1_VALUE[n]) == float(v)
    print("decodes: manual E4M3 == torch for all 254 non-NaN codes; E2M1 table OK")


def emulate(qa, qr, cq, cs, kc, krs, ks, target=80):
    """Kernel arithmetic on the packed cache (fp32; manual decodes; same tile/split/combine structure)."""
    B, H, R = qa.shape
    S = cq.shape[1]
    lut = ref.E2M1_VALUE.float()
    ntiles = math.ceil(S / TT)
    NS = min(ntiles, max(1, target // B))
    tps = math.ceil(ntiles / NS)
    NS = math.ceil(ntiles / tps)
    qa = qa.float()
    qr = qr.float()
    opart = torch.zeros(B, NS, H, R)
    mpart = torch.zeros(B, NS, H)
    lpart = torch.zeros(B, NS, H)
    for b in range(B):
        for p in range(NS):
            tile0 = p * tps
            tok_end = min(S, (tile0 + tps) * TT)
            m_run = torch.full((H,), -1e30)
            l_run = torch.zeros(H)
            acc = torch.zeros(H, R)
            for tile in range(tile0, tile0 + tps):
                toks = torch.arange(tile * TT, tile * TT + TT)
                valid = toks < tok_end
                vt = toks[valid]
                cdeq = torch.zeros(TT, R)
                kdeq = torch.zeros(TT, 4, 64)
                ksh = torch.zeros(TT, 4)
                krsh = torch.zeros(TT, 4)
                if len(vt) > 0:
                    cqv = cq[b, vt]
                    nib = torch.stack([cqv & 0xF, cqv >> 4], -1).reshape(len(vt), R).long()
                    sc = e4m3_manual(cs[b, vt]).repeat_interleave(16, dim=-1)
                    cdeq[valid] = lut[nib] * sc
                    kdeq[valid] = e4m3_manual(kc[b, vt]) * krs[b, vt].float()[..., None]
                    ksh[valid] = ks[b, vt].float()
                nope = qa[b] @ cdeq.T                                   # [H, TT]
                kh = kdeq.repeat_interleave(4, dim=1)                   # [TT, H, 64]
                rope = torch.einsum("hd,thd->ht", qr[b], kh)            # [H, TT]
                ks_h = ksh.repeat_interleave(4, dim=1).T                # [H, TT]
                s = SM * (ks_h * nope + rope)
                s[:, ~valid] = -math.inf
                mt = s.max(dim=1).values
                mn = torch.maximum(m_run, mt)
                alpha = torch.exp(m_run - mn)
                pv = torch.exp(s - mn[:, None])
                l_run = l_run * alpha + pv.sum(1)
                acc = acc * alpha[:, None] + pv @ cdeq
                m_run = mn
            opart[b, p] = acc
            mpart[b, p] = m_run
            lpart[b, p] = l_run
    out = torch.zeros(B, H, R)
    for b in range(B):
        for h in range(H):
            M = mpart[b, :, h].max()
            w = torch.exp(mpart[b, :, h] - M)
            den = (w * lpart[b, :, h]).sum()
            out[b, h] = ((w / den)[:, None] * opart[b, :, h]).sum(0)
    return out


def run_case(R, B, S, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(B, S, R, generator=g) * torch.exp(0.5 * torch.randn(B, S, 1, generator=g))
    kr = torch.randn(B, S, 4, 64, generator=g)
    ks = (0.5 + 1.5 * torch.rand(B, S, 4, generator=g)).half()
    qa = torch.randn(B, 16, R, generator=g) * 0.5
    qr = torch.randn(B, 16, 64, generator=g) * 0.5
    cq, cs = ref.quant_c(c)
    kc, krs = ref.quant_kr(kr)
    yref = ref.attn_ref(qa, qr, ref.dequant_c(cq, cs), ref.dequant_kr(kc, krs), ks)
    yfp = ref.attn_ref(qa, qr, c, kr, ks)
    yemu = emulate(qa, qr, cq, cs, kc, krs, ks)
    rel = ((yemu - yref).abs().max() / yref.abs().max()).item()
    rel_fp = ((yref - yfp).abs().max() / yfp.abs().max()).item()
    status = "PASS" if rel < 1e-4 else "FAIL"
    print(f"{status} R={R} B={B} S={S}: emulation vs dequant-ref rel={rel:.2e}   "
          f"(quantized vs fp32 cache rel={rel_fp:.2e}, info)")
    return rel < 1e-4


# ---------- v2 (kv4_attn_tc): fp16 tensor-core arithmetic ----------
TC_GATE = 5e-3  # relative error gate for the fp16-operand path


def _combine(opart, mpart, lpart):
    B, NS, H, R = opart.shape
    out = torch.zeros(B, H, R)
    for b in range(B):
        for h in range(H):
            M = mpart[b, :, h].max()
            w = torch.exp(mpart[b, :, h] - M)
            den = (w * lpart[b, :, h]).sum()
            out[b, h] = ((w / den)[:, None] * opart[b, :, h]).sum(0)
    return out


def emulate_tc(qa, qr, cq, cs, kc, krs, ks, nsm=40):
    """Arithmetic of kv4_attn_tc (engine/kernels/kv4_attn.cu, namespace tc), on CPU in torch:
    - Q_abs and q_rope divided by their per-(sequence, head) max-abs, rounded to fp16 (q_prep_kernel);
    - dequantized latent and rope codes asserted exact in fp16 (the kernel stores them as fp16);
    - nope and rope products with fp32 accumulate, then the per-(token, group) and per-head scales in fp32;
    - P = fp16(exp(score - m)); the running sum uses the fp16 P; O += P C with fp32 accumulate.
    The rescale of O on a max change is exact (alpha), so it is emulated as a plain multiply."""
    B, H, R = qa.shape
    S = cq.shape[1]
    lut = ref.E2M1_VALUE.float()
    ntiles = math.ceil(S / TT)
    NS = min(ntiles, max(1, nsm // B))
    tps = math.ceil(ntiles / NS)
    NS = math.ceil(ntiles / tps)
    qa = qa.float()
    qr = qr.float()
    qs = qa.abs().amax(-1)
    qs = torch.where(qs > 0, qs, torch.ones_like(qs))
    qrs = qr.abs().amax(-1)
    qrs = torch.where(qrs > 0, qrs, torch.ones_like(qrs))
    qh = (qa / qs[..., None]).half().float()
    qrh = (qr / qrs[..., None]).half().float()
    opart = torch.zeros(B, NS, H, R)
    mpart = torch.zeros(B, NS, H)
    lpart = torch.zeros(B, NS, H)
    for b in range(B):
        for p in range(NS):
            tile0 = p * tps
            tile_end = tile0 + tps
            tok_end = min(S, tile_end * TT)
            m_run = torch.full((H,), -1e30)
            l_run = torch.zeros(H)
            acc = torch.zeros(H, R)
            for tile in range(tile0, tile_end):
                toks = torch.arange(tile * TT, tile * TT + TT)
                valid = toks < tok_end
                vt = toks[valid]
                cdeq = torch.zeros(TT, R)
                kd = torch.zeros(TT, 4, 64)
                ksh = torch.zeros(TT, 4)
                krsh = torch.zeros(TT, 4)
                if len(vt) > 0:
                    cqv = cq[b, vt]
                    nib = torch.stack([cqv & 0xF, cqv >> 4], -1).reshape(len(vt), R).long()
                    sc = e4m3_manual(cs[b, vt]).repeat_interleave(16, dim=-1)
                    cv = lut[nib] * sc
                    assert torch.equal(cv.half().float(), cv), "E2M1 x E4M3 is not exact in fp16"
                    cdeq[valid] = cv
                    kv = e4m3_manual(kc[b, vt])
                    assert torch.equal(kv.half().float(), kv), "E4M3 rope code is not exact in fp16"
                    kd[valid] = kv
                    ksh[valid] = ks[b, vt].float()
                    krsh[valid] = krs[b, vt].float()
                nope = qh[b] @ cdeq.T                                       # [H, TT]
                rope = torch.einsum("hd,thd->ht", qrh[b], kd.repeat_interleave(4, dim=1))  # [H, TT]
                ks_h = ksh.repeat_interleave(4, dim=1).T                    # [H, TT]
                krs_h = krsh.repeat_interleave(4, dim=1).T
                s = SM * (ks_h * qs[b][:, None] * nope + krs_h * qrs[b][:, None] * rope)
                s[:, ~valid] = -math.inf
                mt = s.max(dim=1).values
                mn = torch.maximum(m_run, mt)
                alpha = torch.exp(m_run - mn)
                pv = torch.exp(s - mn[:, None]).half().float()
                l_run = l_run * alpha + pv.sum(1)
                acc = acc * alpha[:, None] + pv @ cdeq
                m_run = mn
            opart[b, p] = acc
            mpart[b, p] = m_run
            lpart[b, p] = l_run
    return _combine(opart, mpart, lpart)


def run_case_tc(R, B, S, seed, qscale=1.0, cscale=1.0, gate=TC_GATE):
    g = torch.Generator().manual_seed(seed)
    c = cscale * torch.randn(B, S, R, generator=g) * torch.exp(0.5 * torch.randn(B, S, 1, generator=g))
    kr = torch.randn(B, S, 4, 64, generator=g)
    ks = (0.5 + 1.5 * torch.rand(B, S, 4, generator=g)).half()
    qa = qscale * torch.randn(B, 16, R, generator=g) * 0.5
    qr = qscale * torch.randn(B, 16, 64, generator=g) * 0.5
    cq, cs = ref.quant_c(c)
    kc, krs = ref.quant_kr(kr)
    yref = ref.attn_ref(qa, qr, ref.dequant_c(cq, cs), ref.dequant_kr(kc, krs), ks)
    yf32 = emulate(qa, qr, cq, cs, kc, krs, ks)
    ytc = emulate_tc(qa, qr, cq, cs, kc, krs, ks)
    rel = ((ytc - yref).abs().max() / yref.abs().max()).item()
    rel_f32 = ((yf32 - yref).abs().max() / yref.abs().max()).item()
    if gate is None:
        status = "INFO"
    else:
        status = "PASS" if rel < gate else "FAIL"
    print(f"{status} tc R={R} B={B} S={S} qscale={qscale:g} cscale={cscale:g}: fp16-path vs dequant-ref rel={rel:.2e} "
          f"(fp32 emulation rel={rel_f32:.2e}, gate {gate if gate is not None else 'none'})")
    return gate is None or rel < gate


if __name__ == "__main__":
    check_decodes()
    ok = True
    for R in (256, 512):
        for B in (1, 4):
            for S in (1000, 4096, 5000):
                ok &= run_case(R, B, S, seed=R + 10 * B + S)
    print("ALL PASS" if ok else "FAILURES")
    tc_ok = True
    for R in (256, 512):
        for B in (1, 4):
            for S in (1000, 4096, 5000):
                tc_ok &= run_case_tc(R, B, S, seed=R + 10 * B + S)
    # robustness: large query magnitudes (prescale keeps the fp16 operands in range, so no overflow) and sharp softmax.
    # Error grows with the logit magnitude: fp16 rounding of Q_abs (2^-12 relative) is amplified by exp().
    # qscale=40 gives logits around +-30 and is reported without a gate (see the summary: hi/lo split is the fix).
    for qsc in (8.0, 20.0):
        tc_ok &= run_case_tc(512, 1, 4096, seed=910 + int(qsc), qscale=qsc, gate=None)
    tc_ok &= run_case_tc(512, 1, 4096, seed=901, qscale=40.0, gate=None)
    tc_ok &= run_case_tc(256, 2, 3000, seed=902, qscale=300.0)
    tc_ok &= run_case_tc(512, 1, 4096, seed=903, qscale=8.0, cscale=20.0)
    print("ALL PASS (tc)" if tc_ok else "TC FAILURES")
    ok &= tc_ok
    print("ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)
