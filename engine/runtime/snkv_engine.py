"""SenseNovaKV single-sequence decode engine for SenseNova-Loop-9B (Qwen3.5 mixer + mHC + Loop plan).

Self-contained on purpose: kaggle/engine_e2e/engine_e2e.py embeds this file verbatim (kaggle/render.py), so it
imports only torch and the standard library at module level. The CPU reference backend imports engine/ref/* and
is only used by tests (see README.md in this directory).

Pieces
  pack_w4 / dequant_rows     4-bit group-64 weight format (same bytes as engine/ref/w4_ref.py)
  build_plan, dims_from_hf   the 56-visit execution plan of the Loop model
  export_host                HF model -> host dict of packed weights (packing runs on each weight's own GPU)
  to_device                  host dict -> device weights in the dtypes the kernels need
  Ops                        kernel calls: 'cuda' (compiled extensions) or 'ref' (torch references, CPU tests)
  Engine                     static decode step over preallocated KV / DeltaNet state, CUDA-graph captured
  snapshot_cache             HF DynamicCache (after prefill) -> CPU dict in the engine's layout
"""
import os

import torch
import torch.nn.functional as F

GROUP = 64
EPS = 1e-6
MHC_ITERS = 20
W4_RPW, W4_SLAB = 2, 2048  # w4_gemv launch config: N % 8 == 0 and K % 2048 == 0 for every linear in the model


class Policy:
    """Activation dtypes. act: GEMV / attention / MLP side (fp16 on T4). mhc: mHC, DeltaNet, DeltaNet conv state."""

    def __init__(self, act, mhc):
        self.act, self.mhc = act, mhc

    def __repr__(self):
        return f"Policy(act={self.act}, mhc={self.mhc})"


CUDA_POLICY = Policy(torch.float16, torch.bfloat16)  # the kernels' contracts: w4 fp16 in/out, mhc & deltanet bf16
REF_POLICY = Policy(torch.float32, torch.float32)    # CPU tests: everything fp32, so the plumbing is checked exactly


# ----------------------------------------------------------------------------- 4-bit weight format
def pack_w4(W, rows_per_chunk=2048):
    """W [N, K] (any float dtype, any device) -> (qw uint8 [N, K/2], meta fp16 [N, K/64, 2] = (scale, minv))."""
    N, K = W.shape
    assert K % GROUP == 0, K
    G = K // GROUP
    packs, metas = [], []
    for i in range(0, N, rows_per_chunk):
        Wg = W[i:i + rows_per_chunk].float().reshape(-1, G, GROUP)
        mn = Wg.amin(-1)
        mx = Wg.amax(-1)
        s16 = ((mx - mn) / 15.0).half()
        s16 = torch.where(s16 == 0, torch.ones_like(s16), s16)
        m16 = mn.half()
        q = torch.round((Wg - m16.float()[..., None]) / s16.float()[..., None]).clamp_(0, 15).to(torch.uint8)
        q = q.reshape(-1, 2 * G, 32)
        packs.append((q[..., :16] | (q[..., 16:] << 4)).reshape(-1, K // 2).contiguous())
        metas.append(torch.stack([s16, m16], -1).contiguous())
    return torch.cat(packs, 0), torch.cat(metas, 0)


def dequant_rows(qw, meta, K):
    """Rows of the packed format -> float32 [T, K]. Exact inverse of pack_w4 (up to the stored fp16 scale/minv)."""
    T = qw.shape[0]
    G = K // GROUP
    p = qw.reshape(T, 2 * G, 16)
    q = torch.cat([p & 0xF, p >> 4], -1).reshape(T, G, GROUP).float()
    return (q * meta[..., 0:1].float() + meta[..., 1:2].float()).reshape(T, K)


# ----------------------------------------------------------------------------- plan and dims
def build_plan(num_layers, loop):
    """Physical layer index for each of the execution visits (SenseNova modeling_sensenova_loop.execution_plan)."""
    s, e, c = int(loop["start"]), int(loop["end"]), int(loop["eval_count"])
    return tuple(range(s)) + tuple(range(s, e)) * c + tuple(range(e, num_layers))


def visit_table(layer_types, plan, mhc):
    """[(visit, physical, kind, is_mhc)] with kind 'full' or 'linear'."""
    rows = []
    for v, phys in enumerate(plan):
        kind = "full" if layer_types[phys] == "full_attention" else "linear"
        is_m = mhc is not None and mhc[0] <= phys < mhc[1]
        rows.append((v, phys, kind, is_m))
    return rows


def kv_bytes_per_token(dims, act_bytes=2):
    n_full = sum(1 for r in visit_table(dims["layer_types"], dims["plan"], dims["mhc"]) if r[2] == "full")
    return n_full * 2 * dims["Hkv"] * dims["Dh"] * act_bytes


# ----------------------------------------------------------------------------- host export (HF -> packed)
def export_host(model, pack_rows=2048):
    """Quantize every linear of an HF SenseNova model. Packing runs on each weight's device; results live on CPU.
    Norms, mHC routers, A_log, dt_bias and conv taps are copied as fp32 (exact: the source is bf16 or fp16)."""
    tm = model.model.language_model
    tc = tm.config
    raw = model.config.to_dict()
    mhc = raw.get("mhc")
    dims = dict(
        H=int(tc.hidden_size), L=int(tc.num_hidden_layers), I=int(tc.intermediate_size), V=int(tc.vocab_size),
        Hq=int(tc.num_attention_heads), Hkv=int(tc.num_key_value_heads), Dh=int(tc.head_dim),
        rot=2 * int(len(tm.rotary_emb.inv_freq)),
        Hk=int(tc.linear_num_key_heads), Hv=int(tc.linear_num_value_heads), Dk=int(tc.linear_key_head_dim),
        Dv=int(tc.linear_value_head_dim), CK=int(tc.linear_conv_kernel_dim), eps=float(tc.rms_norm_eps),
        layer_types=list(tc.layer_types), loop=dict(raw["loop_settings"]), mhc=None,
    )
    dims["plan"] = build_plan(dims["L"], dims["loop"])
    dims["mhc"] = (int(mhc["start"]), int(mhc["end"])) if mhc else None
    dims["CD"] = 2 * dims["Hk"] * dims["Dk"] + dims["Hv"] * dims["Dv"]
    assert dims["Hk"] * dims["Dk"] * 2 + dims["Hv"] * dims["Dv"] == dims["CD"]
    assert dims["Dh"] % 2 == 0 and dims["rot"] <= dims["Dh"]

    def lin(mod):
        W = mod.weight.detach()
        qw, meta = pack_w4(W, pack_rows)
        return {"qw": qw.cpu(), "meta": meta.cpu(), "N": int(W.shape[0]), "K": int(W.shape[1])}

    def cpu32(t):
        return t.detach().float().cpu().contiguous()

    layers = {}
    for i, layer in enumerate(tm.layers):
        Lh = {"ln_in": cpu32(layer.input_layernorm.weight), "ln_post": cpu32(layer.post_attention_layernorm.weight),
              "gate": lin(layer.mlp.gate_proj), "up": lin(layer.mlp.up_proj), "down": lin(layer.mlp.down_proj)}
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            Lh.update(in_qkv=lin(la.in_proj_qkv), in_z=lin(la.in_proj_z), in_b=lin(la.in_proj_b),
                      in_a=lin(la.in_proj_a), out=lin(la.out_proj))
            cw = la.conv1d.weight.detach()
            Lh.update(conv_w=cpu32(cw.reshape(cw.shape[0], -1)), A_log=cpu32(la.A_log), dt_bias=cpu32(la.dt_bias),
                      dn_norm=cpu32(la.norm.weight))
        else:
            sa = layer.self_attn
            Lh.update(q=lin(sa.q_proj), k=lin(sa.k_proj), v=lin(sa.v_proj), o=lin(sa.o_proj),
                      qn=cpu32(sa.q_norm.weight), kn=cpu32(sa.k_norm.weight))
        if hasattr(layer, "mhc_attn"):
            Lh["mhc_attn"] = {"W": cpu32(layer.mhc_attn.weight), "b": cpu32(layer.mhc_attn.bias)}
            Lh["mhc_ffn"] = {"W": cpu32(layer.mhc_ffn.weight), "b": cpu32(layer.mhc_ffn.bias)}
        layers[i] = Lh
    return {"dims": dims, "inv_freq": cpu32(tm.rotary_emb.inv_freq),
            "embed": lin(tm.embed_tokens), "norm": cpu32(tm.norm.weight), "lm_head": lin(model.lm_head),
            "layers": layers}


def to_device(host, device, P):
    """Host dict -> device dict in the dtypes the kernels consume (see Policy)."""
    dev = torch.device(device)

    def lin(W):
        return {"qw": W["qw"].to(dev), "meta": W["meta"].to(dev), "N": W["N"], "K": W["K"]}

    def f32(t):
        return t.to(dev, torch.float32)

    layers = {}
    for phys, Lh in host["layers"].items():
        Ld = {k: lin(Lh[k]) for k in ("gate", "up", "down") if k in Lh}
        for k in ("q", "k", "v", "o", "in_qkv", "in_z", "in_b", "in_a", "out"):
            if k in Lh:
                Ld[k] = lin(Lh[k])
        Ld["ln_in_a"] = Lh["ln_in"].to(dev, P.act)
        Ld["ln_in_m"] = Lh["ln_in"].to(dev, P.mhc)
        Ld["ln_post_a"] = Lh["ln_post"].to(dev, P.act)
        Ld["ln_post_m"] = Lh["ln_post"].to(dev, P.mhc)
        if "qn" in Lh:
            Ld["qn"], Ld["kn"] = f32(Lh["qn"]), f32(Lh["kn"])
        if "A_log" in Lh:
            Ld["A_log"], Ld["dt_bias"] = f32(Lh["A_log"]), f32(Lh["dt_bias"])
            Ld["conv_w"] = Lh["conv_w"].to(dev, P.mhc)
            Ld["dn_norm"] = Lh["dn_norm"].to(dev, P.mhc)
        if "mhc_attn" in Lh:
            Ld["mhc_attn_W"], Ld["mhc_attn_b"] = f32(Lh["mhc_attn"]["W"]), f32(Lh["mhc_attn"]["b"])
            Ld["mhc_ffn_W"], Ld["mhc_ffn_b"] = f32(Lh["mhc_ffn"]["W"]), f32(Lh["mhc_ffn"]["b"])
        layers[phys] = Ld
    return {"dims": host["dims"], "inv_freq": f32(host["inv_freq"]), "embed": lin(host["embed"]),
            "lm_head": lin(host["lm_head"]), "final_a": host["norm"].to(dev, P.act), "layers": layers}


def weights_bytes(st):
    total = 0
    for W in [st["embed"], st["lm_head"]] + [x for Ld in st["layers"].values() for x in Ld.values()]:
        if isinstance(W, dict):
            if "qw" in W:
                total += W["qw"].numel() * W["qw"].element_size() + W["meta"].numel() * W["meta"].element_size()
            else:
                total += sum(t.numel() * t.element_size() for t in W.values() if torch.is_tensor(t))
        elif torch.is_tensor(W):
            total += W.numel() * W.element_size()
    total += st["final_a"].numel() * st["final_a"].element_size() + st["inv_freq"].numel() * 4
    return total


# ----------------------------------------------------------------------------- kernels
def load_cuda_extensions(sources, build_dir):
    """sources: {'w4_gemv': path, 'mhc_fused': path, 'deltanet_fused': path} -> {name: extension}.
    Each .cu file carries its own PYBIND11 module, guarded by the macro passed here."""
    from torch.utils.cpp_extension import load
    macros = {"w4_gemv": "W4_TORCH_EXTENSION", "mhc_fused": "MHC_TORCH_EXTENSION",
              "deltanet_fused": "DELTANET_TORCH_EXTENSION"}
    props = torch.cuda.get_device_properties(0)
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{props.major}.{props.minor}"
    os.makedirs(build_dir, exist_ok=True)
    out = {}
    for name, macro in macros.items():
        out[name] = load(name=name, sources=[sources[name]], build_directory=build_dir,
                         extra_cflags=["-O3", f"-D{macro}"],
                         extra_cuda_cflags=["-O3", f"-D{macro}", "-lineinfo"], verbose=False)
    return out


class Ops:
    """The four kernel calls the engine makes. backend 'cuda' needs exts (from load_cuda_extensions);
    backend 'ref' needs ref_dir (engine/ref) and runs the PyTorch references on any device."""

    def __init__(self, backend, exts=None, ref_dir=None):
        assert backend in ("cuda", "ref")
        self.backend = backend
        if backend == "cuda":
            self.w4, self.mhc_ext, self.dn = exts["w4_gemv"], exts["mhc_fused"], exts["deltanet_fused"]
        else:
            import sys
            sys.path.insert(0, ref_dir)
            import deltanet_ref
            import mhc_ref
            self.mhc_ref, self.dn_ref = mhc_ref, deltanet_ref

    def lin(self, x, W, norm=None, resid=None):
        """y = rmsnorm?(x) @ W^T (+ resid). x [T, K] act dtype, W packed. norm: (1+w) RMSNorm weight, act dtype."""
        if self.backend == "cuda":
            return self.w4.w4_gemv(x.contiguous(), W["qw"], W["meta"], norm, resid, EPS, W4_RPW, W4_SLAB)
        xf = x.float()
        if norm is not None:
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + norm.float())
        y = xf @ dequant_rows(W["qw"], W["meta"], W["K"]).t()
        if resid is not None:
            y = y + resid.float()
        return y.to(x.dtype)

    def mhc(self, residual, branch_prev, post_prev, mix_prev, W, b, norm_w):
        """Merge (optional) + router + RMSNorm. Returns (residual, out, post [T,2], mix [T,2,2]); residual is
        updated in place when branch_prev is given (the kernel's contract)."""
        if self.backend == "cuda":
            r, out, post, mix = self.mhc_ext.mhc_step(residual, branch_prev, post_prev, mix_prev, W, b, norm_w,
                                                      MHC_ITERS, 1.0)
            return r, out, post, mix
        if branch_prev is not None:
            residual.copy_(self.mhc_ref.merge(residual, branch_prev, post_prev, mix_prev))
        branch, post, mix = self.mhc_ref.router(residual, W, b, MHC_ITERS, 1.0)
        return residual, self.mhc_ref.rmsnorm(branch, norm_w), post, mix

    def deltanet(self, qkv, z, b, a, conv_state, rec_state, A_log, dt_bias, conv_w, norm_w):
        """One Gated DeltaNet decode step. Updates conv_state and rec_state in place. Returns [1, Hv*Dv] mhc dtype."""
        if self.backend == "cuda":
            return self.dn.deltanet_decode(qkv, z, b, a, conv_state, rec_state, A_log, dt_bias, conv_w, norm_w)
        return self._ref_deltanet(qkv, z, b, a, conv_state, rec_state, A_log, dt_bias, conv_w, norm_w)

    def _ref_deltanet(self, qkv, z, b, a, conv_state, rec_state, A_log, dt_bias, conv_w, norm_w):
        R = self.dn_ref
        T = qkv.shape[0]
        Hv, Dk, Dv = rec_state.shape[1], rec_state.shape[2], rec_state.shape[3]
        CD = qkv.shape[-1]
        KD = (CD - Hv * Dv) // 2
        Hk = KD // Dk
        mixed = R.causal_conv1d_update(qkv.reshape(T, CD, 1), conv_state, conv_w, None, "silu")
        mixed = mixed.transpose(1, 2)
        q, k, v = torch.split(mixed, [KD, KD, Hv * Dv], dim=-1)
        q = q.reshape(T, 1, Hk, Dk)
        k = k.reshape(T, 1, Hk, Dk)
        v = v.reshape(T, 1, Hv, Dv)
        beta, g = R.compute_gates(a.reshape(T, 1, Hv), b.reshape(T, 1, Hv), A_log, dt_bias)
        if Hv // Hk > 1:
            q = q.repeat_interleave(Hv // Hk, dim=2)
            k = k.repeat_interleave(Hv // Hk, dim=2)
        core, last = R.torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=rec_state,
                                                        output_final_state=True, use_qk_l2norm_in_kernel=True)
        rec_state.copy_(last)
        out = R.rmsnorm_gated(core.reshape(-1, Dv), z.reshape(-1, Dv), norm_w)
        return out.reshape(T, Hv * Dv)


# ----------------------------------------------------------------------------- HF cache -> engine layout
def _cache_state(cache, v, which):
    getter = getattr(cache, "get_conv_state" if which == "conv" else "get_recurrent_state", None)
    if getter is not None:
        return getter(v)
    layer = cache.layers[v]
    return (layer.conv_states if which == "conv" else layer.recurrent_states)[0]


def snapshot_cache(cache, kinds):
    """HF DynamicCache after prefill -> CPU dict {'S', 'kv': {v: (K, V)}, 'conv': {v: ..}, 'rec': {v: ..}}.
    kinds[v] is 'full' or 'linear' for execution visit v (slot v of the Loop cache)."""
    snap = {"kv": {}, "conv": {}, "rec": {}, "S": None}
    for v, kind in enumerate(kinds):
        if kind == "full":
            layer = cache.layers[v]
            snap["kv"][v] = (layer.keys.detach().cpu().clone(), layer.values.detach().cpu().clone())
            if snap["S"] is None:
                snap["S"] = int(layer.keys.shape[2])
        else:
            snap["conv"][v] = _cache_state(cache, v, "conv").detach().cpu().clone()
            snap["rec"][v] = _cache_state(cache, v, "rec").detach().float().cpu().clone()
    return snap


# ----------------------------------------------------------------------------- engine
def merge_torch(R, f, post, mix, P):
    """mHC merge (MHCRouter.merge): mix @ R + post * f, dtype as the kernels' contract (mhc policy)."""
    mixed = torch.matmul(mix.to(P.mhc), R)
    upd = post.to(P.mhc).unsqueeze(-1) * f.unsqueeze(-2)
    return mixed + upd


class Engine:
    """Single sequence, batch 1. One decode step = one pass over the 56 visits (see build_plan).
    State per visit: full attention K/V [1, Hkv, max_len, Dh] (act dtype, post-RoPE K);
    DeltaNet conv [1, CD, CK] (mhc dtype) and recurrent [1, Hv, Dk, Dv] (fp32).
    All buffers are allocated once; step() reads token/position from device buffers, so the captured graph
    is the same for every position."""

    def __init__(self, st, ops, max_len, policy, device="cuda:0"):
        self.st, self.ops, self.P = st, ops, policy
        self.dev = torch.device(device)
        d = st["dims"]
        self.d = d
        self.max_len = int(max_len)
        self.visits = visit_table(d["layer_types"], d["plan"], d["mhc"])
        self.kinds = [r[2] for r in self.visits]
        self.scale = d["Dh"] ** -0.5
        act, mdt = policy.act, policy.mhc
        self.K, self.V, self.conv, self.rec = {}, {}, {}, {}
        for v, phys, kind, is_m in self.visits:
            if kind == "full":
                self.K[v] = torch.zeros(1, d["Hkv"], self.max_len, d["Dh"], dtype=act, device=self.dev)
                self.V[v] = torch.zeros(1, d["Hkv"], self.max_len, d["Dh"], dtype=act, device=self.dev)
            else:
                self.conv[v] = torch.zeros(1, d["CD"], d["CK"], dtype=mdt, device=self.dev)
                self.rec[v] = torch.zeros(1, d["Hv"], d["Dk"], d["Dv"], dtype=torch.float32, device=self.dev)
        freqs = torch.arange(self.max_len, dtype=torch.float32, device=self.dev)[:, None] * \
            st["inv_freq"].to(self.dev)[None, :]
        emb = torch.cat([freqs, freqs], -1)
        self.cos_tab = emb.cos().to(act)  # HF rounds cos/sin to the activation dtype before use
        self.sin_tab = emb.sin().to(act)
        self.ar = torch.arange(self.max_len, device=self.dev)
        self.tok_buf = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.pos_buf = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.pos = 0
        self.graph = None
        self._out = None

    # -- sizes
    def kv_bytes(self):
        return sum(t.numel() * t.element_size() for t in list(self.K.values()) + list(self.V.values()))

    def state_bytes(self):
        return sum(t.numel() * t.element_size() for t in list(self.conv.values()) + list(self.rec.values()))

    # -- building blocks (all shapes [1, *] or [Heads, *]; no host sync)
    def _rope(self, x, cos, sin):
        r = self.d["rot"]
        xr, xp = x[..., :r], x[..., r:]
        h = r // 2
        rh = torch.cat([-xr[..., h:], xr[..., :h]], -1)
        return torch.cat([xr * cos + rh * sin, xp], -1)

    def _rmsnorm_zc(self, x, w32):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w32)).to(self.P.act)

    def _attn(self, v, L, x, norm, resid):
        d, act = self.d, self.P.act
        Hq, Hkv, Dh = d["Hq"], d["Hkv"], d["Dh"]
        qg = self.ops.lin(x, L["q"], norm)
        k = self.ops.lin(x, L["k"], norm)
        vv = self.ops.lin(x, L["v"], norm)
        qg = qg.reshape(Hq, 2 * Dh)
        q, gate = qg[:, :Dh], qg[:, Dh:]                       # per-head interleave: [query 256 | gate 256]
        q = self._rmsnorm_zc(q, L["qn"])
        k = self._rmsnorm_zc(k.reshape(Hkv, Dh), L["kn"])
        vv = vv.reshape(Hkv, Dh)
        cos = self.cos_tab.index_select(0, self.pos_buf)       # [1, rot]
        sin = self.sin_tab.index_select(0, self.pos_buf)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        Kc, Vc = self.K[v][0], self.V[v][0]                    # [Hkv, max_len, Dh] views
        Kc.index_copy_(1, self.pos_buf, k.unsqueeze(1))
        Vc.index_copy_(1, self.pos_buf, vv.unsqueeze(1))
        rep = Hq // Hkv                                        # query head h uses kv head h // rep
        qh = (q * self.scale).reshape(Hkv, rep, Dh)
        s = torch.matmul(qh, Kc.transpose(1, 2))               # [Hkv, rep, max_len]
        s = s.masked_fill(~self.mask, float("-inf"))
        p = torch.softmax(s.float(), dim=-1).to(act)
        o = torch.matmul(p, Vc).reshape(Hq, Dh)
        o = o * torch.sigmoid(gate)                            # output gate, before o_proj
        return self.ops.lin(o.reshape(1, Hq * Dh), L["o"], None, resid)

    def _dn(self, v, L, x, norm, resid):
        P = self.P
        qkv = self.ops.lin(x, L["in_qkv"], norm)
        z = self.ops.lin(x, L["in_z"], norm)
        b = self.ops.lin(x, L["in_b"], norm)
        a = self.ops.lin(x, L["in_a"], norm)
        o = self.ops.deltanet(qkv.to(P.mhc), z.to(P.mhc), b.to(P.mhc), a.to(P.mhc), self.conv[v], self.rec[v],
                              L["A_log"], L["dt_bias"], L["conv_w"], L["dn_norm"])
        return self.ops.lin(o.to(P.act), L["out"], None, resid)

    def _mlp(self, L, x, norm, resid):
        g = self.ops.lin(x, L["gate"], norm)
        u = self.ops.lin(x, L["up"], norm)
        return self.ops.lin(F.silu(g) * u, L["down"], None, resid)

    def _plain_visit(self, v, phys, kind, L, h):
        """Stock Qwen3.5 layer: h += mixer(norm(h)); h += mlp(norm(h)). The norm runs in the GEMV prologues."""
        if kind == "full":
            h = self._attn(v, L, h, L["ln_in_a"], h)
        else:
            h = self._dn(v, L, h, L["ln_in_a"], h)
        return self._mlp(L, h, L["ln_post_a"], h)

    def _mhc_visit(self, v, phys, kind, L, h, R, pend):
        """Two mHC sub-blocks per visit. Enter at the first mHC layer (expand h to 2 streams); each sub-block's
        merge of the previous sub-block output happens inside the next mhc_step call. Returns (R, pend)."""
        P, dims = self.P, self.d
        if phys == dims["mhc"][0]:
            R = h.to(P.mhc).unsqueeze(-2).expand(1, 2, dims["H"]).contiguous()
            pend = None
        pb, pp, pm = pend if pend is not None else (None, None, None)
        R, u, post1, mix1 = self.ops.mhc(R, pb, pp, pm, L["mhc_attn_W"], L["mhc_attn_b"], L["ln_in_m"])
        if kind == "full":
            a = self._attn(v, L, u.to(P.act), None, None)
        else:
            a = self._dn(v, L, u.to(P.act), None, None)
        R, b2, post2, mix2 = self.ops.mhc(R, a.to(P.mhc), post1, mix1, L["mhc_ffn_W"], L["mhc_ffn_b"], L["ln_post_m"])
        f = self._mlp(L, b2.to(P.act), None, None).to(P.mhc)
        return R, (f, post2, mix2)

    def _run(self):
        """One decode step for the token in tok_buf at position pos_buf. Returns logits [1, V] (act dtype)."""
        P, d = self.P, self.d
        self.mask = (self.ar <= self.pos_buf).view(1, 1, -1)
        E = self.st["embed"]
        h = dequant_rows(E["qw"].index_select(0, self.tok_buf), E["meta"].index_select(0, self.tok_buf), d["H"])
        h = h.to(P.act)
        R, pend = None, None
        mstart, mend = d["mhc"]
        for v, phys, kind, is_m in self.visits:
            L = self.st["layers"][phys]
            if not is_m:
                h = self._plain_visit(v, phys, kind, L, h)
                continue
            R, pend = self._mhc_visit(v, phys, kind, L, h, R, pend)
            if phys == mend - 1:                                # leave: average the streams, back to 1 stream
                f, post2, mix2 = pend
                h = merge_torch(R, f, post2, mix2, P).mean(-2).to(P.act)
                R, pend = None, None
        return self.ops.lin(h, self.st["lm_head"], self.st["final_a"], None)

    # -- public API
    def capture(self):
        """Capture one decode step as a CUDA graph (CUDA devices only). Warm-up runs at the last slot and the
        DeltaNet states are restored afterwards, so the engine state is untouched."""
        if self.dev.type != "cuda":
            return
        saved = {v: (self.conv[v].clone(), self.rec[v].clone()) for v in self.conv}
        self.tok_buf.fill_(0)
        self.pos_buf.fill_(self.max_len - 1)
        side = torch.cuda.Stream(device=self.dev)
        side.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._run()
        torch.cuda.current_stream(self.dev).wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._out = self._run()
        self.graph = g
        for v, (c, r) in saved.items():
            self.conv[v].copy_(c)
            self.rec[v].copy_(r)
        torch.cuda.synchronize(self.dev)

    def load_prefill(self, snap):
        """Copy an HF prefill snapshot (snapshot_cache) into the engine; the next step() is at position S."""
        S = int(snap["S"])
        assert S + 1 < self.max_len, (S, self.max_len)
        for v, phys, kind, is_m in self.visits:
            if kind == "full":
                k, vv = snap["kv"][v]
                self.K[v][:, :, :S].copy_(k.to(self.dev, self.P.act))
                self.V[v][:, :, :S].copy_(vv.to(self.dev, self.P.act))
            else:
                self.conv[v].copy_(snap["conv"][v].to(self.dev, self.P.mhc))
                self.rec[v].copy_(snap["rec"][v].to(self.dev, torch.float32))
        self.pos = S

    def step(self, token):
        """Feed one token at the current position. Returns the logits view [1, V]; clone it to keep it."""
        assert self.pos < self.max_len - 1, "KV buffer full"
        self.tok_buf.fill_(int(token))
        self.pos_buf.fill_(self.pos)
        if self.graph is not None:
            self.graph.replay()
            out = self._out
        else:
            out = self._run()
        self.pos += 1
        return out
