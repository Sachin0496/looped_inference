"""JAX port of the SenseNova-Loop-9B text model, full-sequence (prefill / training) forward.

Mirrors the PyTorch reference:
  * SenseNova-Loop `modeling_sensenova_loop.py` (Loop plan, MHC routers, enter/leave, MHC layer)
  * transformers `qwen3_5/modeling_qwen3_5.py` (Gated DeltaNet prefill, gated full attention,
    zero-centered RMSNorm, gated RMSNorm, partial RoPE, SwiGLU MLP)

Layout: activations are [B, T, ...]. Gated DeltaNet heads are [B, T, H, D]; the chunked
delta rule works internally in [B, H, N, C, D] (N chunks of C tokens) in float32.
Weights keep the HF nn.Linear layout [out, in].

Physical layer schedule (see ModelCfg): pre layers [0, loop_start), loop section
[loop_start, loop_end) executed `loop_count` times with shared weights, post layers
[loop_end, n_layers). The loop section is scanned over its groups of 4 layers
(linear, linear, linear, full), so layers 4..27 must follow that pattern.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

F32 = jnp.float32
HIGHEST = lax.Precision.HIGHEST
LINEAR = "linear_attention"
FULL = "full_attention"
GROUP = 4  # layers per loop group: [linear, linear, linear, full]
GROUP_PATTERN = (LINEAR, LINEAR, LINEAR, FULL)
F32_KEYS = ("A_log", "dt_bias", "mhc_attn_w", "mhc_attn_b", "mhc_ffn_w", "mhc_ffn_b")


# --------------------------------------------------------------------------- config
@dataclasses.dataclass(frozen=True)
class ModelCfg:
    hidden: int
    n_layers: int
    layer_types: tuple
    n_heads: int
    n_kv: int
    head_dim: int
    rot_dim: int
    rope_theta: float
    eps: float
    lin_nk: int
    lin_nv: int
    lin_dk: int
    lin_dv: int
    conv_k: int
    inter: int
    vocab: int
    loop_start: int
    loop_end: int
    loop_count: int
    mhc_start: int
    mhc_end: int
    streams: int
    sinkhorn_iters: int
    pre_scale: float

    @classmethod
    def from_hf_config(cls, raw: dict) -> "ModelCfg":
        t = raw["text_config"]
        rp = t.get("rope_parameters") or {}
        theta = rp.get("rope_theta", t.get("rope_theta"))
        partial = rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))
        head_dim = int(t["head_dim"])
        loop = raw.get("loop_settings") or t.get("loop_settings")
        mhc = raw.get("mhc") or t.get("mhc")
        if loop is None or mhc is None:
            raise ValueError("config needs loop_settings and mhc")
        if loop["counts"] != [loop["eval_count"]]:
            raise ValueError("only fixed-count Loop checkpoints are supported")
        return cls(
            hidden=int(t["hidden_size"]),
            n_layers=int(t["num_hidden_layers"]),
            layer_types=tuple(t["layer_types"]),
            n_heads=int(t["num_attention_heads"]),
            n_kv=int(t["num_key_value_heads"]),
            head_dim=head_dim,
            rot_dim=int(head_dim * partial),
            rope_theta=float(theta),
            eps=float(t["rms_norm_eps"]),
            lin_nk=int(t["linear_num_key_heads"]),
            lin_nv=int(t["linear_num_value_heads"]),
            lin_dk=int(t["linear_key_head_dim"]),
            lin_dv=int(t["linear_value_head_dim"]),
            conv_k=int(t["linear_conv_kernel_dim"]),
            inter=int(t["intermediate_size"]),
            vocab=int(t["vocab_size"]),
            loop_start=int(loop["start"]),
            loop_end=int(loop["end"]),
            loop_count=int(loop["eval_count"]),
            mhc_start=int(mhc["start"]),
            mhc_end=int(mhc["end"]),
            streams=int(mhc["streams"]),
            sinkhorn_iters=int(mhc["iterations"]),
            pre_scale=float(mhc.get("pre_scale", 1.0)),
        )


def _check_cfg(cfg: ModelCfg) -> None:
    if (cfg.mhc_start, cfg.mhc_end) != (cfg.loop_start, cfg.loop_end):
        raise ValueError("this port expects the mHC range to equal the Loop range")
    n = cfg.loop_end - cfg.loop_start
    if n % GROUP != 0 or n == 0:
        raise ValueError("Loop section must be a whole number of 4-layer groups")
    for g in range(n // GROUP):
        got = cfg.layer_types[cfg.loop_start + GROUP * g: cfg.loop_start + GROUP * (g + 1)]
        if tuple(got) != GROUP_PATTERN:
            raise ValueError(f"Loop group {g} has layer types {got}, expected {GROUP_PATTERN}")


# --------------------------------------------------------------------------- params
def _layer_np(sd: dict, prefix: str, cfg: ModelCfg, i: int) -> dict:
    """Raw numpy tensors of physical layer i under short keys."""
    p = f"{prefix}layers.{i}."
    get = lambda name: np.asarray(sd[p + name])
    d = {
        "input_layernorm": get("input_layernorm.weight"),
        "post_attention_layernorm": get("post_attention_layernorm.weight"),
        "gate_proj": get("mlp.gate_proj.weight"),
        "up_proj": get("mlp.up_proj.weight"),
        "down_proj": get("mlp.down_proj.weight"),
    }
    if cfg.layer_types[i] == LINEAR:
        la = "linear_attn."
        d.update(
            in_proj_qkv=get(la + "in_proj_qkv.weight"),
            in_proj_z=get(la + "in_proj_z.weight"),
            in_proj_b=get(la + "in_proj_b.weight"),
            in_proj_a=get(la + "in_proj_a.weight"),
            conv1d=get(la + "conv1d.weight"),
            A_log=get(la + "A_log"),
            dt_bias=get(la + "dt_bias"),
            norm=get(la + "norm.weight"),
            out_proj=get(la + "out_proj.weight"),
        )
    else:
        sa = "self_attn."
        d.update(
            q_proj=get(sa + "q_proj.weight"),
            k_proj=get(sa + "k_proj.weight"),
            v_proj=get(sa + "v_proj.weight"),
            q_norm=get(sa + "q_norm.weight"),
            k_norm=get(sa + "k_norm.weight"),
            o_proj=get(sa + "o_proj.weight"),
        )
    if cfg.mhc_start <= i < cfg.mhc_end:
        d.update(
            mhc_attn_w=get("mhc_attn.weight"),
            mhc_attn_b=get("mhc_attn.bias"),
            mhc_ffn_w=get("mhc_ffn.weight"),
            mhc_ffn_b=get("mhc_ffn.bias"),
        )
    return d


def _to_jnp(d: dict, dtype) -> dict:
    return {k: jnp.asarray(v, dtype=F32 if k in F32_KEYS else dtype) for k, v in d.items()}


def params_from_state_dict(sd: dict, cfg: ModelCfg, dtype) -> dict:
    """Build the JAX parameter pytree from a torch state dict (name -> numpy array)."""
    _check_cfg(cfg)
    prefix = "model.language_model." if "model.language_model.embed_tokens.weight" in sd else "model."
    if prefix + "embed_tokens.weight" not in sd:
        raise KeyError("text state dict not found (expected model.language_model.* keys)")
    if "lm_head.weight" not in sd:
        raise KeyError("lm_head.weight missing from state dict")

    pre = [_to_jnp(_layer_np(sd, prefix, cfg, i), dtype) for i in range(cfg.loop_start)]
    post = [_to_jnp(_layer_np(sd, prefix, cfg, i), dtype) for i in range(cfg.loop_end, cfg.n_layers)]
    n_groups = (cfg.loop_end - cfg.loop_start) // GROUP
    loop = {}
    for j in range(GROUP):
        per_group = [
            _layer_np(sd, prefix, cfg, cfg.loop_start + GROUP * g + j) for g in range(n_groups)
        ]
        stacked = {k: np.stack([d[k] for d in per_group]) for k in per_group[0]}
        loop[f"l{j}"] = _to_jnp(stacked, dtype)
    return {
        "embed": jnp.asarray(sd[prefix + "embed_tokens.weight"], dtype=dtype),
        "norm": jnp.asarray(sd[prefix + "norm.weight"], dtype=dtype),
        "lm_head": jnp.asarray(sd["lm_head.weight"], dtype=dtype),
        "pre": pre,
        "loop": loop,
        "post": post,
    }


# --------------------------------------------------------------------------- primitives
def _rms_zc(x, w, eps):
    """Zero-centered RMSNorm (Qwen3_5RMSNorm): fp32 normalize, scale by (1 + w), cast back."""
    xf = x.astype(F32)
    y = xf * lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (y * (1.0 + w.astype(F32))).astype(x.dtype)


def _rms_gated(core, z, w, eps):
    """Qwen3_5RMSNormGated: normalize, multiply by weight, gate with SiLU(z), cast back."""
    cdt = core.dtype
    xf = core.astype(F32)
    xn = xf * lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    y = w.astype(cdt) * xn.astype(cdt)
    y = y.astype(F32) * jax.nn.silu(z.astype(F32))
    return y.astype(cdt)


def _rope_tables(T: int, cfg: ModelCfg, dtype):
    inv = 1.0 / (cfg.rope_theta ** (jnp.arange(0, cfg.rot_dim, 2, dtype=F32) / cfg.rot_dim))
    freqs = jnp.arange(T, dtype=F32)[:, None] * inv[None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _apply_rope(x, cos, sin, rot):
    """Partial half-split RoPE on the first `rot` dims. x: [B,T,N,D]; cos/sin: [T, rot]."""
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    xr, xp = x[..., :rot], x[..., rot:]
    h = rot // 2
    rh = jnp.concatenate([-xr[..., h:], xr[..., :h]], axis=-1)
    return jnp.concatenate([xr * c + rh * s, xp], axis=-1)


def _mlp(x, p):
    return (jax.nn.silu(x @ p["gate_proj"].T) * (x @ p["up_proj"].T)) @ p["down_proj"].T


def _sinkhorn(logits, iters):
    """Log-space Sinkhorn over the last two axes; returns exp of the normalized logits (fp32)."""

    def body(_, L):
        L = L - jax.scipy.special.logsumexp(L, axis=-1, keepdims=True)
        L = L - jax.scipy.special.logsumexp(L, axis=-2, keepdims=True)
        return L

    return jnp.exp(lax.fori_loop(0, iters, body, logits))


# --------------------------------------------------------------------------- Gated DeltaNet
def _l2norm(x):
    """l2norm with eps added to the sum of squares (FLA / transformers convention)."""
    return x * lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-6)


def _causal_conv_silu(x, w, K):
    """Causal depthwise conv1d (left pad K-1, cross-correlation) then SiLU.

    x: [B,T,C]; w: [C,1,K] (torch layout). Output t uses x[t-K+1..t]; w[:, K-1] hits x[t].
    """
    B, T, C = x.shape
    xp = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0)))
    wk = w[:, 0, :]
    out = sum(xp[:, j:j + T, :] * wk[:, j][None, None, :] for j in range(K))
    return jax.nn.silu(out)


def gated_delta_chunked(q, k, v, g, beta, chunk: int = 64):
    """Chunked gated delta rule (matches torch_chunk_gated_delta_rule), float32.

    q, k: [B,T,H,Dk] (already head-repeated), v: [B,T,H,Dv], g (log decay, <= 0) and beta: [B,T,H].
    q and k are L2-normalized inside; q is scaled by Dk**-0.5. Returns [B,T,H,Dv] float32.
    """
    q, k, v, g, beta = (x.astype(F32) for x in (q, k, v, g, beta))
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    C = chunk
    q = _l2norm(q) * (Dk ** -0.5)
    k = _l2norm(k)
    q, k, v = (jnp.swapaxes(x, 1, 2) for x in (q, k, v))  # [B,H,T,D]
    g, beta = (jnp.swapaxes(x, 1, 2) for x in (g, beta))  # [B,H,T]

    pad = (-T) % C
    if pad:
        q, k, v = (jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0))) for x in (q, k, v))
        g, beta = (jnp.pad(x, ((0, 0), (0, 0), (0, pad))) for x in (g, beta))
    N = (T + pad) // C

    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    chunked = lambda x: x.reshape(B, H, N, C, x.shape[-1])
    q, k, v_beta, k_beta = (chunked(x) for x in (q, k, v_beta, k_beta))
    g = g.reshape(B, H, N, C)

    cum = jnp.cumsum(g, axis=-1)  # within-chunk cumulative log decay
    lower = jnp.tril(jnp.ones((C, C), dtype=bool))  # includes diagonal
    diff = cum[..., :, None] - cum[..., None, :]  # [..., i, j] = cum_i - cum_j
    decay = jnp.where(lower, jnp.exp(jnp.where(lower, diff, 0.0)), 0.0)

    kk = jnp.matmul(k_beta, jnp.swapaxes(k, -1, -2), precision=HIGHEST)
    ut = kk * decay  # UT system; its strict lower part is what the solve uses
    intra = jnp.matmul(q, jnp.swapaxes(k, -1, -2), precision=HIGHEST) * decay
    A = jnp.tril(ut, -1) + jnp.eye(C, dtype=F32)  # unit lower triangular
    rhs = jnp.concatenate([v_beta, k_beta * jnp.exp(cum)[..., None]], axis=-1)
    sol = lax.linalg.triangular_solve(A, rhs, left_side=True, lower=True, unit_diagonal=True)
    new_values, k_cumdecay = sol[..., :Dv], sol[..., Dv:]

    q_dec = q * jnp.exp(cum)[..., None]
    k_dec = k * jnp.exp(cum[..., -1:] - cum)[..., None]
    chunk_decay = jnp.exp(cum[..., -1])  # [B,H,N]

    def step(S, xs):
        qd, kd, intra_c, nv_c, kcd, cdec = xs
        v_new = nv_c - jnp.matmul(kcd, S, precision=HIGHEST)
        out = jnp.matmul(qd, S, precision=HIGHEST) + jnp.matmul(intra_c, v_new, precision=HIGHEST)
        S = S * cdec[..., None, None] + jnp.matmul(jnp.swapaxes(kd, -1, -2), v_new, precision=HIGHEST)
        return S, out

    xs = tuple(jnp.moveaxis(x, 2, 0) for x in (q_dec, k_dec, intra, new_values, k_cumdecay, chunk_decay))
    S0 = jnp.zeros((B, H, Dk, Dv), dtype=F32)
    _, outs = lax.scan(step, S0, xs)  # outs: [N,B,H,C,Dv]
    out = jnp.moveaxis(outs, 0, 2).reshape(B, H, N * C, Dv)[:, :, :T]
    return jnp.swapaxes(out, 1, 2)


def gated_delta_recurrent(q, k, v, g, beta):
    """Naive per-token gated delta rule (reference for tests only). Same signature as the chunked one."""
    q, k, v, g, beta = (x.astype(F32) for x in (q, k, v, g, beta))
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    q = _l2norm(q) * (Dk ** -0.5)
    k = _l2norm(k)
    tm = lambda x: jnp.moveaxis(x, 1, 0)  # [T,B,H,...]
    q, k, v, g, beta = (tm(x) for x in (q, k, v, g, beta))

    def step(S, xs):
        qt, kt, vt, gt, bt = xs
        S = S * jnp.exp(gt)[..., None, None]
        kv = jnp.einsum("bhkv,bhk->bhv", S, kt, precision=HIGHEST)
        delta = (vt - kv) * bt[..., None]
        S = S + kt[..., :, None] * delta[..., None, :]
        o = jnp.einsum("bhkv,bhk->bhv", S, qt, precision=HIGHEST)
        return S, o

    S0 = jnp.zeros((B, H, Dk, Dv), dtype=F32)
    _, outs = lax.scan(step, S0, (q, k, v, g, beta))
    return jnp.moveaxis(outs, 0, 1)  # [B,T,H,Dv]


def _deltanet(x, p, cfg: ModelCfg):
    cdt = x.dtype
    B, T, _ = x.shape
    nk, nv, dk, dv = cfg.lin_nk, cfg.lin_nv, cfg.lin_dk, cfg.lin_dv
    key_dim, val_dim = nk * dk, nv * dv

    qkv = x @ p["in_proj_qkv"].T
    z = (x @ p["in_proj_z"].T).reshape(B, T, nv, dv)
    b = x @ p["in_proj_b"].T
    a = x @ p["in_proj_a"].T

    mixed = _causal_conv_silu(qkv, p["conv1d"], cfg.conv_k)
    q = mixed[..., :key_dim].reshape(B, T, nk, dk)
    k = mixed[..., key_dim:2 * key_dim].reshape(B, T, nk, dk)
    v = mixed[..., 2 * key_dim:].reshape(B, T, nv, dv)

    beta = jax.nn.sigmoid(b).astype(F32)
    g = -jnp.exp(p["A_log"].astype(F32)) * jax.nn.softplus(a.astype(F32) + p["dt_bias"].astype(F32))
    rep = nv // nk
    q = jnp.repeat(q, rep, axis=2)  # repeat_interleave: v-head h uses k-head h // rep
    k = jnp.repeat(k, rep, axis=2)

    core = gated_delta_chunked(q, k, v, g, beta).astype(cdt)
    core = _rms_gated(core, z, p["norm"], cfg.eps)
    return core.reshape(B, T, val_dim) @ p["out_proj"].T


# --------------------------------------------------------------------------- gated full attention
def _full_attn(x, p, cfg: ModelCfg, cos, sin, sp, kv_fn, base_c):
    """Gated full attention. Returns (output, c) where c is the kv_fn side output (student only)."""
    B, T, _ = x.shape
    nh, G, hd = cfg.n_heads, cfg.n_kv, cfg.head_dim

    qg = (x @ p["q_proj"].T).reshape(B, T, nh, 2 * hd)
    query, gate = qg[..., :hd], qg[..., hd:]  # per-head interleaved query / gate
    gate = gate.reshape(B, T, nh * hd)
    query = _rms_zc(query, p["q_norm"], cfg.eps)

    c = None
    if sp is None:
        kraw = x @ p["k_proj"].T
        k = _rms_zc(kraw.reshape(B, T, G, hd), p["k_norm"], cfg.eps)
        v = (x @ p["v_proj"].T).reshape(B, T, G, hd)
    else:
        k, v, c = kv_fn(x, sp, p["k_norm"], base_c)
        k = k.reshape(B, T, G, hd)
        v = v.reshape(B, T, G, hd)

    query = _apply_rope(query, cos, sin, cfg.rot_dim)
    k = _apply_rope(k, cos, sin, cfg.rot_dim)

    rep = nh // G
    qh = query.reshape(B, T, G, rep, hd)  # query head h = g * rep + r uses kv head g
    scores = jnp.einsum("btgrd,bsgd->bgrts", qh, k, preferred_element_type=F32) * (hd ** -0.5)
    causal = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores = jnp.where(causal, scores, -jnp.inf)
    probs = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    o = jnp.einsum("bgrts,bsgd->btgrd", probs, v).reshape(B, T, nh * hd)
    o = o * jax.nn.sigmoid(gate)
    return o @ p["o_proj"].T, c


# --------------------------------------------------------------------------- layers
def _mix(x, p, kind, cfg, cos, sin, sp, kv_fn, base_c):
    if kind == LINEAR:
        return _deltanet(x, p, cfg), None
    return _full_attn(x, p, cfg, cos, sin, sp, kv_fn, base_c)


def _plain_layer(p, h, kind, cfg, cos, sin, sp=None, kv_fn=None, base_c=None):
    a, c = _mix(_rms_zc(h, p["input_layernorm"], cfg.eps), p, kind, cfg, cos, sin, sp, kv_fn, base_c)
    h = h + a
    h = h + _mlp(_rms_zc(h, p["post_attention_layernorm"], cfg.eps), p)
    return h, c


def _router(w, b, R, cfg: ModelCfg):
    """MHCRouter.forward: returns (branch [B,T,H], post [B,T,S] f32, mix [B,T,S,S] f32)."""
    B, T, S, H = R.shape
    flat = R.reshape(B, T, S * H).astype(F32)
    n = flat * lax.rsqrt(jnp.mean(flat * flat, axis=-1, keepdims=True) + 1e-6)
    logits = jnp.matmul(n, w.T, precision=HIGHEST) + b
    pre = jax.nn.sigmoid(logits[..., :S])
    post = 2.0 * jax.nn.sigmoid(logits[..., S:2 * S])
    mix = _sinkhorn(logits[..., 2 * S:].reshape(B, T, S, S), cfg.sinkhorn_iters)
    branch = jnp.einsum("bts,btsh->bth", pre.astype(R.dtype), R)
    if cfg.pre_scale != 1.0:
        branch = branch * cfg.pre_scale
    return branch, post, mix


def _merge(R, branch, post, mix):
    """MHCRouter.merge: mix the streams, add the branch output scaled by post."""
    cdt = R.dtype
    mixed = jnp.einsum("btij,btjh->btih", mix.astype(cdt), R)
    upd = post.astype(cdt)[..., None] * branch[:, :, None, :]
    return mixed + upd


def _mhc_layer(p, R, kind, cfg, cos, sin, sp=None, kv_fn=None, base_c=None):
    """MHCDecoderLayer on residual streams R [B,T,S,H]. Returns (R, c)."""
    branch, post, mix = _router(p["mhc_attn_w"], p["mhc_attn_b"], R, cfg)
    x = _rms_zc(branch, p["input_layernorm"], cfg.eps)
    a, c = _mix(x, p, kind, cfg, cos, sin, sp, kv_fn, base_c)
    R = _merge(R, a, post, mix)
    branch, post, mix = _router(p["mhc_ffn_w"], p["mhc_ffn_b"], R, cfg)
    f = _mlp(_rms_zc(branch, p["post_attention_layernorm"], cfg.eps), p)
    R = _merge(R, f, post, mix)
    return R, c


# --------------------------------------------------------------------------- forward
def _identity(x):
    return x


def forward_hidden(params, ids, cfg: ModelCfg, *, student=None, kv_fn=None, remat: bool = False,
                   constrain: Callable[[Any], Any] = _identity):
    """Full-sequence forward. ids: [B,T] int. Returns final-normed hidden states [B,T,H].

    student: None (teacher path) or {"pre": sp3, "loop0": sp_stacked, "loop1": sp_stacked, "post": sp31}
             where each sp is a dict for the full-attention layer's kv_fn (leading axis 6 for loop).
    kv_fn:   kv_fn(x, sp, k_norm_w, base_c) -> (k [B,T,G,HD] normed pre-RoPE, v [B,T,G,HD], c).
             Required when student is given. Loop pass 0 gets base_c=None; later passes receive the
             previous pass's c for the same group.
    """
    _check_cfg(cfg)
    if student is not None and kv_fn is None:
        raise ValueError("student given without kv_fn")
    cdt = params["embed"].dtype
    T = ids.shape[1]
    cos, sin = _rope_tables(T, cfg, cdt)
    wrap = jax.checkpoint if remat else (lambda f: f)

    def plain(p, h, i, sp):
        kind = cfg.layer_types[i]
        fn = lambda p_, h_, sp_: _plain_layer(p_, h_, kind, cfg, cos, sin, sp_, kv_fn, None)[0]
        return constrain(wrap(fn)(p, h, sp))

    def enter(h):
        # Expand [B,T,H] to S identical residual streams [B,T,S,H].
        return jnp.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], cfg.streams, h.shape[2]))

    h = constrain(params["embed"][ids])
    for i, p in enumerate(params["pre"]):
        sp = student["pre"] if (student is not None and cfg.layer_types[i] == FULL) else None
        h = plain(p, h, i, sp)

    pattern = cfg.layer_types[cfg.loop_start:cfg.loop_start + GROUP]
    prev_c = None
    for pass_idx in range(cfg.loop_count):
        use_student = student is not None
        xs = {"p": params["loop"]}
        if use_student:
            xs["sp"] = student[f"loop{pass_idx}"]
            if prev_c is not None:
                xs["bc"] = prev_c
        R = constrain(enter(h))

        def body(R, xs, use_student=use_student):
            lp, sp, bc = xs["p"], xs.get("sp"), xs.get("bc")
            c_out = None
            for j in range(GROUP):
                kind = pattern[j]
                is_student = use_student and kind == FULL
                R, c = _mhc_layer(
                    lp[f"l{j}"], R, kind, cfg, cos, sin,
                    sp if is_student else None, kv_fn if is_student else None,
                    bc if is_student else None,
                )
                R = constrain(R)
                if c is not None:
                    c_out = c
            return R, c_out

        body = wrap(body)
        R, ys = lax.scan(body, R, xs)
        h = constrain(R.mean(axis=-2))  # leave: average streams after the last Loop layer
        prev_c = ys

    for k, p in enumerate(params["post"]):
        i = cfg.loop_end + k
        sp = student["post"] if (student is not None and cfg.layer_types[i] == FULL) else None
        h = plain(p, h, i, sp)

    return _rms_zc(h, params["norm"], cfg.eps)


def logits_fn(params, h):
    """Language-model head. Returns float32 logits [B,T,V]."""
    prec = HIGHEST if h.dtype == F32 else None
    return jnp.matmul(h, params["lm_head"].T, preferred_element_type=F32, precision=prec)
