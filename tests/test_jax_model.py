"""CPU checks for tpu/jax_model.py against the PyTorch SenseNova-Loop reference on a tiny random model.

Run:  python tests/test_jax_model.py      (or pytest tests/test_jax_model.py)
Env:  SENSENOVA_SRC  path to the SenseNova-Loop checkout's src/ directory
      (default: the scratchpad clone used during development).
"""
import json
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

import jax
import jax.numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tpu"))
import jax_model as jm  # noqa: E402

SRC = os.environ.get(
    "SENSENOVA_SRC",
    "/private/tmp/claude-501/-Users-sachin-Development-3DCFD/01a648ca-e0c2-4807-aaab-9b166a7972c1/"
    "scratchpad/src_dl/SenseNova-Loop/src",
)
WEIGHTS = "sensenova/SenseNova-Loop-9B-Preview"
TOL_LOGITS = 2e-3
TOL_STUDENT = 1e-5
TOL_DELTA = 1e-4


# --------------------------------------------------------------------------- tiny torch model
def _build_tiny():
    """Same tiny shrink as kaggle/distill/distill.py (TINY branch), with randomized small params."""
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, SRC)
    from sensenova_loop.models import register
    from sensenova_loop.models.modeling_sensenova_loop import SenseNovaLoopConfig
    from transformers import AutoModelForCausalLM

    register()
    raw = json.load(open(hf_hub_download(WEIGHTS, "config.json")))
    t = raw["text_config"]
    t.update(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
             linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
             vocab_size=512, mtp_num_hidden_layers=0, pad_token_id=0, eos_token_id=1)
    raw.update(pad_token_id=0, eos_token_id=1, image_token_id=500, video_token_id=501,
               vision_start_token_id=502, vision_end_token_id=503)
    raw["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    raw.pop("auto_map", None)
    cfg = SenseNovaLoopConfig(**raw)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    # Non-zero everything (norm offsets, A_log, dt_bias, conv, mHC routers) so no bug hides behind zeros.
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=gen))
    model.eval()
    return raw, model


_CACHE = {}


def _tiny():
    if "model" not in _CACHE:
        raw, model = _build_tiny()
        cfg = jm.ModelCfg.from_hf_config(raw)
        sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
              if k.startswith("model.language_model.") or k == "lm_head.weight"}
        params = jm.params_from_state_dict(sd, cfg, jnp.float32)
        _CACHE.update(raw=raw, model=model, cfg=cfg, params=params, sd=sd)
    return _CACHE


def _ids(vocab, shape=(2, 150), seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, shape, generator=g)


def _report(name, got, ref):
    diff = np.abs(got - ref)
    rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12)
    print(f"  {name}: max|diff|={diff.max():.3e}  rel_l2={rel:.3e}  max|ref|={np.abs(ref).max():.3e}")
    return float(diff.max())


# --------------------------------------------------------------------------- test 1
def test_chunked_vs_recurrent_delta_rule():
    print("test_chunked_vs_recurrent_delta_rule")
    key = jax.random.PRNGKey(0)
    kq, kk, kv, kg, kb = jax.random.split(key, 5)
    B, T, H, Dk, Dv = 2, 200, 3, 16, 24  # T not a multiple of 64
    q = jax.random.normal(kq, (B, T, H, Dk), jnp.float32)
    k = jax.random.normal(kk, (B, T, H, Dk), jnp.float32)
    v = jax.random.normal(kv, (B, T, H, Dv), jnp.float32)
    g = -jax.random.uniform(kg, (B, T, H), jnp.float32, 0.0, 1.0)  # log decay <= 0
    beta = jax.random.uniform(kb, (B, T, H), jnp.float32, 0.02, 0.98)
    o_c = np.asarray(jm.gated_delta_chunked(q, k, v, g, beta))
    o_r = np.asarray(jm.gated_delta_recurrent(q, k, v, g, beta))
    assert o_c.shape == o_r.shape == (B, T, H, Dv)
    d = _report("chunked vs recurrent", o_c, o_r)
    assert d < TOL_DELTA, d


# --------------------------------------------------------------------------- test 2
def test_logits_match_torch():
    print("test_logits_match_torch")
    c = _tiny()
    ids = _ids(c["cfg"].vocab)
    with torch.no_grad():
        ref = c["model"](input_ids=ids, use_cache=False).logits.float().numpy()
    h = jm.forward_hidden(c["params"], jnp.asarray(ids.numpy()), c["cfg"])
    got = np.asarray(jm.logits_fn(c["params"], h))
    assert got.shape == ref.shape, (got.shape, ref.shape)
    d = _report("logits (eager)", got, ref)
    assert d < TOL_LOGITS, d


# --------------------------------------------------------------------------- test 3
def test_logits_match_torch_remat_and_jit():
    print("test_logits_match_torch_remat_and_jit")
    c = _tiny()
    cfg, params = c["cfg"], c["params"]
    ids_t = _ids(cfg.vocab)
    ids = jnp.asarray(ids_t.numpy())
    with torch.no_grad():
        ref = c["model"](input_ids=ids_t, use_cache=False).logits.float().numpy()

    h_remat = jm.forward_hidden(params, ids, cfg, remat=True)
    got_remat = np.asarray(jm.logits_fn(params, h_remat))
    d1 = _report("logits (remat=True)", got_remat, ref)
    assert d1 < TOL_LOGITS, d1

    jit_fwd = jax.jit(lambda p, i: jm.logits_fn(p, jm.forward_hidden(p, i, cfg)))
    got_jit = np.asarray(jit_fwd(params, ids))
    d2 = _report("logits (jit)", got_jit, ref)
    assert d2 < TOL_LOGITS, d2

    jit_remat = jax.jit(lambda p, i: jm.logits_fn(p, jm.forward_hidden(p, i, cfg, remat=True)))
    got_jr = np.asarray(jit_remat(params, ids))
    d3 = _report("logits (jit + remat)", got_jr, ref)
    assert d3 < TOL_LOGITS, d3


# --------------------------------------------------------------------------- test 4
def _zc_norm(x, w, eps):
    xf = x.astype(jnp.float32)
    y = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (y * (1.0 + w.astype(jnp.float32))).astype(x.dtype)


def test_student_plumbing_and_grad():
    print("test_student_plumbing_and_grad")
    c = _tiny()
    cfg, params = c["cfg"], c["params"]
    ids = jnp.asarray(_ids(cfg.vocab).numpy())
    B, T = ids.shape
    G, HD = cfg.n_kv, cfg.head_dim
    seen = []  # base_c shapes seen by kv_fn, recorded at trace time

    def teacher_kv_fn(x, sp, k_norm_w, base_c):
        seen.append(None if base_c is None else tuple(base_c.shape))
        kraw = x @ sp["k_proj"].T
        k = _zc_norm(kraw.reshape(x.shape[0], x.shape[1], G, HD), k_norm_w, cfg.eps)
        v = (x @ sp["v_proj"].T).reshape(x.shape[0], x.shape[1], G, HD)
        return k, v, kraw

    def sp_of(layer):
        return {"k_proj": layer["k_proj"], "v_proj": layer["v_proj"]}

    loop3 = params["loop"]["l3"]  # full-attention layer inside each group, stacked over 6 groups
    student = {
        "pre": sp_of(params["pre"][3]),
        "loop0": sp_of(loop3),
        "loop1": sp_of(loop3),
        "post": sp_of(params["post"][-1]),
    }
    h_teacher = jm.forward_hidden(params, ids, cfg)
    logits_teacher = np.asarray(jm.logits_fn(params, h_teacher))

    seen.clear()
    h_student = jm.forward_hidden(params, ids, cfg, student=student, kv_fn=teacher_kv_fn)
    logits_student = np.asarray(jm.logits_fn(params, h_student))
    d = _report("student(teacher kv) vs teacher", logits_student, logits_teacher)
    assert d < TOL_STUDENT, d

    # pre layer 3, loop pass 0 (traced once), loop pass 1 (traced once), post layer 31
    print(f"  kv_fn base_c shapes seen at trace time: {seen}")
    assert seen[0] is None and seen[3] is None, seen
    assert seen[1] is None, seen
    assert seen[2] == (B, T, G * HD), seen
    # the c from pass 0 must also be the expected [6, B, T, G*HD] stack
    n_groups = (cfg.loop_end - cfg.loop_start) // jm.GROUP
    assert seen[2] == (B, T, G * HD)
    print(f"  pass-1 base_c per-group shape {seen[2]} (stacked over {n_groups} groups by scan)")

    # Differentiability of a student param.
    def loss(sp0):
        st = dict(student, loop0=sp0)
        h = jm.forward_hidden(params, ids, cfg, student=st, kv_fn=teacher_kv_fn)
        return jnp.sum(jm.logits_fn(params, h) ** 2) * 1e-6

    grads = jax.grad(loss)(student["loop0"])
    for name in ("k_proj", "v_proj"):
        gname = np.asarray(grads[name])
        assert gname.shape == np.asarray(student["loop0"][name]).shape
        assert np.all(np.isfinite(gname)), name
        norm = float(np.linalg.norm(gname))
        print(f"  d loss / d loop0.{name}: shape={gname.shape} l2={norm:.3e}")
        assert norm > 0, name


# --------------------------------------------------------------------------- runner
def _run_all():
    tests = [test_chunked_vs_recurrent_delta_rule, test_logits_match_torch,
             test_logits_match_torch_remat_and_jit, test_student_plumbing_and_grad]
    for t in tests:
        t()
        print("  PASS")
    print("ALL PASSED")


if __name__ == "__main__":
    _run_all()
