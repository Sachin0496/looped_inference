"""CPU checks for the decode engine plumbing (engine/runtime/snkv_engine.py).

1. The 4-bit packer matches engine/ref/w4_ref.py byte for byte; row dequant matches w4_ref.dequant.
2. The execution plan: 56 visits, 42 linear / 14 full, 48 mHC visits (physical 4..27 twice).
3. Whole-model check on a TINY random SenseNova model (real layer count, plan and layer types; small dims):
   HF forward with the DEQUANTIZED packed weights (so no quantization error in the comparison) against the engine
   running the reference kernels in fp32. Prefill snapshot -> teacher-forced decode steps; logits must agree.
4. Same, with the mHC/DeltaNet side in bf16 (the GPU policy's rounding points), looser tolerance.

Run: python3 tests/test_engine_cpu.py   (needs torch, transformers and the SenseNova-Loop source checkout;
SNKV_SRC overrides the default checkout path, SNKV_CONFIG the config.json path).
"""
import copy
import glob
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine" / "runtime"))
sys.path.insert(0, str(ROOT / "engine" / "ref"))
SRC = os.environ.get("SNKV_SRC", "/private/tmp/claude-501/-Users-sachin-Development-3DCFD/01a648ca-e0c2-4807-aaab-9b166a7972c1/scratchpad/src_dl/SenseNova-Loop/src")
sys.path.insert(0, SRC)

import torch
import snkv_engine as E
import w4_ref

torch.set_num_threads(4)


def check_packer():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(40, 128, generator=g) * 0.05
    qw, meta = E.pack_w4(W, rows_per_chunk=16)
    qw_ref, meta_ref = w4_ref.pack(W, rows_per_chunk=16)
    assert torch.equal(qw, qw_ref), "packed bytes differ from w4_ref.pack"
    assert torch.equal(meta, meta_ref), "meta differs from w4_ref.pack"
    assert torch.allclose(E.dequant_rows(qw, meta, 128), w4_ref.dequant(qw, meta, 128), atol=0, rtol=0)
    err = (E.dequant_rows(qw, meta, 128) - W).abs().max().item()
    print(f"[packer] byte-identical to w4_ref; max |dequant - W| = {err:.4f} (step ~{float(meta[..., 0].float().max()):.4f})")


def check_plan():
    plan = E.build_plan(32, {"start": 4, "end": 28, "eval_count": 2, "counts": [2]})
    assert len(plan) == 56 and plan[:4] == (0, 1, 2, 3) and plan[4:28] == tuple(range(4, 28))
    assert plan[28:52] == tuple(range(4, 28)) and plan[52:] == (28, 29, 30, 31)
    full_layers = {3, 7, 11, 15, 19, 23, 27, 31}
    layer_types = ["full_attention" if i in full_layers else "linear_attention" for i in range(32)]
    rows = E.visit_table(layer_types, plan, (4, 28))
    n_full = sum(r[2] == "full" for r in rows)
    n_mhc = sum(r[3] for r in rows)
    assert n_full == 14 and len(rows) - n_full == 42 and n_mhc == 48, (n_full, n_mhc)
    print(f"[plan] 56 visits: {len(rows) - n_full} linear, {n_full} full, {n_mhc} mHC")


def model_config():
    cands = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--sensenova--SenseNova-Loop-9B-Preview/snapshots/*/config.json"))
    assert cands, "SenseNova-Loop-9B-Preview config.json not in the HF cache"
    raw = json.load(open(cands[0]))
    t = raw["text_config"]
    t.update(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
             linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
             vocab_size=512, mtp_num_hidden_layers=0, pad_token_id=0, eos_token_id=1)
    raw.update(pad_token_id=0, eos_token_id=1, image_token_id=500, video_token_id=501,
               vision_start_token_id=502, vision_end_token_id=503)
    raw["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    raw.pop("auto_map", None)
    return raw


def tiny_model(seed=0):
    from transformers import AutoModelForCausalLM
    from sensenova_loop.models import register
    from sensenova_loop.models.modeling_sensenova_loop import SenseNovaLoopConfig
    register()
    cfg = SenseNovaLoopConfig(**model_config())
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.08)
    return model.eval()


def linear_modules(model):
    """(host key path, module) for every tensor the engine quantizes, in export_host naming."""
    tm = model.model.language_model
    out = [("embed", tm.embed_tokens), ("lm_head", model.lm_head)]
    for i, layer in enumerate(tm.layers):
        out += [((i, "gate"), layer.mlp.gate_proj), ((i, "up"), layer.mlp.up_proj), ((i, "down"), layer.mlp.down_proj)]
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            out += [((i, "in_qkv"), la.in_proj_qkv), ((i, "in_z"), la.in_proj_z), ((i, "in_b"), la.in_proj_b),
                    ((i, "in_a"), la.in_proj_a), ((i, "out"), la.out_proj)]
        else:
            sa = layer.self_attn
            out += [((i, "q"), sa.q_proj), ((i, "k"), sa.k_proj), ((i, "v"), sa.v_proj), ((i, "o"), sa.o_proj)]
    return out


def dequant_copy(model, host):
    """Deep copy of the model whose quantized linears hold the dequantized packed values."""
    m = copy.deepcopy(model)
    for key, mod in linear_modules(m):
        W = host["embed"] if key == "embed" else host["lm_head"] if key == "lm_head" else host["layers"][key[0]][key[1]]
        Wd = E.dequant_rows(W["qw"], W["meta"], W["K"]).reshape(W["N"], W["K"])
        with torch.no_grad():
            mod.weight.copy_(Wd.to(mod.weight.dtype))
    return m


def run_case(name, policy, tol, min_top1, steps=8, prompt_len=24, seed=0):
    model = tiny_model(seed)
    host = E.export_host(model)
    dims = host["dims"]
    deq = dequant_copy(model, host)
    g = torch.Generator().manual_seed(seed + 1)
    prompt = torch.randint(2, 500, (1, prompt_len), generator=g)
    with torch.no_grad():
        out = deq(input_ids=prompt, use_cache=True)
    cache = out.past_key_values
    kinds = [r[2] for r in E.visit_table(dims["layer_types"], dims["plan"], dims["mhc"])]
    snap = E.snapshot_cache(cache, kinds)
    assert snap["S"] == prompt_len
    t1 = int(out.logits[0, -1].argmax())
    fed = [t1] + [int(x) for x in torch.randint(2, 500, (steps - 1,), generator=g)]
    st = E.to_device(host, "cpu", policy)
    ops = E.Ops("ref", ref_dir=str(ROOT / "engine" / "ref"))
    eng = E.Engine(st, ops, max_len=prompt_len + steps + 4, policy=policy, device="cpu")
    eng.load_prefill(snap)
    worst, rows = 0.0, []
    for k, tok in enumerate(fed):
        with torch.no_grad():
            hf = deq(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True).logits[0, -1].float()
        ours = eng.step(tok).float()[0]
        scale = max(1.0, float(hf.abs().max()))
        rel = float((ours - hf).abs().max()) / scale
        same = int(ours.argmax()) == int(hf.argmax())
        worst = max(worst, rel)
        rows.append((k + 1, rel, same))
    print(f"[{name}] policy {policy}: worst rel max|dlogit| {worst:.2e} over {steps} steps; top1 agree "
          f"{sum(r[2] for r in rows)}/{steps}")
    assert worst < tol, f"{name}: engine logits deviate from HF (worst rel {worst:.3e} > {tol})"
    agree = sum(r[2] for r in rows)
    assert agree >= min_top1, f"{name}: top-1 agreement {agree}/{steps} < {min_top1}"
    return worst


def main():
    check_packer()
    check_plan()
    run_case("fp32 plumbing", E.REF_POLICY, tol=1e-4, min_top1=8)
    # bf16 rounding of the mHC / DeltaNet side. HF itself run in bf16 deviates from HF fp32 by 4-20% on this random
    # tiny model (measured with the same weights), so this case is a sanity bound, not an equality check.
    run_case("mhc bf16 side", E.Policy(torch.float32, torch.bfloat16), tol=0.25, min_top1=6)
    print("ALL CPU CHECKS PASSED")


if __name__ == "__main__":
    main()
