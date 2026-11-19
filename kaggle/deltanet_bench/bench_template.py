# SenseNovaKV: fused Gated DeltaNet decode step vs the transformers-code reference (correctness + speed).
# Rendered by kaggle/render.py: the kernel and reference sources are embedded below.
import json, os, sys, time, types

import torch
from torch.utils.cpp_extension import load

CU_SRC = r'''__CU_SOURCE__'''
REF_SRC = r'''__REF_SOURCE__'''

assert torch.cuda.is_available(), "needs a GPU"
dev = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
print(f"device: {props.name}  sm_{props.major}{props.minor}  torch {torch.__version__}  cuda {torch.version.cuda}")
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{props.major}.{props.minor}"

work = "/kaggle/working/build" if os.path.isdir("/kaggle") else "./build"
os.makedirs(work, exist_ok=True)
with open(f"{work}/deltanet_fused.cu", "w") as f:
    f.write(CU_SRC)
t0 = time.time()
ext = load(name="deltanet_fused", sources=[f"{work}/deltanet_fused.cu"], build_directory=work,
           extra_cflags=["-O3", "-DDELTANET_TORCH_EXTENSION"],
           extra_cuda_cflags=["-O3", "-DDELTANET_TORCH_EXTENSION", "-lineinfo"], verbose=False)
print(f"compiled in {time.time() - t0:.1f}s")

dn = types.ModuleType("deltanet_ref")
sys.modules["deltanet_ref"] = dn
exec(REF_SRC, dn.__dict__)
W = dn.make_weights(0, device=dev)
NV, DV, DK, CONV = dn.NUM_V_HEADS, dn.HEAD_V_DIM, dn.HEAD_K_DIM, dn.CONV_DIM
STATE_BYTES = NV * DK * DV * 4          # fp32 recurrent state per sequence: 2 MiB
results = {"device": props.name, "torch": torch.__version__, "correctness": [], "speed": []}


def ordered(x):
    i = x.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -32768 - i, i)  # monotone map of bf16 bit patterns


def ulp_dist(a, b):
    return (ordered(a) - ordered(b)).abs()


def kern(inp, st):
    return ext.deltanet_decode(inp["qkv"], inp["z"], inp["b"], inp["a"], st["conv"], st["rec"],
                               W["A_log"], W["dt_bias"], W["conv_w"], W["norm_w"])


def ref(inp, st):
    return dn.deltanet_decode_step(inp["qkv"], inp["z"], inp["b"], inp["a"], st["conv"], st["rec"],
                                   W["A_log"], W["dt_bias"], W["conv_w"], W["norm_w"])


def clone_state(st):
    return {"conv": st["conv"].clone(), "rec": st["rec"].clone()}


# ---------------- correctness: 64 carried decode steps per T ----------------
all_ok = True
for T in (1, 4, 16):
    gen = torch.Generator().manual_seed(1000 + T)
    conv0, rec0 = dn.make_state(T, gen, device=dev)
    st_r = {"conv": conv0.clone(), "rec": rec0.clone()}
    st_k = {"conv": conv0.clone(), "rec": rec0.clone()}
    worst = {"max_abs_over_outmax": 0.0, "rel_l2": 0.0, "max_ulp": 0, "min_frac_ulp_le1": 1.0, "outmax": 0.0}
    trace = []
    for step in range(64):
        inp = dn.make_step_inputs(T, gen, device=dev)
        o_r = ref(inp, st_r)
        o_k = kern(inp, st_k)
        d = (o_k.float() - o_r.float()).abs()
        scale = o_r.float().abs().max()
        u = ulp_dist(o_k, o_r)
        worst["max_abs_over_outmax"] = max(worst["max_abs_over_outmax"], float(d.max() / scale))
        worst["outmax"] = max(worst["outmax"], float(scale))
        worst["rel_l2"] = max(worst["rel_l2"], float(d.norm() / o_r.float().norm()))
        worst["max_ulp"] = max(worst["max_ulp"], int(u.max()))
        worst["min_frac_ulp_le1"] = min(worst["min_frac_ulp_le1"], float((u <= 1).float().mean()))
        if step in (0, 1, 7, 31, 63):
            trace.append({"step": step, "max_abs": float(d.max()), "max_ulp": int(u.max())})
    conv_diff = float((st_k["conv"].float() - st_r["conv"].float()).abs().max())
    rs = st_r["rec"].abs().max()
    state_rel = float((st_k["rec"] - st_r["rec"]).abs().max() / rs)
    state_rel_l2 = float((st_k["rec"] - st_r["rec"]).norm() / st_r["rec"].norm())
    ok = (worst["max_abs_over_outmax"] <= 2 ** -7 and worst["min_frac_ulp_le1"] >= 0.99
          and state_rel <= 1e-4 and conv_diff == 0.0)
    row = {"T": T, "steps": 64, **worst, "final_state_max_abs_over_max": state_rel,
           "final_state_rel_l2": state_rel_l2, "final_state_max_abs": float((st_k["rec"] - st_r["rec"]).abs().max()),
           "conv_state_max_abs_diff": conv_diff, "trace": trace, "ok": bool(ok),
           "bf16_ulp_at_outmax": float(2 ** -7 * worst["outmax"])}
    all_ok &= ok
    results["correctness"].append(row)
    print(("PASS" if ok else "FAIL"), json.dumps(row))
results["all_correct"] = bool(all_ok)


# ---------------- speed ----------------
def time_fn(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters  # microseconds per call


def time_graph(fn, reps, replays=40):
    """GPU time per call: `reps` calls captured in ONE CUDA graph, replayed `replays` times. Host launch
    overhead is outside the number, the way the engine runs a whole decode step."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    return time_fn(g.replay, iters=replays, warmup=3) / reps


def clocks():
    import subprocess
    try:
        q = "clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active"
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"n/a ({exc})"


# copy bandwidth (read + write) of a large buffer: the roofline the state traffic is compared with
buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
dst = torch.empty_like(buf)
copy_us = time_fn(lambda: dst.copy_(buf), iters=20, warmup=3)
peak_gbs = 2 * buf.numel() / (copy_us * 1e-6) / 1e9
results["copy_bw_gbs"] = peak_gbs
print(f"device copy bandwidth: {peak_gbs:.0f} GB/s  clocks: {clocks()}")
del buf, dst

for T in (1, 4, 16):
    gen = torch.Generator().manual_seed(7 + T)
    inp = dn.make_step_inputs(T, gen, device=dev)
    conv0, rec0 = dn.make_state(T, gen, device=dev)
    st_r = {"conv": conv0.clone(), "rec": rec0.clone()}
    st_k = {"conv": conv0.clone(), "rec": rec0.clone()}
    row = {"T": T, "clocks_before": clocks()}
    row["ref_eager_us"] = time_fn(lambda: ref(inp, st_r), iters=50, warmup=5)
    row["ref_graph_us"] = time_graph(lambda: ref(inp, st_r), reps=20, replays=20)
    row["kern_eager_us"] = time_fn(lambda: kern(inp, st_k), iters=200)
    row["kern_graph_us"] = time_graph(lambda: kern(inp, st_k), reps=100)
    state_traffic = T * 2 * STATE_BYTES                       # read + write, fp32 state
    other_per_seq = CONV * 4 * 2 * 2 + CONV * 2 + 4096 * 2 + 2 * NV * 2 + 4096 * 2  # conv state rw, qkv, z, a/b, out
    all_traffic = state_traffic + T * other_per_seq
    row["state_bytes_per_call"] = state_traffic
    row["state_bw_gbs"] = state_traffic / (row["kern_graph_us"] * 1e-6) / 1e9
    row["state_bw_frac_of_copy"] = row["state_bw_gbs"] / peak_gbs
    row["all_bw_gbs"] = all_traffic / (row["kern_graph_us"] * 1e-6) / 1e9
    row["kern_us_per_seq_step"] = row["kern_graph_us"] / T
    row["ref_graph_us_per_seq_step"] = row["ref_graph_us"] / T
    row["speedup_vs_ref_graph"] = row["ref_graph_us"] / row["kern_graph_us"]
    row["speedup_vs_ref_eager"] = row["ref_eager_us"] / row["kern_graph_us"]
    row["clocks_after"] = clocks()
    results["speed"].append(row)
    print(json.dumps(row))

# 42 DeltaNet visits per decoded token (SenseNova loop schedule), T=1 kernel time
for row in results["speed"]:
    if row["T"] == 1:
        results["per_token_ms_42_visits"] = {
            "ref_eager": row["ref_eager_us"] * 42 / 1000,
            "ref_graph": row["ref_graph_us"] * 42 / 1000,
            "kernel_graph": row["kern_graph_us"] * 42 / 1000}
        print("per decoded token, 42 DeltaNet visits (ms):", results["per_token_ms_42_visits"])

out = "/kaggle/working/results.json" if os.path.isdir("/kaggle/working") else "results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print("ALL CORRECT" if all_ok else "CORRECTNESS FAILURES — see FAIL rows")
if not all_ok:
    raise RuntimeError("deltanet correctness check failed")
