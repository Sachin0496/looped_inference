# SenseNovaKV: fused mHC kernel vs PyTorch reference (correctness + speed).
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
with open(f"{work}/mhc_fused.cu", "w") as f:
    f.write(CU_SRC)
t0 = time.time()
ext = load(name="mhc_fused", sources=[f"{work}/mhc_fused.cu"], build_directory=work,
           extra_cflags=["-O3", "-DMHC_TORCH_EXTENSION"],
           extra_cuda_cflags=["-O3", "-DMHC_TORCH_EXTENSION", "-lineinfo"], verbose=False)
print(f"compiled in {time.time() - t0:.1f}s")

ref = types.ModuleType("mhc_ref")
sys.modules["mhc_ref"] = ref  # torch.compile needs an importable module
exec(REF_SRC, ref.__dict__)
COUNTERS = torch.zeros(1024, dtype=torch.int32, device=dev)

results = {"device": props.name, "torch": torch.__version__, "correctness": [], "speed": []}


def ordered(x):
    i = x.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -32768 - i, i)  # monotone map of bf16 bit patterns


def ulp_stats(a, b):
    d = (ordered(a) - ordered(b)).abs()
    return {"max_ulp": int(d.max()), "frac_exact": float((d == 0).float().mean()),
            "frac_le1": float((d <= 1).float().mean())}


def fused(inp, merge, pre_scale=1.0, split=False):
    res = inp["residual"].clone()
    if split:
        a = (inp["branch_prev"], inp["post_prev"], inp["mix_prev"]) if merge else (None, None, None)
        return ext.mhc_step_split(res, *a, inp["weight"], inp["bias"], inp["norm_w_bf16"], COUNTERS, 20, pre_scale)
    if merge:
        return ext.mhc_step(res, inp["branch_prev"], inp["post_prev"], inp["mix_prev"], inp["weight"],
                            inp["bias"], inp["norm_w_bf16"], 20, pre_scale)
    return ext.mhc_step(res, None, None, None, inp["weight"], inp["bias"], inp["norm_w_bf16"], 20, pre_scale)


def reference(inp, merge, pre_scale=1.0):
    kw = dict(branch_prev=inp["branch_prev"], post_prev=inp["post_prev"], mix_prev=inp["mix_prev"]) if merge else {}
    return ref.mhc_step(inp["residual"], inp["weight"], inp["bias"], inp["norm_w_bf16"], iterations=20,
                        pre_scale=pre_scale, **kw)


def get_inputs(T, seed=0):
    inp = ref.make_inputs(T, device=dev, seed=seed)
    inp["norm_w_bf16"] = inp["norm_w"].to(torch.bfloat16)  # checkpoint norm weights are bf16
    return inp


# ---------------- correctness ----------------
all_ok = True
for T in (1, 4, 16, 128):
    for merge in (False, True):
        for pre_scale, split in ((1.0, False), (0.5, False), (1.0, True), (0.5, True)):
            inp = get_inputs(T, seed=T * 7 + merge)
            r_res, r_out, r_post, r_mix = reference(inp, merge, pre_scale)
            f_res, f_out, f_post, f_mix = fused(inp, merge, pre_scale, split)
            row = {"T": T, "merge": merge, "pre_scale": pre_scale, "path": "split" if split else "block",
                   "out": ulp_stats(f_out, r_out), "residual": ulp_stats(f_res, r_res),
                   "post_maxabs": float((f_post - r_post).abs().max()),
                   "mix_maxabs": float((f_mix - r_mix).abs().max())}
            ok = (row["out"]["max_ulp"] <= 2 and row["out"]["frac_le1"] >= 0.999 and row["residual"]["max_ulp"] <= 1
                  and row["post_maxabs"] < 1e-5 and row["mix_maxabs"] < 1e-5)
            row["ok"] = bool(ok)
            all_ok &= ok
            results["correctness"].append(row)
            print(("PASS" if ok else "FAIL"), json.dumps(row))
results["all_correct"] = bool(all_ok)


# ---------------- speed ----------------
def time_fn(fn, iters=300, warmup=30):
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


def time_graph(fn, reps=50, replays=40):
    """GPU time per call: `reps` calls captured in ONE graph, replayed `replays` times.
    Keeps host launch cost out of the number, the way the engine runs a whole decode step."""
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


def count_kernels(fn):
    from torch.profiler import ProfilerActivity, profile
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return sum(1 for ev in p.events() if ev.device_type == torch.autograd.DeviceType.CUDA)


def clocks():
    import subprocess
    try:
        q = "clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active"
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"n/a ({exc})"


# measured copy bandwidth for a roofline reference
buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
dst = torch.empty_like(buf)
copy_us = time_fn(lambda: dst.copy_(buf), iters=20, warmup=3)
peak_gbs = 2 * buf.numel() / (copy_us * 1e-6) / 1e9
results["copy_bw_gbs"] = peak_gbs
print(f"device copy bandwidth: {peak_gbs:.0f} GB/s  clocks: {clocks()}")
del buf, dst


def fused_inplace(inp, res, split):
    """No clone: the merge writes into `res` in place, as in the engine."""
    a = (inp["branch_prev"], inp["post_prev"], inp["mix_prev"])
    if split:
        return ext.mhc_step_split(res, *a, inp["weight"], inp["bias"], inp["norm_w_bf16"], COUNTERS, 20, 1.0)
    return ext.mhc_step(res, *a, inp["weight"], inp["bias"], inp["norm_w_bf16"], 20, 1.0)


for T in (1, 4, 16, 512):
    inp = get_inputs(T)
    res = inp["residual"].clone()
    ref_fn = lambda: reference(inp, True)
    row = {"T": T, "clocks_before": clocks()}
    row["ref_kernels_per_call"] = count_kernels(ref_fn)
    row["ref_eager_us"] = time_fn(ref_fn, iters=100, warmup=10)
    try:
        row["ref_graph_us"] = time_graph(ref_fn, reps=5, replays=20)
    except Exception as exc:  # noqa: BLE001
        row["ref_graph_us"] = None; print("graph capture of reference failed:", exc)
    for name, split in (("block", False), ("split", True)):
        fn = lambda: fused_inplace(inp, res, split)
        row[f"{name}_kernels_per_call"] = count_kernels(fn)
        row[f"{name}_eager_us"] = time_fn(fn)
        row[f"{name}_graph_us"] = time_graph(fn)
    row["clocks_after"] = clocks()
    # bytes: residual read+write, branch read, out write (bf16) + router weights (fp32, once per token block)
    bytes_moved = T * (2 * 2 * 4096 * 2 + 4096 * 2 + 4096 * 2) + 8 * 8192 * 4 * min(T, props.multi_processor_count)
    best = min(row["block_graph_us"], row["split_graph_us"])
    row["best_fused_us"] = best
    row["best_bw_frac_of_copy"] = bytes_moved / (best * 1e-6) / 1e9 / peak_gbs
    row["speedup_vs_eager"] = row["ref_eager_us"] / best
    row["speedup_vs_ref_graph"] = (row["ref_graph_us"] / best) if row["ref_graph_us"] else None
    results["speed"].append(row)
    print(json.dumps(row))

# torch.compile last, so its worker processes cannot disturb the other timings
try:
    compiled = torch.compile(ref.mhc_step, dynamic=False)
    for row in results["speed"]:
        inp = get_inputs(row["T"])
        kw = dict(branch_prev=inp["branch_prev"], post_prev=inp["post_prev"], mix_prev=inp["mix_prev"])
        cfn = lambda: compiled(inp["residual"], inp["weight"], inp["bias"], inp["norm_w_bf16"], **kw)
        row["compiled_eager_us"] = time_fn(cfn, iters=100, warmup=10)
        try:
            row["compiled_graph_us"] = time_graph(cfn, reps=10, replays=20)
        except Exception as exc:  # noqa: BLE001
            row["compiled_graph_us"] = None; print("graph capture of compiled ref failed:", exc)
        print("compiled", row["T"], row["compiled_eager_us"], row["compiled_graph_us"])
except Exception as exc:  # noqa: BLE001
    print("torch.compile failed:", exc)

# mHC covers layers 4..27, each run twice: 48 executions x 2 steps = 96 per decoded token
CALLS = 96
for row in results["speed"]:
    if row["T"] == 1:
        results["per_token_ms"] = {k: (row.get(f) or 0) * CALLS / 1000 for k, f in (
            ("ref_eager", "ref_eager_us"), ("ref_graph", "ref_graph_us"), ("compiled_graph", "compiled_graph_us"),
            ("fused_best", "best_fused_us"))}
        print("per decoded token, 96 mHC steps (ms):", results["per_token_ms"])

out = "/kaggle/working/results.json" if os.path.isdir("/kaggle/working") else "results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print("ALL CORRECT" if all_ok else "CORRECTNESS FAILURES — see FAIL rows")
