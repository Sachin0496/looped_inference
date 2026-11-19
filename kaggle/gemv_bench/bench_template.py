# SenseNovaKV: fused 4-bit weight GEMV. Correctness vs fp32 reference, then timing vs cuBLAS fp16
# and vs dequant-then-matmul. Rendered by kaggle/render.py: kernel and reference sources are inlined.
import json, math, os, sys, time, types

import torch
import torch.nn.functional as F
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
with open(f"{work}/w4_gemv.cu", "w") as f:
    f.write(CU_SRC)
t0 = time.time()
ext = load(name="w4_gemv", sources=[f"{work}/w4_gemv.cu"], build_directory=work,
           extra_cflags=["-O3", "-DW4_TORCH_EXTENSION"],
           extra_cuda_cflags=["-O3", "-DW4_TORCH_EXTENSION", "-lineinfo"], verbose=False)
print(f"compiled in {time.time() - t0:.1f}s")

ref = types.ModuleType("w4_ref")
sys.modules["w4_ref"] = ref
exec(REF_SRC, ref.__dict__)

results = {"device": props.name, "torch": torch.__version__, "correctness": [], "speed": []}
REL_TOL = 1e-3          # fp16 output rounding is 2^-11 of the value; allow 2x that plus fp32 noise
SHAPES = [(8192, 4096), (1024, 4096), (4096, 4096), (12288, 4096), (4096, 12288), (248320, 4096)]
TS = (1, 2, 4, 8)
CONFIGS = [(1, 2048), (2, 2048), (4, 2048), (1, 4096), (2, 4096), (4, 4096)]  # (rows per warp, K slab)
WEIGHT_RING_BYTES = 96 << 20  # rotate over copies so every call reads DRAM, not L2


def time_graph(fns, replays=20, warm=3):
    """Mean GPU time per call in us. All calls are captured in ONE CUDA graph and replayed,
    so host launch cost is excluded (as in the engine's whole-step graphs)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(replays):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / (replays * len(fns))


# measured copy bandwidth (read + write bytes / time), the same convention as the mhc bench
buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
dst = torch.empty_like(buf)
for _ in range(3):
    dst.copy_(buf)
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(20):
    dst.copy_(buf)
e1.record()
torch.cuda.synchronize()
copy_us = e0.elapsed_time(e1) * 1000 / 20
COPY_GBS = 2 * buf.numel() / (copy_us * 1e-6) / 1e9
results["copy_bw_gbs"] = COPY_GBS
print(f"device copy bandwidth: {COPY_GBS:.1f} GB/s")
del buf, dst


def packed_bytes(N, K):
    return N * (K // 2) + N * (K // 64) * 4


def ring_size(nbytes):
    return max(2, min(8, math.ceil(WEIGHT_RING_BYTES / nbytes)))


def ok_str(err):
    return "PASS" if err["rel"] <= REL_TOL else "FAIL"


all_ok = True
for (N, K) in SHAPES:
    R = ring_size(packed_bytes(N, K))
    print(f"\n=== N={N} K={K}  weight copies in ring: {R} ===", flush=True)
    packs = []
    for i in range(R):
        p, m = ref.make_weights(N, K, dev, seed=100 + i)
        packs.append((p, m))
    p0, m0 = packs[0]

    # ---------- correctness ----------
    for T in TS:
        g = torch.Generator(device=dev).manual_seed(T * 31 + N)
        x = (torch.randn(T, K, generator=g, device=dev) * 1.5).half()
        nw = (torch.randn(K, generator=g, device=dev) * 0.1).half()
        rs = (torch.randn(T, N, generator=g, device=dev) * 0.5).half()
        for norm, resid in ((False, False), (True, False), (False, True), (True, True)):
            yref = ref.gemv_ref(x, p0, m0, K, nw, rs if resid else None, norm)
            scale = yref.abs().max().item()
            for rpw, slab in CONFIGS:
                if N % (4 * rpw) != 0 or K % slab != 0:
                    continue
                y = ext.w4_gemv(x, p0, m0, nw if norm else None, rs if resid else None, 1e-6, rpw, slab)
                err = (y.float() - yref).abs()
                row = {"N": N, "K": K, "T": T, "norm": norm, "resid": resid, "rpw": rpw, "slab": slab,
                       "max_abs": err.max().item(), "max_abs_over_ymax": err.max().item() / scale,
                       "rel": err.max().item() / scale}
                row["status"] = ok_str(row)
                all_ok &= row["status"] == "PASS"
                results["correctness"].append(row)
                print(row["status"], json.dumps(row), flush=True)
            del yref

    # ---------- speed: ours, every config ----------
    rows = {}
    for T in TS:
        x = torch.randn(T, K, device=dev).half()
        nw = torch.randn(K, device=dev).half() * 0.1
        rs = torch.randn(T, N, device=dev).half()
        for norm, resid in ((False, False), (True, True)):
            for rpw, slab in CONFIGS:
                if N % (4 * rpw) != 0 or K % slab != 0:
                    continue
                fns = [(lambda i=i: ext.w4_gemv(x, packs[i][0], packs[i][1], nw if norm else None,
                                               rs if resid else None, 1e-6, rpw, slab)) for i in range(R)]
                us = time_graph(fns)
                gbs = packed_bytes(N, K) / (us * 1e-6) / 1e9
                row = {"method": "w4_gemv", "N": N, "K": K, "T": T, "norm": norm, "resid": resid,
                       "rpw": rpw, "slab": slab, "us": us, "gbs": gbs, "pct_copy": gbs / COPY_GBS}
                results["speed"].append(row)
                rows.setdefault((T, norm), []).append(row)
                print(json.dumps(row), flush=True)

    # ---------- baseline: dequant-then-matmul (torch ops, fp16) ----------
    for T in TS:
        x = torch.randn(T, K, device=dev).half()
        fns = [(lambda i=i: F.linear(x, ref.dequant_torch_fp16(packs[i][0], packs[i][1], K))) for i in range(R)]
        us = time_graph(fns, replays=5)
        gbs = packed_bytes(N, K) / (us * 1e-6) / 1e9
        row = {"method": "dequant_torch_then_matmul", "N": N, "K": K, "T": T, "us": us, "gbs": gbs,
               "pct_copy": gbs / COPY_GBS}
        results["speed"].append(row)
        print(json.dumps(row), flush=True)

    # ---------- baseline: cuBLAS fp16 on the full dequantized weight ----------
    w16s = [ref.dequant_torch_fp16(packs[i][0], packs[i][1], K) for i in range(R)]
    del packs
    torch.cuda.empty_cache()
    for T in TS:
        x = torch.randn(T, K, device=dev).half()
        fns = [(lambda i=i: F.linear(x, w16s[i])) for i in range(R)]
        us = time_graph(fns)
        nbytes = N * K * 2
        gbs = nbytes / (us * 1e-6) / 1e9
        row = {"method": "cublas_fp16", "N": N, "K": K, "T": T, "us": us, "gbs": gbs,
               "pct_copy": gbs / COPY_GBS}
        results["speed"].append(row)
        print(json.dumps(row), flush=True)
    del w16s
    torch.cuda.empty_cache()

# ---------- summary ----------
print("\nSUMMARY (best w4_gemv config per shape/T, plain = no norm/resid):")
best = {}
for r in results["speed"]:
    if r["method"] == "w4_gemv" and not r["norm"]:
        key = (r["N"], r["K"], r["T"])
        if key not in best or r["us"] < best[key]["us"]:
            best[key] = r
summary = []
for (N, K, T), b in sorted(best.items()):
    base = {r["method"]: r for r in results["speed"]
            if r["N"] == N and r["K"] == K and r["T"] == T and r["method"] != "w4_gemv"}
    cb = base.get("cublas_fp16")
    dq = base.get("dequant_torch_then_matmul")
    line = {"N": N, "K": K, "T": T, "w4_us": round(b["us"], 2), "w4_gbs": round(b["gbs"], 1),
            "w4_pct": round(100 * b["pct_copy"], 1), "rpw": b["rpw"], "slab": b["slab"],
            "cublas_us": round(cb["us"], 2) if cb else None, "cublas_gbs": round(cb["gbs"], 1) if cb else None,
            "dequant_us": round(dq["us"], 2) if dq else None}
    summary.append(line)
    print(json.dumps(line))
results["summary"] = summary
results["all_correct"] = bool(all_ok)

out = "/kaggle/working/results.json" if os.path.isdir("/kaggle/working") else "results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print("ALL CORRECT" if all_ok else "CORRECTNESS FAILURES, see FAIL rows")
if not all_ok:
    raise SystemExit(1)
