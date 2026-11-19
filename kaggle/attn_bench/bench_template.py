# SenseNovaKV: latent decode attention over an FP4 compressed cache (split-KV) vs PyTorch SDPA over an
# uncompressed fp16 GQA cache. Two kernels in one module: kv4_attn (v1, fp32 CUDA cores, gate 1e-3) and
# kv4_attn_tc (v2, WMMA fp16 tensor cores, gate 5e-3). Correctness against the fp32 reference on the dequantized
# cache, then timing of SDPA, v1 and v2 in the same run (same cache ring, same contexts).
# Rendered by kaggle/render.py: the kernel and reference sources are embedded below.
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
with open(f"{work}/kv4_attn.cu", "w") as f:
    f.write(CU_SRC)
t0 = time.time()
ext = load(name="kv4_attn", sources=[f"{work}/kv4_attn.cu"], build_directory=work,
           extra_cflags=["-O3", "-DKV4_TORCH_EXTENSION"],
           extra_cuda_cflags=["-O3", "-DKV4_TORCH_EXTENSION", "-lineinfo"], verbose=False)
print(f"compiled in {time.time() - t0:.1f}s")

ref = types.ModuleType("kv4_ref")
sys.modules["kv4_ref"] = ref
exec(REF_SRC, ref.__dict__)

results = {"device": props.name, "torch": torch.__version__, "correctness": [], "speed": []}
REL_TOL = {"kv4_attn": 1e-3, "kv4_attn_tc": 5e-3}
KERNELS = [
    ("kv4_attn", lambda qa, qr, c: ext.kv4_attn(qa, qr, c["cq"], c["cs"], c["kc"], c["krs"], c["ks"])),
    ("kv4_attn_tc", lambda qa, qr, c: ext.kv4_attn_tc(qa, qr, c["cq"], c["cs"], c["kc"], c["krs"], c["ks"])),
]
CTX = [4096, 32768, 131072, 524288]
RS = (256, 512)
TS = (1, 4)
MB96 = 96 << 20
GQA_KV_HEADS, HEAD_DIM = 4, 256
BASE_BPT = 2 * GQA_KV_HEADS * HEAD_DIM * 2             # K and V, fp16, per token per layer: 4096 B


def comp_bpt(R):
    # FP4 latent + E4M3 block scales + FP8 rope key (4 groups x 64) + fp16 kr scale (x4) + fp16 ks (x4)
    return R // 2 + R // 16 + 4 * 64 + 4 * 2 + 4 * 2


def ring_size(nbytes):
    """Copies of the cache rotated per call, so that every call reads DRAM rather than L2."""
    return max(2, min(8, math.ceil(MB96 / nbytes)))


def time_graph(fns, replays=20, warm=3):
    """Mean GPU time per call in us. All calls are captured in ONE CUDA graph and replayed."""
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


# measured copy bandwidth (read + write bytes / time), the same convention as the other benches
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


def make_cache(T, S, R, seed, keep_fp32):
    """Random latent / rope key, quantized into the compressed format. Returns a dict of device tensors."""
    g = torch.Generator(device=dev).manual_seed(seed)
    c = torch.randn(T, S, R, generator=g, device=dev) * torch.exp(0.5 * torch.randn(T, S, 1, generator=g, device=dev))
    kr = torch.randn(T, S, 4, 64, generator=g, device=dev)
    ks = (0.5 + 1.5 * torch.rand(T, S, 4, generator=g, device=dev)).half()
    cq, cs = ref.quant_c(c)
    kc, krs = ref.quant_kr(kr)
    d = dict(cq=cq, cs=cs, kc=kc, krs=krs, ks=ks)
    if keep_fp32:
        d.update(c=c, kr=kr)
    return d


def queries(T, R, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    qa = (torch.randn(T, 16, R, generator=g, device=dev) * 0.5).contiguous()
    qr = (torch.randn(T, 16, 64, generator=g, device=dev) * 0.5).contiguous()
    return qa, qr


# ---------- correctness: kernel vs fp32 reference on the dequantized cache ----------
all_ok = True
for R in RS:
    for T in TS:
        for S in (1000, 4096, 32768):
            cache = make_cache(T, S, R, seed=R * 7 + T * 3 + S, keep_fp32=True)
            qa, qr = queries(T, R, seed=R + T + S)
            yref = ref.attn_ref(qa, qr, ref.dequant_c(cache["cq"], cache["cs"]),
                                ref.dequant_kr(cache["kc"], cache["krs"]), cache["ks"])
            yfp = ref.attn_ref(qa, qr, cache["c"], cache["kr"], cache["ks"])
            scale = yref.abs().max().item()
            rel_fp = ((yref - yfp).abs().max() / yfp.abs().max()).item()
            for name, fn in KERNELS:
                y = fn(qa, qr, cache)
                torch.cuda.synchronize()
                rel = ((y - yref).abs().max() / scale).item()
                row = {"method": name, "R": R, "T": T, "S": S, "rel_vs_dequant_ref": rel,
                       "rel_vs_fp32_cache_info": rel_fp, "tol": REL_TOL[name],
                       "status": "PASS" if rel < REL_TOL[name] else "FAIL"}
                all_ok &= row["status"] == "PASS"
                results["correctness"].append(row)
                print(json.dumps(row), flush=True)
                del y
            del cache, yref, yfp
            torch.cuda.empty_cache()

# ---------- speed: compressed kernel vs SDPA over uncompressed fp16 GQA cache ----------
for T in TS:
    for S in CTX:
        if S == 524288 and T != 1:
            continue  # memory: the baseline cache at T=4 does not fit with the rings
        qg = torch.randn(T, GQA_KV_HEADS, 4, HEAD_DIM, device=dev).half()  # 16 query heads as 4 rows per KV head
        base_bytes = T * S * BASE_BPT
        nb = ring_size(base_bytes)
        bK = [torch.randn(T, GQA_KV_HEADS, S, HEAD_DIM, device=dev).half() for _ in range(nb)]
        bV = [torch.randn(T, GQA_KV_HEADS, S, HEAD_DIM, device=dev).half() for _ in range(nb)]
        fns = [(lambda i=i: F.scaled_dot_product_attention(qg, bK[i], bV[i])) for i in range(nb)]
        us_b = time_graph(fns)
        gbs_b = T * S * BASE_BPT / (us_b * 1e-6) / 1e9
        row = {"method": "sdpa_fp16_gqa", "T": T, "S": S, "us": us_b, "cache_bytes": T * S * BASE_BPT,
               "bytes_per_token_layer": BASE_BPT, "gbs": gbs_b, "pct_copy": gbs_b / COPY_GBS}
        results["speed"].append(row)
        print(json.dumps(row), flush=True)
        del bK, bV, fns
        torch.cuda.empty_cache()

        for R in RS:
            bpt = comp_bpt(R)
            cbytes = T * S * bpt
            nr = ring_size(cbytes)
            base = make_cache(T, S, R, seed=R + 17 * T + S, keep_fp32=False)
            ring = [{k: v.clone() for k, v in base.items()} for _ in range(nr)]
            del base
            qa, qr = queries(T, R, seed=5)
            us_v1 = None
            for name, fn in KERNELS:   # v1 and v2 timed on the same ring, in the same run
                fns = [(lambda i=i, fn=fn: fn(qa, qr, ring[i])) for i in range(nr)]
                us = time_graph(fns)
                gbs = cbytes / (us * 1e-6) / 1e9
                row = {"method": name, "R": R, "T": T, "S": S, "us": us, "cache_bytes": cbytes,
                       "bytes_per_token_layer": bpt, "gbs": gbs, "pct_copy": gbs / COPY_GBS,
                       "speedup_vs_sdpa": us_b / us, "bytes_ratio_vs_sdpa": BASE_BPT / bpt}
                if us_v1 is None:
                    us_v1 = us
                else:
                    row["speedup_vs_v1"] = us_v1 / us
                results["speed"].append(row)
                print(json.dumps(row), flush=True)
                del fns
            del ring
            torch.cuda.empty_cache()

# ---------- summary ----------
print("\nSUMMARY (per call; cache bytes = the bytes read from DRAM per call):")
for r in results["speed"]:
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
results["all_correct"] = bool(all_ok)

out = "/kaggle/working/results.json" if os.path.isdir("/kaggle/working") else "results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print("ALL CORRECT" if all_ok else "CORRECTNESS FAILURES, see FAIL rows")
if not all_ok:
    raise SystemExit(1)
