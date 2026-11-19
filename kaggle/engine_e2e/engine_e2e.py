# SenseNovaKV engine_e2e v2: the HF fp16 teacher and the 4-bit engine run in SEPARATE processes, so the engine's memory
# is measured with nothing else on the GPU. Engine source: engine/runtime/snkv_engine.py (embedded by kaggle/render.py).
#
# Process 1 (this script, default mode):
#   install -> weights (release GGUF converted as kv_release.py does; Preview snapshot if that fails)
#   -> hf_load (2 GPUs) -> hf_validation_0..2 (prefill + 64 greedy steps; full fp16 logits and prefill snapshots kept)
#   -> hf_ref_save (/kaggle/working/hf_ref.pt: prompt ids, HF greedy tokens, HF logits, prefill snapshots)
#   -> pack (4-bit, packing on each weight's GPU) -> pack_save (/tmp/engine_weights/packed.pt)
#   -> hf_bench_1024 -> free_hf -> engine_child (subprocess = process 2, then this process exits)
# Process 2 (python engine_e2e.py --engine-child, CUDA_VISIBLE_DEVICES=0, fresh interpreter):
#   engine_load -> engine_build (cuda:0, kernels compiled) -> engine_validation (teacher-forced with the HF greedy
#   tokens, logits compared with hf_ref.pt) -> engine_bench_1024 / _8192 / _32768. Each context gets a fresh engine
#   with max_len = ctx + 72; its KV is filled by the engine's own CUDA-graph decode step over a synthetic prompt
#   (no prefill kernel exists in the engine), then 64 timed decode steps. Memory per context: torch max_memory_allocated
#   and max_memory_reserved, plus nvidia-smi memory.used.
# Stages, each one try/except: a traceback is logged and the run goes on with what is left.
# results.json is rewritten after every stage by both processes; engine_e2e.log holds the full log.
import sys as _sys0, os as _os0, traceback as _tb0
def _hard_exit_on_error(et, ev, tb):  # streaming-dataset threads keep the process alive after an exception
    _tb0.print_exception(et, ev, tb); _sys0.stdout.flush(); _sys0.stderr.flush(); _os0._exit(1)
_sys0.excepthook = _hard_exit_on_error

import gc, json, os, shutil, subprocess, sys, time, traceback

# expandable segments for process 1 (HF stages); process 2 runs without it (CUDA graphs, see run_engine_child)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

T0 = time.time()
TIME_BUDGET_S = 10.0 * 3600  # no new stage starts after this (process 2 inherits the absolute deadline)
DEADLINE = float(os.environ.get("SNKV_DEADLINE", T0 + TIME_BUDGET_S))
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out"
BUILD = os.path.join(OUT, "build")
os.makedirs(BUILD, exist_ok=True)
LOGF = open(os.path.join(OUT, "engine_e2e.log"), "a", buffering=1)
RESULTS_PATH = os.path.join(OUT, "results.json")
HF_REF = os.path.join(OUT, "hf_ref.pt")
PACK_DIR = "/tmp/engine_weights"
PACKED = os.path.join(PACK_DIR, "packed.pt")
RES = {"version": "engine_e2e v2", "status": "running", "weights": None, "stages": {}, "errors": {},
       "hf": {"validation": [], "bench": {}}, "engine": {"build": {}, "validation": [], "bench": {}},
       "verdict": None, "log": []}

PREVIEW_REPO = "sensenova/SenseNova-Loop-9B-Preview"
GGUF_REPO = "sensenova/SenseNova-Loop-9B-GGUF"
GGUF_FILE = "SenseNova-Loop-9B-BF16.gguf"
GGUF_DIR, META_DIR, HF_DIR = "/tmp/release_src", "/tmp/preview_meta", "/tmp/release_hf"
CONVERTER = "/tmp/gguf_to_hf.py"
STEPS = 64            # validation: teacher-forced steps per prompt
N_VAL = 3             # validation prompts
HF_BENCH_CTX = [1024]
ENGINE_BENCH_CTX = [1024, 8192, 32768]
BENCH_STEPS = 64
MEM_TARGET_GIB = 8.0
DEV = "cuda:0"

ENGINE_SRC = r'''__ENGINE_SOURCE__'''
W4_CU_SRC = r'''__W4_CU_SOURCE__'''
MHC_CU_SRC = r'''__MHC_CU_SOURCE__'''
DN_CU_SRC = r'''__DN_CU_SOURCE__'''
CONVERTER_SRC = r'''__CONVERTER_SOURCE__'''

PROMPT_TEXTS = [
    "The water cycle describes how water moves between the oceans, the atmosphere and the land. Evaporation lifts "
    "vapour from seas and lakes, condensation gathers it into clouds, and precipitation returns it to the surface, "
    "where rivers carry it back to the sea. Plants take up groundwater and release vapour through their leaves. ",
    "A recipe for a simple tomato soup starts with onions softened in olive oil, then garlic, ripe tomatoes, a little "
    "stock and a pinch of salt. Simmer until the tomatoes break down, blend until smooth, and finish with cream and "
    "fresh basil. Serve warm with bread that has been toasted in the oven. ",
    "Compilers translate source code into machine instructions in several stages: lexing, parsing, semantic analysis, "
    "intermediate representation, optimisation and code generation. Each stage can report errors, and a good compiler "
    "keeps reporting them without stopping at the first mistake, so that a programmer sees many problems at once. ",
]


def log(*a):
    msg = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOGF.write(msg + "\n")
    RES["log"].append(msg)


def save():
    with open(RESULTS_PATH, "w") as f:
        json.dump(RES, f, indent=1, default=str)


def sh(cmd):
    log("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


def stage(name, fn, *args, **kwargs):
    t0 = time.time()
    log(f"== stage {name} start")
    try:
        r = fn(*args, **kwargs)
        RES["stages"][name] = {"ok": True, "wall_s": round(time.time() - t0, 1)}
        log(f"== stage {name} ok ({time.time() - t0:.0f}s)")
        return r
    except Exception:
        tb = traceback.format_exc()
        log(f"== stage {name} FAILED\n{tb}")
        RES["stages"][name] = {"ok": False, "wall_s": round(time.time() - t0, 1)}
        RES["errors"][name] = tb
        return None
    finally:
        save()


def budget_left():
    return DEADLINE - time.time()


def load_engine_module():
    """engine/runtime/snkv_engine.py as a module (pure torch). Written to BUILD so process 2 can load the same bytes."""
    import importlib.util
    spec_path = os.path.join(BUILD, "snkv_engine.py")
    open(spec_path, "w").write(ENGINE_SRC)
    spec = importlib.util.spec_from_file_location("snkv_engine", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = None  # the engine module, set by main() and engine_child()


# ------------------------------------------------------------------------------ weights
def install():
    sh(f"{sys.executable} -m pip install -q 'transformers==5.18.0' 'accelerate>=1.11,<2' 'safetensors>=0.8' "
       f"datasets gguf huggingface_hub 2>&1 | tail -2")
    sh(f"{sys.executable} -m pip install -q --no-deps git+https://github.com/OpenSenseNova/SenseNova-Loop.git 2>&1 | tail -2")
    RES["torch"] = torch.__version__
    RES["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    log("torch", torch.__version__, "gpus", RES["gpus"])


def release_hf_dir():
    """kv_release.py's steps (same converter, same flags): ungated BF16 GGUF -> text-only HF safetensors."""
    from huggingface_hub import HfApi, hf_hub_download
    os.makedirs(GGUF_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    gguf = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=GGUF_DIR)
    log("gguf at", gguf, f"{os.path.getsize(gguf) / 1e9:.2f} GB")
    names = [f for f in HfApi().list_repo_files(PREVIEW_REPO)
             if "/" not in f and f.endswith((".json", ".jinja", ".txt", ".model", ".tiktoken"))]
    for f in names:
        hf_hub_download(PREVIEW_REPO, f, local_dir=META_DIR)
    open(CONVERTER, "w").write(CONVERTER_SRC)
    conv_args = ["--gguf", gguf, "--config", os.path.join(META_DIR, "config.json"),
                 "--preview-index", os.path.join(META_DIR, "model.safetensors.index.json")]
    for extra in (["--check"], ["--text-only", "--out", HF_DIR]):
        cmd = [sys.executable, "-u", CONVERTER, *conv_args, *extra]
        log("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
    for f in names:
        if f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(os.path.join(META_DIR, f), os.path.join(HF_DIR, f))
    cfg_path = os.path.join(HF_DIR, "config.json")
    raw_cfg = json.load(open(cfg_path))
    raw_cfg["text_config"]["mtp_num_hidden_layers"] = 0
    json.dump(raw_cfg, open(cfg_path, "w"), indent=2)
    wm = json.load(open(os.path.join(HF_DIR, "model.safetensors.index.json")))["weight_map"]
    dangling = sorted({f for f in wm.values() if not os.path.exists(os.path.join(HF_DIR, f))})
    assert not dangling, f"converted weight index points at absent files: {dangling}"
    os.remove(gguf)
    shutil.rmtree(GGUF_DIR, ignore_errors=True)
    return HF_DIR


def prepare_weights():
    try:
        path = release_hf_dir()
        RES["weights"] = "release-gguf-converted"
        return path
    except Exception:
        RES["errors"]["weights_release"] = traceback.format_exc()
        log("release conversion failed; falling back to the Preview checkpoint\n" + RES["errors"]["weights_release"])
        from huggingface_hub import snapshot_download
        path = snapshot_download(PREVIEW_REPO, local_dir="/tmp/weights",
                                 allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt",
                                                 "vocab*", "merges*"])
        RES["weights"] = "preview-fallback"
        return path


# ------------------------------------------------------------------------------ process 1: HF side
def load_hf(path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from sensenova_loop.models import register
    register()
    tok = AutoTokenizer.from_pretrained(path)
    cfg = AutoConfig.from_pretrained(path)
    cfg.text_config.mtp_num_hidden_layers = 0
    model = AutoModelForCausalLM.from_pretrained(
        path, config=cfg, dtype=torch.float16, attn_implementation="sdpa", device_map="auto",
        max_memory={0: "13GiB", 1: "14GiB", "cpu": "24GiB"})
    model.eval()
    model.requires_grad_(False)
    log("HF loaded; device map sample:", {k: str(v) for k, v in list(model.hf_device_map.items())[:4]})
    return model, tok


def hf_kinds(model):
    return [r[2] for r in E.visit_table(model.model.language_model.config.layer_types,
                                        E.build_plan(model.config.text_config.num_hidden_layers,
                                                     model.config.to_dict()["loop_settings"]),
                                        (model.config.to_dict()["mhc"]["start"],
                                         model.config.to_dict()["mhc"]["end"]))]


def prompt_ids(tok):
    out = []
    for text in PROMPT_TEXTS:
        ids = tok(text * 4, add_special_tokens=False)["input_ids"][:384]
        out.append(torch.tensor([ids]))
    return out


def random_ids(ctx, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (1, ctx), generator=g)


def hf_dev(model):
    return model.get_input_embeddings().weight.device


def hf_validate(model, ids, kinds):
    """Prefill (kept as a CPU snapshot for the engine), then STEPS greedy steps. Returns the prompt ids, the fed tokens
    and the HF logits after each fed token (fp16, full vocab, CPU)."""
    dev = hf_dev(model)
    with torch.no_grad():
        out = model(input_ids=ids.to(dev), use_cache=True)
    cache = out.past_key_values
    snap = E.snapshot_cache(cache, kinds)
    t = int(out.logits[0, -1].argmax())
    del out
    fed, logits = [], []
    for _ in range(STEPS):
        fed.append(t)
        with torch.no_grad():
            o = model(input_ids=torch.tensor([[t]], device=dev), past_key_values=cache, use_cache=True)
        lg = o.logits[0, -1]
        logits.append(lg.half().cpu())
        t = int(lg.argmax())
        del o
    del cache
    return {"ids": ids.cpu(), "S": int(ids.shape[1]), "snap": snap, "fed": fed, "logits": torch.stack(logits)}


def save_hf_ref(vals):
    ref = {"version": 2, "steps": STEPS, "prompts": [v["ids"] for v in vals], "S": [v["S"] for v in vals],
           "fed": [v["fed"] for v in vals], "logits": [v["logits"] for v in vals], "snap": [v["snap"] for v in vals]}
    torch.save(ref, HF_REF)
    RES["hf"]["ref_file"] = {"path": HF_REF, "prompts": len(vals), "bytes": os.path.getsize(HF_REF)}
    log(f"hf_ref saved: {len(vals)} prompts, {os.path.getsize(HF_REF) / 2**20:.0f} MiB")
    return HF_REF


def hf_bench(model, ctx):
    dev = hf_dev(model)
    ids = random_ids(ctx, seed=ctx)
    attn = torch.ones_like(ids)

    def gen(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids=ids.to(dev), attention_mask=attn.to(dev), max_new_tokens=n, min_new_tokens=n,
                           do_sample=False, pad_token_id=0, use_cache=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    gen(4)
    t1 = min(gen(1) for _ in range(2))
    t65 = min(gen(BENCH_STEPS + 1) for _ in range(2))
    return {"ctx": ctx, "prefill_plus_1_s": t1, "prefill_plus_65_s": t65,
            "decode_tok_per_s": BENCH_STEPS / max(t65 - t1, 1e-9), "method": "generate(65) - generate(1)",
            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30}


def pack_save(model):
    host = E.export_host(model)  # 4-bit packing on each weight's GPU, results on CPU
    os.makedirs(PACK_DIR, exist_ok=True)
    torch.save(host, PACKED)
    size = os.path.getsize(PACKED)
    json.dump({"bytes": size, "dims": host["dims"]}, open(os.path.join(PACK_DIR, "meta.json"), "w"), default=str)
    RES["engine"]["packed"] = {"path": PACKED, "bytes": size}
    log(f"packed weights saved: {size / 2**30:.2f} GiB at {PACKED}")
    return size


# ------------------------------------------------------------------------------ process 1 -> 2
def run_engine_child():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env["SNKV_DEADLINE"] = str(DEADLINE)
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--engine-child"]
    log("$ (process 2) CUDA_VISIBLE_DEVICES=0", " ".join(cmd))
    save()
    p = None
    try:
        p = subprocess.run(cmd, env=env, timeout=max(300.0, budget_left() - 120))
        log("process 2 exit code", p.returncode)
    finally:
        # process 2 wrote results.json (it loaded ours and appended); adopt its state
        if os.path.exists(RESULTS_PATH):
            loaded = json.load(open(RESULTS_PATH))
            RES.clear()
            RES.update(loaded)
    return p.returncode


# ------------------------------------------------------------------------------ process 2: engine side
def load_packed():
    host = torch.load(PACKED, map_location="cpu", weights_only=False)
    log("packed weights loaded from", PACKED)
    return host


def load_hf_ref():
    ref = torch.load(HF_REF, map_location="cpu", weights_only=False)
    log(f"hf_ref loaded: {len(ref['fed'])} prompts, {ref['steps']} steps")
    return ref


def engine_build(host):
    B = BUILD
    srcs = {"w4_gemv": os.path.join(B, "w4_gemv.cu"), "mhc_fused": os.path.join(B, "mhc_fused.cu"),
            "deltanet_fused": os.path.join(B, "deltanet_fused.cu")}
    for name, text in (("w4_gemv", W4_CU_SRC), ("mhc_fused", MHC_CU_SRC), ("deltanet_fused", DN_CU_SRC)):
        open(srcs[name], "w").write(text)
    t0 = time.time()
    exts = E.load_cuda_extensions(srcs, os.path.join(B, "ext"))
    RES["engine"]["build"]["compile_s"] = round(time.time() - t0, 1)
    log(f"kernels compiled in {time.time() - t0:.0f}s")
    ops = E.Ops("cuda", exts=exts)
    st = E.to_device(host, DEV, E.CUDA_POLICY)
    torch.cuda.synchronize(DEV)
    resident = torch.cuda.memory_allocated(DEV)
    RES["engine"]["build"]["weights_bytes"] = E.weights_bytes(st)
    RES["engine"]["build"]["resident_allocated_bytes"] = resident
    RES["engine"]["build"]["policy"] = repr(E.CUDA_POLICY)
    log(f"engine weights on {DEV}: {E.weights_bytes(st) / 2**30:.2f} GiB (allocated {resident / 2**30:.2f} GiB)")
    return st, ops


def engine_validate(st, ops, ref):
    rows_all = []
    for i in range(len(ref["fed"])):
        S, fed = ref["S"][i], ref["fed"][i]
        eng = E.Engine(st, ops, max_len=S + STEPS + 4, policy=E.CUDA_POLICY, device=DEV)
        eng.capture()
        eng.load_prefill(ref["snap"][i])
        rows = []
        for k in range(STEPS):
            out = eng.step(fed[k]).float()[0]
            ref_lg = ref["logits"][i][k].to(DEV).float()
            lp_h = F.log_softmax(ref_lg, -1)
            lp_e = F.log_softmax(out, -1)
            kl = float((lp_h.exp() * (lp_h - lp_e)).sum())
            rows.append({"step": k + 1, "top1_agree": bool(int(out.argmax()) == int(ref_lg.argmax())),
                         "max_abs_dlogit": float((out - ref_lg).abs().max()), "kl_hf_engine": kl})
        summ = {"prompt": i, "S": S, "steps": rows,
                "top1_agree_frac": sum(r["top1_agree"] for r in rows) / len(rows),
                "step1_max_abs_dlogit": rows[0]["max_abs_dlogit"],
                "mean_kl_hf_engine": sum(r["kl_hf_engine"] for r in rows) / len(rows)}
        log(f"prompt {i}: top1 {summ['top1_agree_frac']:.3f}, step1 max|dlogit| {summ['step1_max_abs_dlogit']:.4f}, "
            f"mean KL {summ['mean_kl_hf_engine']:.5f}")
        RES["engine"]["validation"].append(summ)
        save()
        rows_all.append(summ)
        del eng
        gc.collect()
        torch.cuda.empty_cache()
    if rows_all:
        RES["engine"]["validation_summary"] = {
            "top1_agree_frac": sum(s["top1_agree_frac"] for s in rows_all) / len(rows_all),
            "step1_max_abs_dlogit": max(s["step1_max_abs_dlogit"] for s in rows_all),
            "mean_kl_hf_engine": sum(s["mean_kl_hf_engine"] for s in rows_all) / len(rows_all),
            "reference": "HF fp16 greedy tokens teacher-forced; full fp16 logits"}
    return rows_all


def own_gpu_memory_mib():
    """nvidia-smi memory.used of GPU 0 (whole device) and of this process (None if the container hides pids)."""
    gpu0 = own = None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                           capture_output=True, text=True, timeout=60).stdout.strip()
        gpu0 = float(q.splitlines()[0])
    except Exception as e:
        log("nvidia-smi gpu query failed:", e)
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60).stdout
        for line in q.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == os.getpid():
                own = float(parts[1])
    except Exception as e:
        log("nvidia-smi app query failed:", e)
    return gpu0, own


def engine_bench_ctx(st, ops, ctx):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEV)
    base = torch.cuda.memory_allocated(DEV)
    torch.cuda.reset_peak_memory_stats(DEV)
    eng = E.Engine(st, ops, max_len=ctx + BENCH_STEPS + 8, policy=E.CUDA_POLICY, device=DEV)
    eng.capture()
    # prefill: the engine has no prefill kernel, so the ctx-token synthetic prompt goes through the decode step
    g = torch.Generator().manual_seed(ctx)
    prompt = torch.randint(1000, 100000, (ctx,), generator=g).tolist()
    torch.cuda.synchronize(DEV)
    t0 = time.perf_counter()
    for t in prompt:
        eng.step(t)
    torch.cuda.synchronize(DEV)
    fill_s = time.perf_counter() - t0
    log(f"ctx {ctx}: filled {ctx} tokens in {fill_s:.0f}s ({ctx / fill_s:.1f} tok/s)")
    for _ in range(4):
        eng.step(1)
    torch.cuda.synchronize(DEV)
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        eng.step(1)
    torch.cuda.synchronize(DEV)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(DEV)
    reserved = torch.cuda.max_memory_reserved(DEV)
    gpu0, own = own_gpu_memory_mib()
    rec = {"ctx": ctx, "max_len": ctx + BENCH_STEPS + 8, "decode_tok_per_s": BENCH_STEPS / dt,
           "ms_per_token": 1000 * dt / BENCH_STEPS, "decode_steps": BENCH_STEPS,
           "prefill": {"method": "engine decode step token-by-token over a synthetic prompt (CUDA graph replay)",
                       "tokens": ctx, "wall_s": round(fill_s, 1), "tok_per_s": ctx / fill_s},
           "base_allocated_GiB": base / 2**30, "kv_allocated_GiB": eng.kv_bytes() / 2**30,
           "state_bytes": eng.state_bytes(), "peak_allocated_bytes": peak, "peak_allocated_GiB": peak / 2**30,
           "peak_reserved_GiB": reserved / 2**30, "smi_gpu0_used_MiB": gpu0, "smi_own_process_MiB": own}
    log(f"engine ctx {ctx}: {rec['decode_tok_per_s']:.2f} tok/s ({rec['ms_per_token']:.2f} ms/tok), "
        f"peak allocated {peak / 2**30:.2f} GiB, reserved {reserved / 2**30:.2f} GiB, "
        f"smi gpu0 {gpu0} MiB, own {own} MiB")
    RES["engine"]["bench"][str(ctx)] = rec
    save()
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def verdict():
    parts, ok = [], True
    for ctx in (8192, 32768):
        r = RES["engine"]["bench"].get(str(ctx))
        if r is None:
            parts.append(f"{ctx // 1024}K not run")
            ok = False
            continue
        g = r["peak_allocated_GiB"]
        parts.append(f"{ctx // 1024}K peak {g:.2f} GiB")
        ok = ok and g <= MEM_TARGET_GIB
    RES["verdict"] = ("VERDICT (torch max_memory_allocated, weights included): " + "; ".join(parts)
                      + f" vs {MEM_TARGET_GIB:.0f} GiB target: " + ("PASS" if ok else "FAIL"))
    log(RES["verdict"])


# ------------------------------------------------------------------------------ mains
def main():
    global E
    stage("install", install)
    path = stage("weights", prepare_weights)
    if path is None:
        RES["status"] = "failed: no weights"
        return
    E = load_engine_module()  # pure torch; safe before the HF model is loaded

    loaded = stage("hf_load", load_hf, path)
    if loaded is None:
        RES["status"] = "failed: hf_load (no engine run)"
        save()
        return
    model, tok = loaded
    kinds = hf_kinds(model)

    vals = []
    for i, ids in enumerate(prompt_ids(tok)[:N_VAL]):
        r = stage(f"hf_validation_{i}", hf_validate, model, ids, kinds)
        if r is not None:
            vals.append(r)
            RES["hf"]["validation"].append({"prompt": i, "S": r["S"], "fed_tokens": r["fed"][:8] + ["..."],
                                            "n_steps": len(r["fed"])})
            save()
    if vals:
        stage("hf_ref_save", save_hf_ref, vals)
    del vals
    packed_size = stage("pack", pack_save, model)
    if packed_size is not None:
        r = stage("hf_bench_1024", hf_bench, model, HF_BENCH_CTX[0])
        if r is not None:
            RES["hf"]["bench"][str(HF_BENCH_CTX[0])] = r
            save()
    del model, loaded
    gc.collect()
    torch.cuda.empty_cache()
    stage("free_hf", lambda: None)

    if packed_size is None or not os.path.exists(PACKED):
        RES["status"] = "no engine run: pack failed"
        save()
        return
    stage("engine_child", run_engine_child)
    RES["status"] = "done: " + ("all stages ok" if not RES["errors"] else
                                f"{len(RES['errors'])} stage(s) failed: {', '.join(RES['errors'])}")
    save()


def engine_child():
    global E
    RES.clear()
    RES.update(json.load(open(RESULTS_PATH)))
    RES["status"] = "running: process 2 (engine)"
    save()
    log("process 2 start; visible devices", torch.cuda.device_count(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    E = load_engine_module()
    host = stage("engine_load", load_packed)
    built = stage("engine_build", engine_build, host) if host is not None else None
    host = None
    if built is None:
        RES["status"] = "engine_build failed"
        save()
        return
    st, ops = built
    ref = stage("engine_load_ref", load_hf_ref)
    if ref is not None:
        stage("engine_validation", engine_validate, st, ops, ref)
    del ref
    for ctx in ENGINE_BENCH_CTX:
        if budget_left() < 900:
            log("time budget low: skipping engine bench", ctx)
            break
        stage(f"engine_bench_{ctx}", engine_bench_ctx, st, ops, ctx)
    verdict()
    RES["status"] = "process 2 done: " + ("all stages ok" if not RES["errors"] else
                                          f"{len(RES['errors'])} stage(s) failed: {', '.join(RES['errors'])}")
    save()


if __name__ == "__main__":
    child = "--engine-child" in sys.argv
    try:
        engine_child() if child else main()
    except Exception:
        RES["errors"]["main"] = traceback.format_exc()
        log(RES["errors"]["main"])
    finally:
        RES["child_wall_s" if child else "wall_s"] = round(time.time() - T0, 1)
        save()
        log("finished; outputs in", OUT, sorted(os.listdir(OUT)))

# Streaming-dataset worker threads can keep the interpreter alive after the script ends (seen on Kaggle:
# distill A printed "done" at 3.6 h, then the session stayed RUNNING for hours). Exit hard after flushing.
import sys as _sys, os as _os
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)
