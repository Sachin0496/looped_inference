# SenseNovaKV engine_e2e v1: single-sequence decode engine (engine/runtime/snkv_engine.py, embedded below) against
# the HF fp16 teacher on 2xT4, plus HF and engine benchmarks. Sources are embedded by kaggle/render.py.
#
# Stages, each one try/except: a traceback is logged and the run goes on with what is left.
#   install -> weights (release GGUF converted as kv_release.py does; Preview snapshot if that fails)
#   -> hf_load (2 GPUs) -> hf_validation x3 (prefill + 64 greedy steps; logits and prefill caches kept on CPU)
#   -> hf_bench 1K / 8K (HF generate, decode tokens/s; prefill caches kept for the engine)
#   -> pack (4-bit, packing on each weight's GPU) -> free HF -> engine_build (cuda:0, kernels compiled)
#   -> engine_validation (teacher-forced: the HF greedy tokens are fed to the engine, logits compared per step)
#   -> engine_bench 1K / 8K (CUDA graph tokens/s, peak memory on cuda:0), projected KV at 32K / 128K.
# results.json is rewritten after every stage; engine_e2e.log holds the full log.
import gc, json, os, shutil, subprocess, sys, time, traceback, types

import torch
import torch.nn.functional as F

T0 = time.time()
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out"
BUILD = os.path.join(OUT, "build")
os.makedirs(BUILD, exist_ok=True)
LOGF = open(os.path.join(OUT, "engine_e2e.log"), "a", buffering=1)
RES = {"version": "engine_e2e v1", "status": "running", "weights": None, "stages": {}, "errors": {},
       "hf": {"validation": [], "bench": {}}, "engine": {"build": {}, "validation": [], "bench": {}}, "log": []}

PREVIEW_REPO = "sensenova/SenseNova-Loop-9B-Preview"
GGUF_REPO = "sensenova/SenseNova-Loop-9B-GGUF"
GGUF_FILE = "SenseNova-Loop-9B-BF16.gguf"
GGUF_DIR, META_DIR, HF_DIR = "/tmp/release_src", "/tmp/preview_meta", "/tmp/release_hf"
CONVERTER = "/tmp/gguf_to_hf.py"
STEPS = 64
N_VAL = 3
BENCH_CTX = [1024, 8192]
BENCH_STEPS = 64
TIME_BUDGET_S = 10.0 * 3600  # no new HF bench / engine bench after this

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
    with open(os.path.join(OUT, "results.json"), "w") as f:
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
    return TIME_BUDGET_S - (time.time() - T0)


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


# ------------------------------------------------------------------------------ HF side
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
    """Prefill (kept as a CPU snapshot for the engine), then STEPS greedy steps. Returns the fed tokens and the HF
    logits after each fed token (fp16, CPU)."""
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
    return {"S": int(ids.shape[1]), "snap": snap, "fed": fed, "logits": torch.stack(logits)}


def hf_bench(model, ctx, kinds):
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
    res = {"ctx": ctx, "prefill_plus_1_s": t1, "prefill_plus_65_s": t65,
           "decode_tok_per_s": BENCH_STEPS / max(t65 - t1, 1e-9), "method": "generate(65) - generate(1)"}
    with torch.no_grad():
        out = model(input_ids=ids.to(dev), use_cache=True)
    res["snap"] = E.snapshot_cache(out.past_key_values, kinds)
    del out
    return res


def pack_host(model):
    return E.export_host(model)


# ------------------------------------------------------------------------------ engine side
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
    st = E.to_device(host, "cuda:0", E.CUDA_POLICY)
    RES["engine"]["build"]["weights_bytes"] = E.weights_bytes(st)
    RES["engine"]["build"]["policy"] = repr(E.CUDA_POLICY)
    log(f"engine weights on cuda:0: {E.weights_bytes(st) / 2**30:.2f} GiB")
    return st, ops


def engine_validate(st, ops, val, kinds):
    rows_all = []
    for i, hv in enumerate(val):
        eng = E.Engine(st, ops, max_len=hv["S"] + STEPS + 4, policy=E.CUDA_POLICY, device="cuda:0")
        eng.capture()
        eng.load_prefill(hv["snap"])
        rows = []
        for k in range(STEPS):
            out = eng.step(hv["fed"][k]).float()[0]
            ref = hv["logits"][k].to("cuda:0").float()
            lp_h = F.log_softmax(ref, -1)
            lp_e = F.log_softmax(out, -1)
            kl = float((lp_h.exp() * (lp_h - lp_e)).sum())
            rows.append({"step": k + 1, "top1_agree": bool(int(out.argmax()) == int(ref.argmax())),
                         "max_abs_dlogit": float((out - ref).abs().max()), "kl_hf_engine": kl})
        summ = {"prompt": i, "S": hv["S"], "steps": rows,
                "top1_agree_frac": sum(r["top1_agree"] for r in rows) / len(rows),
                "step1_max_abs_dlogit": rows[0]["max_abs_dlogit"],
                "mean_kl_hf_engine": sum(r["kl_hf_engine"] for r in rows) / len(rows)}
        log(f"prompt {i}: top1 {summ['top1_agree_frac']:.3f}, step1 max|dlogit| {summ['step1_max_abs_dlogit']:.4f}, "
            f"mean KL {summ['mean_kl_hf_engine']:.5f}")
        RES["engine"]["validation"].append(summ)
        save()
        rows_all.append(summ)
        del eng
        torch.cuda.empty_cache()
    if rows_all:
        RES["engine"]["validation_summary"] = {
            "top1_agree_frac": sum(s["top1_agree_frac"] for s in rows_all) / len(rows_all),
            "step1_max_abs_dlogit": max(s["step1_max_abs_dlogit"] for s in rows_all),
            "mean_kl_hf_engine": sum(s["mean_kl_hf_engine"] for s in rows_all) / len(rows_all)}
    return rows_all


def engine_bench(st, ops, bench_snaps):
    kv_tok = E.kv_bytes_per_token(st["dims"])
    state_bytes = None
    for ctx in BENCH_CTX:
        if budget_left() < 900:
            log("time budget low: skipping engine bench", ctx)
            break
        torch.cuda.synchronize("cuda:0")
        torch.cuda.reset_peak_memory_stats("cuda:0")
        eng = E.Engine(st, ops, max_len=ctx + BENCH_STEPS + 8, policy=E.CUDA_POLICY, device="cuda:0")
        eng.capture()
        eng.load_prefill(bench_snaps[ctx])
        for _ in range(4):
            eng.step(1)
        torch.cuda.synchronize("cuda:0")
        t0 = time.perf_counter()
        for _ in range(BENCH_STEPS):
            eng.step(1)
        torch.cuda.synchronize("cuda:0")
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated("cuda:0")
        state_bytes = eng.state_bytes()
        rec = {"ctx": ctx, "decode_tok_per_s": BENCH_STEPS / dt, "ms_per_token": 1000 * dt / BENCH_STEPS,
               "kv_allocated_bytes": eng.kv_bytes(), "state_bytes": state_bytes,
               "peak_allocated_bytes_cuda0": peak, "peak_allocated_GiB": peak / 2**30}
        log(f"engine ctx {ctx}: {rec['decode_tok_per_s']:.2f} tok/s ({rec['ms_per_token']:.2f} ms/tok), "
            f"peak {peak / 2**30:.2f} GiB")
        RES["engine"]["bench"][str(ctx)] = rec
        save()
        del eng
        torch.cuda.empty_cache()
    proj = {}
    for ctx in (32768, 131072):
        proj[str(ctx)] = {"kv_bytes": kv_tok * ctx, "kv_GiB": kv_tok * ctx / 2**30, "method": "57344 B/token x ctx"}
    RES["engine"]["projection"] = {"kv_bytes_per_token": kv_tok, "state_bytes_constant": state_bytes, "by_ctx": proj}


# ------------------------------------------------------------------------------ main
def main():
    global E
    stage("install", install)
    path = stage("weights", prepare_weights)
    if path is None:
        RES["status"] = "failed: no weights"
        return
    # the engine module is pure torch: it is safe to import before the HF model is loaded
    spec_path = os.path.join(BUILD, "snkv_engine.py")
    open(spec_path, "w").write(ENGINE_SRC)
    import importlib.util
    spec = importlib.util.spec_from_file_location("snkv_engine", spec_path)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)

    loaded = stage("hf_load", load_hf, path)
    model, tok = loaded if loaded is not None else (None, None)  # noqa
    if model is not None:
        kinds = [r[2] for r in E.visit_table(model.model.language_model.config.layer_types,
                                             E.build_plan(model.config.text_config.num_hidden_layers,
                                                          model.config.to_dict()["loop_settings"]),
                                             (model.config.to_dict()["mhc"]["start"],
                                              model.config.to_dict()["mhc"]["end"]))]
        val = []
        for i, ids in enumerate(prompt_ids(tok)[:N_VAL]):
            r = stage(f"hf_validation_{i}", hf_validate, model, ids, kinds)
            if r is not None:
                val.append(r)
                RES["hf"]["validation"].append({"prompt": i, "S": r["S"], "fed_tokens": r["fed"][:8] + ["..."],
                                                "n_steps": len(r["fed"])})
                save()
        bench_snaps = {}
        for ctx in BENCH_CTX:
            if budget_left() < 3600:
                log("time budget low: skipping HF bench", ctx)
                break
            r = stage(f"hf_bench_{ctx}", hf_bench, model, ctx, kinds)
            if r is not None:
                bench_snaps[ctx] = r.pop("snap")
                RES["hf"]["bench"][str(ctx)] = r
                save()
        host = stage("pack", pack_host, model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        stage("free_hf", lambda: None)
    else:
        host, bench_snaps, val = None, {}, []
    if host is None or not torch.cuda.is_available():
        RES["status"] = "no engine run (HF stages only): " + ("hf_load failed" if model is None else "pack failed")
        save()
        return
    built = stage("engine_build", engine_build, host)
    if built is None:
        RES["status"] = "engine_build failed"
        save()
        return
    st, ops = built
    stage("engine_validation", engine_validate, st, ops, val, kinds) if val else log("no validation snapshots")
    if bench_snaps:
        stage("engine_bench", engine_bench, st, ops, bench_snaps)
    RES["status"] = "done (see stages/errors)"
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        RES["errors"]["main"] = traceback.format_exc()
        log(RES["errors"]["main"])
    finally:
        RES["wall_s"] = round(time.time() - T0, 1)
        save()
        log("finished; outputs in", OUT, sorted(os.listdir(OUT)))

# Streaming-dataset worker threads can keep the interpreter alive after the script ends (seen on Kaggle:
# distill A printed "done" at 3.6 h, then the session stayed RUNNING for hours). Exit hard after flushing.
import sys as _sys, os as _os
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)
