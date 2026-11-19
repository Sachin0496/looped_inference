# SenseNova-Loop-9B RELEASE weights: ungated BF16 GGUF -> HF safetensors (text-only), then the existing
# KV analysis (kaggle/kv_analysis/kv_analysis.py) on the converted directory.
# Converter (engine/convert/gguf_to_hf.py) and KV analysis sources are embedded by kaggle/render.py.
# Note: the converter takes the Preview config.json (release config is not published); GGUF metadata
# overrides are applied by the converter itself. Vision tower is not in the GGUF (random init, text-only use).
import glob, json, importlib.util, os, shutil, subprocess, sys, time, traceback

T0 = time.time()
GGUF_REPO = "sensenova/SenseNova-Loop-9B-GGUF"
GGUF_FILE = "SenseNova-Loop-9B-BF16.gguf"
PREVIEW_REPO = "sensenova/SenseNova-Loop-9B-Preview"
GGUF_DIR = "/tmp/release_src"
META_DIR = "/tmp/preview_meta"
HF_DIR = "/tmp/release_hf"
CONVERTER = "/tmp/gguf_to_hf.py"
KV_SCRIPT = "/tmp/kv_analysis.py"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out"
os.makedirs(OUT, exist_ok=True)
LOGF = open(os.path.join(OUT, "kv_release.log"), "a", buffering=1)

CONVERTER_SRC = r'''__CONVERTER_SOURCE__'''
KV_SRC = r'''__KV_ANALYSIS_SOURCE__'''


def log(*a):
    msg = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOGF.write(msg + "\n")


def run(cmd, env=None, cwd=None, check=True):
    """Run a command, streaming merged stdout/stderr to the log. Returns (rc, lines)."""
    shown = cmd if isinstance(cmd, str) else " ".join(cmd)
    log("$", shown)
    p = subprocess.Popen(cmd, shell=isinstance(cmd, str), env=env, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        print(line, flush=True)
        LOGF.write(line + "\n")
    rc = p.wait()
    log(f"exit {rc}")
    if check and rc != 0:
        raise RuntimeError(f"command failed (rc={rc}): {shown}")
    return rc, lines


def disk(tag):
    log(f"disk ({tag}):")
    run("df -h /tmp /kaggle/working", check=False)


def key_diff():
    """Names/dtypes/shapes of the converted safetensors vs the Preview checkpoint (the reference layout)."""
    spec = importlib.util.spec_from_file_location("gguf_to_hf", CONVERTER)
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    index_path = os.path.join(META_DIR, "model.safetensors.index.json")
    index = json.load(open(index_path))
    ref = conv.load_preview_shapes(None, index)  # Preview shard headers, fetched with HTTP range reads
    got = {}
    files = sorted(glob.glob(os.path.join(HF_DIR, "*.safetensors")))
    log("converted shards:", [os.path.basename(f) for f in files])
    for fn in files:
        hdr, _ = conv.read_st_header_local(fn)
        for k, v in hdr.items():
            if k != "__metadata__":
                got[k] = {"dtype": v["dtype"], "shape": v["shape"], "file": os.path.basename(fn)}
    out_index = os.path.join(HF_DIR, "model.safetensors.index.json")
    dangling = []
    if os.path.exists(out_index):
        wm = json.load(open(out_index))["weight_map"]
        dangling = sorted({f for f in wm.values() if not os.path.exists(os.path.join(HF_DIR, f))})
    skip = lambda n: n.startswith("model.visual.") or n.startswith("mtp.")  # expected absent with --text-only
    missing = sorted(n for n in ref if n not in got and not skip(n))
    unexpected = sorted(n for n in got if n not in ref)
    mismatch = [{"name": n, "preview": {"dtype": ref[n]["dtype"], "shape": ref[n]["shape"]},
                 "converted": {"dtype": got[n]["dtype"], "shape": got[n]["shape"]}}
                for n in sorted(got) if n in ref and (got[n]["dtype"] != ref[n]["dtype"] or got[n]["shape"] != ref[n]["shape"])]
    diff = {"missing": missing, "unexpected": unexpected, "mismatch": mismatch, "index_dangling_files": dangling,
            "converted_count": len(got), "preview_count": len(ref)}
    json.dump(diff, open(os.path.join(OUT, "key_diff.json"), "w"), indent=1)
    log(f"KEY DIFF: converted {len(got)} / preview {len(ref)} tensors; missing {len(missing)}, "
        f"unexpected {len(unexpected)}, shape/dtype mismatch {len(mismatch)}, index files absent {dangling}")
    for label, rows in (("missing", missing), ("unexpected", unexpected), ("mismatch", mismatch)):
        for r in rows[:40]:
            log(f"  {label}: {r}")


def main():
    disk("start")
    run([sys.executable, "-m", "pip", "install", "-q", "gguf", "huggingface_hub"])
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(GGUF_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    log("downloading", GGUF_REPO, GGUF_FILE, "(18.4 GB)")
    gguf = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=GGUF_DIR)
    log("gguf at", gguf, f"{os.path.getsize(gguf) / 1e9:.2f} GB")

    names = [f for f in HfApi().list_repo_files(PREVIEW_REPO)
             if "/" not in f and f.endswith((".json", ".jinja", ".txt", ".model", ".tiktoken"))]
    for f in names:
        hf_hub_download(PREVIEW_REPO, f, local_dir=META_DIR)
    log("preview config/tokenizer files:", names)

    open(CONVERTER, "w").write(CONVERTER_SRC)
    conv_args = ["--gguf", gguf, "--config", os.path.join(META_DIR, "config.json"),
                 "--preview-index", os.path.join(META_DIR, "model.safetensors.index.json")]
    run([sys.executable, "-u", CONVERTER, *conv_args, "--check"])
    run([sys.executable, "-u", CONVERTER, *conv_args, "--text-only", "--out", HF_DIR])

    # The converter writes config.json and the weight index; add the Preview tokenizer and chat template files.
    for f in names:
        if f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(os.path.join(META_DIR, f), os.path.join(HF_DIR, f))
    log("converted dir:", sorted(os.listdir(HF_DIR)))

    os.remove(gguf)
    shutil.rmtree(GGUF_DIR, ignore_errors=True)
    log("deleted GGUF to free disk")
    disk("after conversion")

    open(KV_SCRIPT, "w").write(KV_SRC)
    env = dict(os.environ, SNKV_WEIGHTS=HF_DIR, PYTHONUNBUFFERED="1")
    rc, _ = run([sys.executable, "-u", KV_SCRIPT], env=env, cwd="/tmp", check=False)
    if rc != 0:
        raise RuntimeError(f"kv_analysis failed (rc={rc}); see the log above")
    log("done; outputs in", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        log("FAILED; key diff of the converted directory against the Preview checkpoint follows")
        try:
            key_diff()
        except Exception:
            traceback.print_exc()
        sys.exit(1)

# Streaming-dataset worker threads can keep the interpreter alive after the script ends (seen on Kaggle:
# distill A printed "done" at 3.6 h, then the session stayed RUNNING for hours). Exit hard after flushing.
import sys as _sys, os as _os
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)
