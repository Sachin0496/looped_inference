# SenseNovaKV B1: training-free KV analysis of SenseNova-Loop-9B on Kaggle 2xT4.
#
# 1. Load the model (fp16, split across 2 GPUs), measure baseline perplexity.
# 2. One calibration pass collects second-order statistics of the pre-norm [K;V] projection outputs
#    (2048-d per token) for every full-attention visit (14 = 2 outside the loop + 6 looped layers x 2),
#    plus cross-covariances visit1<->visit2 and between paired layers.
# 3. From the statistics alone (no training) build variants and measure perplexity:
#      pv{r}            per-visit PCA latent of rank r          (MLA-style low-rank KV, one latent per visit)
#      pv{r}_fp4[h]     + FP4 (E2M1, E4M3 scale per 16) latent [h: Hadamard-rotated first]
#      share{r}...      visit 2 rebuilds its K/V from visit 1's latent (one latent per physical layer)
#      share{r}_pair... + paired looped layers (7,11) (15,19) (23,27): 2nd layer rebuilds from 1st's latent
#    Regression maps are least-squares fits computed from the covariances (closed form).
# Every number written to results.json is measured in this run.
import json, math, os, subprocess, sys, time

T0 = time.time()
WEIGHTS = os.environ.get("SNKV_WEIGHTS", "sensenova/SenseNova-Loop-9B-Preview")
TINY = os.environ.get("SNKV_TINY")  # path to a SenseNova-Loop source checkout: CPU smoke test, random tiny model
SEQ = 64 if TINY else 2048
N_CALIB = int(os.environ.get("SNKV_N_CALIB", 32))
EVAL_BUDGET_S = float(os.environ.get("SNKV_EVAL_BUDGET_S", 3.5 * 3600))
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out"
os.makedirs(OUT, exist_ok=True)
RES = {"weights": WEIGHTS, "seq": SEQ, "log": []}


def log(*a):
    msg = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    RES["log"].append(msg)


def save():
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(RES, f, indent=1)


def sh(cmd):
    log("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


if TINY:
    sys.path.insert(0, os.path.join(TINY, "src"))
else:
    sh("pip install -q 'transformers==5.18.0' 'accelerate>=1.11,<2' 'safetensors>=0.8' datasets 2>&1 | tail -2")
    sh("pip install -q --no-deps git+https://github.com/OpenSenseNova/SenseNova-Loop.git 2>&1 | tail -2")

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

log("torch", torch.__version__, "gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

if TINY:
    path = None
elif os.path.isdir(WEIGHTS):
    path = WEIGHTS
else:
    path = snapshot_download(WEIGHTS, local_dir="/tmp/weights",
                             allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt", "vocab*", "merges*"])  # mtp.safetensors is listed in the weight index, so it must be present
log("weights at", path)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from sensenova_loop.models import register
register()

if TINY:
    from huggingface_hub import hf_hub_download
    from sensenova_loop.models.modeling_sensenova_loop import SenseNovaLoopConfig
    raw = json.load(open(hf_hub_download(WEIGHTS, "config.json")))
    t = raw["text_config"]
    t.update(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
             linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
             vocab_size=512, mtp_num_hidden_layers=0, pad_token_id=0, eos_token_id=1)
    raw.update(pad_token_id=0, eos_token_id=1, image_token_id=500, video_token_id=501, vision_start_token_id=502, vision_end_token_id=503)
    raw["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    raw.pop("auto_map", None)
    cfg = SenseNovaLoopConfig(**raw)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    DTYPE = torch.float32
    class _Tok:
        eos_token_id = 1
        def __call__(self, t, add_special_tokens=False):
            return {"input_ids": [ord(ch) % 500 + 2 for ch in t]}
    tok = _Tok()
else:
    tok = AutoTokenizer.from_pretrained(path)
    cfg = AutoConfig.from_pretrained(path)
    cfg.text_config.mtp_num_hidden_layers = 0  # MTP head not needed here (and its companion file is not downloaded)
    DTYPE = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        path, config=cfg, dtype=DTYPE, attn_implementation="sdpa", device_map="auto",
        max_memory={0: "13GiB", 1: "14GiB", "cpu": "24GiB"})
model.eval()
log("loaded; device map sample:", {k: str(v) for k, v in list(getattr(model, "hf_device_map", {}).items())[:6]})

text_model = model.model.language_model
tcfg = text_model.config
FULL = [i for i, t in enumerate(tcfg.layer_types) if t == "full_attention"]
LOOP = (cfg.loop_settings["start"], cfg.loop_settings["end"])
LOOPED = [i for i in FULL if LOOP[0] <= i < LOOP[1]]
PAIRS = [(LOOPED[k], LOOPED[k + 1]) for k in range(0, len(LOOPED) - 1, 2)]
H, HD = tcfg.num_key_value_heads, tcfg.head_dim
KVD = 2 * H * HD
log("full-attn layers", FULL, "looped", LOOPED, "pairs", PAIRS, "kv dim", KVD)
RES.update(full=FULL, looped=LOOPED, pairs=PAIRS, kv_dim=KVD)

# ------------------------------------------------------------------ data


def chunks_from_texts(texts, n, skip=0):
    ids, out = [], []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id or 0])
        while len(ids) >= SEQ:
            if skip > 0:
                skip -= 1
            else:
                out.append(torch.tensor(ids[:SEQ]))
                if len(out) >= n:
                    return out
            ids = ids[SEQ:]
    return out


if TINY:
    import random
    rnd_ = random.Random(0)
    WIKI = chunks_from_texts(["".join(rnd_.choice("abcdefgh ij") for _ in range(64 * 70))], 64)
    fw_texts = ["".join(rnd_.choice("klmnop qrs") for _ in range(400)) for _ in range(4000)]
else:
    from datasets import load_dataset
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    WIKI = chunks_from_texts(["\n".join(wiki["text"])], 64)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    fw_texts = [r["text"] for _, r in zip(range(4000), fw)]
CALIB = chunks_from_texts(fw_texts[:2000], N_CALIB)
FWEVAL = chunks_from_texts(fw_texts[2000:], 64)
log(f"data: calib {len(CALIB)}, wiki eval {len(WIKI)}, fineweb eval {len(FWEVAL)} x {SEQ}")

# ------------------------------------------------------------------ hooks
attn = {i: text_model.layers[i].self_attn for i in FULL}
STATE = {"mode": "off", "visit": {}, "stash_v": {}, "stash_kv": {}, "stash_c": {}}


def reset_visits(*_):
    STATE["visit"] = {i: -1 for i in FULL}
    STATE["stash_v"].clear(); STATE["stash_kv"].clear(); STATE["stash_c"].clear()


text_model.register_forward_pre_hook(lambda m, a: reset_visits())
for i in FULL:
    def pre(m, a, kw=None, i=i):
        STATE["visit"][i] += 1
    attn[i].register_forward_pre_hook(pre)

STATS = {}


def stat(key, dim_a, dim_b, dev):
    if key not in STATS:
        STATS[key] = {"n": 0, "sa": torch.zeros(dim_a, dtype=torch.float64, device=dev),
                      "sb": torch.zeros(dim_b, dtype=torch.float64, device=dev),
                      "sab": torch.zeros(dim_a, dim_b, dtype=torch.float64, device=dev)}
    return STATS[key]


def accumulate(key, a, b):
    s = stat(key, a.shape[-1], b.shape[-1], a.device)
    a = a.reshape(-1, a.shape[-1]).double(); b = b.reshape(-1, b.shape[-1]).double().to(a.device)
    s["n"] += a.shape[0]; s["sa"] += a.sum(0); s["sb"] += b.sum(0); s["sab"] += a.T @ b


VARIANT = {}


def fp4_quant(c, group=16):
    """E2M1 values with an E4M3 scale per group of 16 along the last dim (simulated)."""
    shp = c.shape
    g = c.float().reshape(-1, group)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float().clamp_min(2 ** -9)
    x = (g / scale).clamp(-6, 6)
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=c.device)
    q = grid[(x.abs().unsqueeze(-1) - grid).abs().argmin(-1)] * x.sign()
    return (q * scale).reshape(shp).to(c.dtype)


def hadamard(n, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def latent_of(i, v, kv):
    """kv [*, KVD] -> latent [*, r] using the visit's PCA basis (+ optional rotation/FP4)."""
    P = VARIANT["P"][(i, v)]; mu = VARIANT["mu"][(i, v)]
    c = (kv.float() - mu) @ P
    R = VARIANT["rotd"][c.device] if VARIANT.get("rotd") else None
    if R is not None:
        c = c @ R
    if VARIANT.get("fp4"):
        c = fp4_quant(c)
    if R is not None:
        c = c @ R.T
    return c


def k_hook(m, args, out, i=None):
    mode = STATE["mode"]
    if mode == "off":
        return out
    x = args[0]
    v = STATE["visit"][i]
    vout = F.linear(x, attn[i].v_proj.weight)
    kv = torch.cat([out, vout], -1)
    if mode == "collect":
        accumulate(("self", i, v), kv, kv)
        if i in LOOPED and v == 1:
            accumulate(("visit", i), kv, STATE["stash_kv"][(i, 0)])        # cov(kv2, kv1)
        for a, b in PAIRS:
            if i == b:
                accumulate(("pair", a, b, v), kv, STATE["stash_kv"][(a, v)])  # cov(kv_b, kv_a) same visit
                if v == 1:
                    accumulate(("pair01", a, b), kv, STATE["stash_kv"][(a, 0)])  # cov(kv_b visit2, kv_a visit1)
        STATE["stash_kv"][(i, v)] = kv
        return out
    # eval
    kind = VARIANT["kind"]
    if kind == "base":
        new = kv
    else:
        src = (i, v)
        if kind in ("share", "share_pair") and i in LOOPED and v == 1:
            src = (i, 0)
        if kind == "share_pair":
            for a, b in PAIRS:
                if i == b:
                    src = (a, 0)  # only (a, visit 1) is stored for the pair
        if src == (i, v):
            c = latent_of(i, v, kv)
            STATE["stash_c"][(i, v)] = c
            new = c @ VARIANT["P"][(i, v)].T + VARIANT["mu"][(i, v)]
        else:
            c = STATE["stash_c"][src].to(kv.device)
            Bm, mu = VARIANT["map"][((i, v), src)]
            new = c @ Bm.T + mu
        new = new.to(kv.dtype)
    STATE["stash_v"][i] = new[..., H * HD:]
    return new[..., :H * HD]


def v_hook(m, args, out, i=None):
    if STATE["mode"] != "eval":
        return out
    return STATE["stash_v"].pop(i)


for i in FULL:
    attn[i].k_proj.register_forward_hook(lambda m, a, o, i=i: k_hook(m, a, o, i))
    attn[i].v_proj.register_forward_hook(lambda m, a, o, i=i: v_hook(m, a, o, i))

dev0 = next(model.parameters()).device


@torch.no_grad()
def nll(seqs, limit_s=None):
    tot, cnt, t0 = 0.0, 0, time.time()
    for k, s in enumerate(seqs):
        ids = s.unsqueeze(0).to(dev0)
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
        tgt = ids[0, 1:].to(h.device)
        lp = 0.0
        for c0 in range(0, h.shape[0], 256):
            logits = model.lm_head(h[c0:c0 + 256]).float()
            lp = lp + F.cross_entropy(logits.to(tgt.device), tgt[c0:c0 + 256], reduction="sum")
        if not torch.isfinite(lp):
            return float("nan"), cnt
        tot += lp.item(); cnt += SEQ - 1
        if limit_s and time.time() - t0 > limit_s:
            break
    return tot / max(cnt, 1), cnt


# ------------------------------------------------------------------ baseline + throughput
STATE["mode"] = "off"
t0 = time.time()
b_w, n = nll(WIKI[:2])
tps = n / (time.time() - t0)
log(f"fp16 warm-up: wiki nll {b_w:.4f} on {n} tokens, {tps:.0f} tok/s")
RES["throughput_tok_s"] = tps
if not math.isfinite(b_w):
    log("fp16 produced non-finite loss; aborting (needs bf16 path)")
    save(); sys.exit(1)

VARIANTS_PLAN = ["base", "pv2048", "pv512", "pv256", "pv128", "pv256_fp4", "pv256_fp4h", "pv512_fp4h",
                 "share512", "share256", "share256_fp4h", "share512_fp4h", "share256_pair_fp4h"]
per_variant_s = EVAL_BUDGET_S / len(VARIANTS_PLAN)
n_eval = max(4, min(64, int(per_variant_s * tps / (2 * SEQ))))
log(f"eval: {n_eval} wiki + {n_eval} fineweb sequences per variant")
RES["n_eval_seqs_each"] = n_eval

# ------------------------------------------------------------------ calibration
STATE["mode"] = "collect"
with torch.no_grad():
    for k, s in enumerate(CALIB):
        model.model(input_ids=s.unsqueeze(0).to(dev0), use_cache=False)
log(f"calibration done on {len(CALIB) * SEQ} tokens")
STATE["mode"] = "off"


def centered(key):
    """All statistics are moved to one device: paired layers can live on different GPUs."""
    s = STATS[key]; n = s["n"]
    ma, mb = s["sa"].to(dev0) / n, s["sb"].to(dev0) / n
    return s["sab"].to(dev0) / n - torch.outer(ma, mb), ma, mb


def rk(r):
    return max(4, r // 16) if TINY else r  # tiny smoke test has KV dim 128


EIG = {}
spectra = {}
for i in FULL:
    for v in ((0, 1) if i in LOOPED else (0,)):
        C, mu, _ = centered(("self", i, v))
        evals, evecs = torch.linalg.eigh(C)
        evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
        EIG[(i, v)] = (evals, evecs, mu)
        frac = (evals.cumsum(0) / evals.sum()).cpu()
        spectra[f"{i}.{v}"] = {f"energy@{r}": float(frac[min(rk(r), len(frac)) - 1]) for r in (64, 128, 256, 512, 1024)}
RES["pca_energy"] = spectra
log("PCA energy (layer.visit):", json.dumps(spectra))

# R^2 of predicting visit-2 KV from visit-1 latent / paired layer latent, per rank
r2 = {}
for r_name in (256, 512):
    r = rk(r_name)
    for i in LOOPED:
        C21, mu2, mu1 = centered(("visit", i))
        ev1, P1, _ = EIG[(i, 0)]
        Pr, lam = P1[:, :r], ev1[:r].clamp_min(1e-9)
        B = (C21 @ Pr) / lam
        C22 = EIG[(i, 1)]
        tot = C22[0].sum()
        expl = (B * (C21 @ Pr)).sum()
        r2[f"visit r{r_name} L{i}"] = float(expl / tot)
    for a, b in PAIRS:
        for v in (0, 1):
            Cba, mub, mua = centered(("pair", a, b, v))
            eva, Pa, _ = EIG[(a, v)]
            Pr, lam = Pa[:, :r], eva[:r].clamp_min(1e-9)
            B = (Cba @ Pr) / lam
            r2[f"pair r{r_name} L{a}->L{b} v{v}"] = float((B * (Cba @ Pr)).sum() / EIG[(b, v)][0].sum())
RES["r2_explained_variance"] = r2
log("R2:", json.dumps(r2))
save()

# keep PCA bases for B2 initialisation
torch.save({f"{i}.{v}": {"evecs": e[1][:, :1024].float().cpu(), "evals": e[0][:1024].float().cpu(), "mu": e[2].float().cpu()}
            for (i, v), e in EIG.items()}, f"{OUT}/pca_bases.pt")


def build_variant(name):
    VARIANT.clear()
    if name == "base":
        VARIANT["kind"] = "base"; return
    kind = "share_pair" if "pair" in name else ("share" if name.startswith("share") else "pv")
    r = rk(int("".join(ch for ch in name.split("_")[0] if ch.isdigit())))
    VARIANT.update(kind=kind, r=r, fp4="fp4" in name)
    devs = {attn[i].k_proj.weight.device for i in FULL}
    VARIANT["rotd"] = {d: hadamard(r, d) for d in devs} if name.endswith("h") else None
    VARIANT["P"], VARIANT["mu"], VARIANT["map"] = {}, {}, {}
    for (i, v), (ev, P, mu) in EIG.items():
        VARIANT["P"][(i, v)] = P[:, :r].float(); VARIANT["mu"][(i, v)] = mu.float()

    def fit(dst, src, C_ds, mu_d):
        ev, P, _ = EIG[src]
        Pr, lam = P[:, :r], ev[:r].clamp_min(1e-9)
        return ((C_ds @ Pr) / lam).float(), mu_d.float()

    if kind in ("share", "share_pair"):
        for i in LOOPED:
            C21, mu2, _ = centered(("visit", i))
            VARIANT["map"][((i, 1), (i, 0))] = fit((i, 1), (i, 0), C21, mu2)
    if kind == "share_pair":
        for a, b in PAIRS:
            Cb0, mub0, _ = centered(("pair", a, b, 0))
            VARIANT["map"][((b, 0), (a, 0))] = fit((b, 0), (a, 0), Cb0, mub0)
            Cb1, mub1, _ = centered(("pair01", a, b))
            VARIANT["map"][((b, 1), (a, 0))] = fit((b, 1), (a, 0), Cb1, mub1)


def bytes_per_token(name):
    if name == "base":
        return 14 * KVD * 2
    r = VARIANT["r"]
    per_latent = (r * 0.5 + r / 16) if VARIANT["fp4"] else 2 * r
    n_lat = {"pv": 14, "share": len(FULL), "share_pair": len(FULL) - len(PAIRS)}[VARIANT["kind"]]
    return n_lat * per_latent


RES["variants"] = {}
for name in VARIANTS_PLAN:
    build_variant(name)
    for key in ("P", "mu"):
        if key in VARIANT:
            VARIANT[key] = {k: t.to(attn[k[0]].k_proj.weight.device) for k, t in VARIANT[key].items()}
    if VARIANT.get("map"):
        VARIANT["map"] = {k: (B.to(attn[k[0][0]].k_proj.weight.device), m.to(attn[k[0][0]].k_proj.weight.device))
                          for k, (B, m) in VARIANT["map"].items()}
    STATE["mode"] = "off" if name == "base" else "eval"
    t0 = time.time()
    w, nw = nll(WIKI[:n_eval])
    f, nf = nll(FWEVAL[:n_eval])
    row = {"wiki_nll": w, "wiki_ppl": math.exp(w) if math.isfinite(w) else None,
           "fineweb_nll": f, "fineweb_ppl": math.exp(f) if math.isfinite(f) else None,
           "tokens": nw + nf, "seconds": time.time() - t0, "kv_bytes_per_token": bytes_per_token(name)}
    RES["variants"][name] = row
    log(name, json.dumps(row))
    save()

base = RES["variants"]["base"]
for name, row in RES["variants"].items():
    if row["wiki_nll"] is not None and math.isfinite(row["wiki_nll"]):
        row["d_wiki_ppl_pct"] = 100 * (math.exp(row["wiki_nll"] - base["wiki_nll"]) - 1)
        row["d_fineweb_ppl_pct"] = 100 * (math.exp(row["fineweb_nll"] - base["fineweb_nll"]) - 1)
save()
log("done")

# Streaming-dataset worker threads can keep the interpreter alive after the script ends (seen on Kaggle:
# distill A printed "done" at 3.6 h, then the session stayed RUNNING for hours). Exit hard after flushing.
import sys as _sys, os as _os
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)
