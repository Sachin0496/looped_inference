# Training-free LOOP-CONVERGENCE diagnostic for SenseNova-Loop-9B (Preview weights, 2xT4).
#
# Question: how cheap is it to skip the second loop pass per token?
# 1. Convergence signal, from the 2-loop run: r_t = ||h2_t - h1_t|| / ||h1_t||, where h1 / h2 are the
#    outputs of physical layer 27 (last layer of the loop block [4, 28); mHC leave() is applied there)
#    at loop pass 1 / pass 2. Captured by a forward hook that counts visits (reset per forward).
# 2. Cost of skipping pass 2: the same sequence with the loop executed once (_loop_plan = range(32)).
#    Per position: KL(p_2loops || p_1loop), top-1 agreement, dNLL = NLL_1loop - NLL_2loop on the true next token.
# 3. Summaries: NLL/PPL, quantiles and histogram of r, Spearman(r, KL), Spearman(r, dNLL), r by position
#    bucket, an "oracle exit" table at quantiles of r, and the 20 most / least converged tokens.
#
# CAVEAT (also written into results.json): this is a per-token approximation. The 1-loop run changes every
# position, so KL_t and dNLL_t also include the effect of earlier positions' skipped pass through their
# attention inputs. A real early exit would also change later tokens' attention to visit-2 KV, which this
# diagnostic does not model. The oracle-exit table is a decomposition of measured per-token costs by r
# quantile; it is not an exit policy and not an end-to-end quality number.
#
# Setup, weight download and smoke mode mirror kaggle/kv_analysis/kv_analysis.py.
# Smoke test: SNKV_TINY=<SenseNova-Loop checkout> python3 loop_conv.py  (CPU, random tiny model)
# Output: /kaggle/working/results.json and /kaggle/working/per_token.npz
import sys as _sys0, os as _os0, traceback as _tb0
def _hard_exit_on_error(et, ev, tb):  # streaming-dataset threads keep the process alive after an exception
    _tb0.print_exception(et, ev, tb); _sys0.stdout.flush(); _sys0.stderr.flush(); _os0._exit(1)
_sys0.excepthook = _hard_exit_on_error

import json, math, os, subprocess, sys, time

T0 = time.time()
WEIGHTS = os.environ.get("SNKV_WEIGHTS", "sensenova/SenseNova-Loop-9B-Preview")
TINY = os.environ.get("SNKV_TINY")  # path to a SenseNova-Loop source checkout: CPU smoke test, random tiny model
SEQ = 64 if TINY else 2048
N_SEQ = int(os.environ.get("SNKV_N_SEQ", 4 if TINY else 16))  # sequences per dataset
OUT = os.environ.get("SNKV_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else "./out")
os.makedirs(OUT, exist_ok=True)
CAVEAT = ("PER-TOKEN APPROXIMATION. The 1-loop run changes every position, so KL_t and dNLL_t at position t "
          "also include the effect of earlier positions' skipped second pass (through their attention inputs). "
          "A real early exit would also change later tokens' attention to visit-2 KV, which this diagnostic "
          "does not model. The oracle-exit table groups measured per-token costs by r quantile; it is not an "
          "exit policy and not an end-to-end quality number.")
RES = {"weights": WEIGHTS, "seq": SEQ, "n_seq_per_dataset": N_SEQ, "tiny": bool(TINY), "caveat": CAVEAT, "log": []}


def log(*a):
    msg = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    RES["log"].append(msg)


def save():
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(RES, f, indent=1, default=float)


def sh(cmd):
    log("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


if TINY:
    sys.path.insert(0, os.path.join(TINY, "src"))
else:
    sh("pip install -q 'transformers==5.18.0' 'accelerate>=1.11,<2' 'safetensors>=0.8' datasets 2>&1 | tail -2")
    sh("pip install -q --no-deps git+https://github.com/OpenSenseNova/SenseNova-Loop.git 2>&1 | tail -2")

import numpy as np
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

        def decode(self, ids):
            return "".join(f"<{i}>" for i in ids)
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
NL = text_model.config.num_hidden_layers
LOOP = (cfg.loop_settings["start"], cfg.loop_settings["end"])
LAST = LOOP[1] - 1  # physical layer 27: last looped layer, output has mHC leave() applied
PLAN2 = tuple(text_model._loop_plan)  # default 2-loop plan
PLAN1 = tuple(range(NL))  # loop executed once; mHC enter (at LOOP[0]) and leave (at LAST) still run once each
assert PLAN2 == tuple(range(LOOP[0])) + tuple(range(*LOOP)) * cfg.loop_settings["eval_count"] + tuple(range(LOOP[1], NL)), PLAN2
assert PLAN2.count(LAST) == 2 and PLAN1.count(LAST) == 1
log("loop block", LOOP, "last looped layer", LAST, "plan2 len", len(PLAN2), "plan1 len", len(PLAN1))
RES.update(loop=LOOP, last_looped_layer=LAST, plan2_len=len(PLAN2), plan1_len=len(PLAN1))

# Visit capture: forward hook on the last looped layer; one entry per visit, reset in run().
CAP = {"on": False, "h": []}


def _cap(m, args, out):
    if CAP["on"]:
        x = out[0] if isinstance(out, tuple) else out
        CAP["h"].append(x[0].detach())  # [T, D]


text_model.layers[LAST].register_forward_hook(_cap)


@torch.no_grad()
def run(ids, plan):
    """Final hidden states [T, D] and the layer-LAST output of every visit, for one execution plan."""
    text_model._loop_plan = plan
    CAP["h"], CAP["on"] = [], True
    try:
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    finally:
        CAP["on"] = False
        text_model._loop_plan = PLAN2
    visits = CAP["h"]
    CAP["h"] = []
    return h, visits


# ------------------------------------------------------------------ data


def chunks_from_texts(texts, n):
    ids, out = [], []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id or 0])
        while len(ids) >= SEQ:
            out.append(torch.tensor(ids[:SEQ]))
            if len(out) >= n:
                return out
            ids = ids[SEQ:]
    return out


if TINY:
    import random
    rnd_ = random.Random(0)
    WIKI_TXT = ["".join(rnd_.choice("abcdefgh ij") for _ in range(SEQ * N_SEQ * 2))]
    FW_TXT = ["".join(rnd_.choice("klmnop qrs") for _ in range(400)) for _ in range(200)]
else:
    from datasets import load_dataset
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    WIKI_TXT = ["\n".join(wiki["text"])]
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    FW_TXT = [r["text"] for _, r in zip(range(600), fw)]
DATA = {"wikitext2": chunks_from_texts(WIKI_TXT, N_SEQ), "fineweb_edu": chunks_from_texts(FW_TXT, N_SEQ)}
assert all(len(v) == N_SEQ for v in DATA.values()), {k: len(v) for k, v in DATA.items()}
log(f"data: {N_SEQ} sequences x {SEQ} tokens per dataset", {k: len(v) for k, v in DATA.items()})
RES["data"] = {k: len(v) for k, v in DATA.items()}

dev0 = next(model.parameters()).device
dev_head = model.lm_head.weight.device


@torch.no_grad()
def per_sequence(ids):
    T = ids.shape[1]
    h2, v2 = run(ids, PLAN2)
    assert len(v2) == 2, f"expected 2 visits to layer {LAST}, got {len(v2)}"
    h1, v1 = run(ids, PLAN1)
    assert len(v1) == 1, f"expected 1 visit to layer {LAST}, got {len(v1)}"
    a1, a2 = v2[0].float(), v2[1].float()
    # Plan check: the first pass of the 2-loop run must equal the only pass of the 1-loop run.
    plan_diff = float((a1 - v1[0].float()).norm() / v1[0].float().norm().clamp_min(1e-12))
    r = (a2 - a1).norm(dim=-1) / a1.norm(dim=-1).clamp_min(1e-12)  # [T]
    tgt = ids[0, 1:].to(dev_head)
    kl, agree, nll2, nll1 = [], [], [], []
    for c0 in range(0, T - 1, 128):
        c1 = min(c0 + 128, T - 1)
        l2 = F.log_softmax(model.lm_head(h2[c0:c1].to(dev_head)).float(), dim=-1)
        l1 = F.log_softmax(model.lm_head(h1[c0:c1].to(dev_head)).float(), dim=-1)
        kl.append((l2.exp() * (l2 - l1)).sum(-1))
        agree.append(l2.argmax(-1) == l1.argmax(-1))
        g = tgt[c0:c1, None]
        nll2.append(-l2.gather(-1, g).squeeze(-1))
        nll1.append(-l1.gather(-1, g).squeeze(-1))
    cat = lambda xs: torch.cat(xs).cpu().numpy()
    out = {"r": r[:T - 1].cpu().numpy(), "kl": cat(kl), "nll1": cat(nll1), "nll2": cat(nll2),
           "agree": cat(agree), "tok": ids[0, :T - 1].cpu().numpy(), "pos": np.arange(T - 1)}
    out["dnll"] = out["nll1"] - out["nll2"]
    assert np.isfinite(out["kl"]).all() and np.isfinite(out["nll1"]).all() and np.isfinite(out["nll2"]).all(), "non-finite"
    return out, plan_diff


ACC = {}
PLAN_DIFFS = []
t0 = time.time()
for name, seqs in DATA.items():
    ACC[name] = []
    for k, s in enumerate(seqs):
        d, pd = per_sequence(s.unsqueeze(0).to(dev0))
        PLAN_DIFFS.append(pd)
        ACC[name].append(d)
        if k % 4 == 0:
            log(f"{name} seq {k + 1}/{len(seqs)}: mean r {d['r'].mean():.4g}, mean dNLL {d['dnll'].mean():.4g}, "
                f"plan check rel diff {pd:.2e}")
log(f"forward passes done in {time.time() - t0:.0f}s")
RES["plan_check_visit1_rel_diff_max"] = max(PLAN_DIFFS)
log("plan check (visit-1 output, 2-loop vs 1-loop plan): max rel diff", max(PLAN_DIFFS))
assert max(PLAN_DIFFS) < 0.05, "single-pass plan does not reproduce the first loop pass"

# ------------------------------------------------------------------ analysis

NAMES = list(DATA.keys())
DS = np.concatenate([np.full(len(d["r"]), i) for i, n in enumerate(NAMES) for d in ACC[n]])
A = {k: np.concatenate([d[k] for n in NAMES for d in ACC[n]]) for k in ("r", "kl", "dnll", "nll1", "nll2", "agree", "tok", "pos")}
R, KL, DN, POS, AG = A["r"].astype(np.float64), A["kl"].astype(np.float64), A["dnll"].astype(np.float64), A["pos"], A["agree"]
N_TOK = len(R)
log("tokens scored:", N_TOK)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def summary(mask):
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return {"n": 0}
    return {"n": int(m.sum()), "frac": float(m.mean()), "mean_kl": float(KL[m].mean()),
            "mean_dnll": float(DN[m].mean()), "top1_agree": float(AG[m].mean())}


def ppl_block(sel):
    n1, n2 = A["nll1"][sel].astype(np.float64), A["nll2"][sel].astype(np.float64)
    return {"tokens": int(sel.sum()), "nll_1loop": float(n1.mean()), "nll_2loop": float(n2.mean()),
            "ppl_1loop": float(math.exp(n1.mean())), "ppl_2loop": float(math.exp(n2.mean())),
            "ppl_delta_pct": float(100 * (math.exp(n1.mean() - n2.mean()) - 1)),
            "mean_kl": float(KL[sel].mean()), "median_kl": float(np.median(KL[sel])),
            "top1_agree": float(AG[sel].mean())}


all_sel = np.ones(N_TOK, dtype=bool)
RES["overall"] = ppl_block(all_sel)
RES["per_dataset"] = {n: ppl_block(DS == i) for i, n in enumerate(NAMES)}
RES["spearman"] = {"r_vs_kl": spearman(R, KL), "r_vs_dnll": spearman(R, DN),
                   "per_dataset_r_vs_kl": {n: spearman(R[DS == i], KL[DS == i]) for i, n in enumerate(NAMES)},
                   "per_dataset_r_vs_dnll": {n: spearman(R[DS == i], DN[DS == i]) for i, n in enumerate(NAMES)}}
QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
RES["r_quantiles"] = {f"q{q:.2f}": float(v) for q, v in zip(QS, np.quantile(R, QS))}
RES["r_mean"] = float(R.mean())
counts, edges = np.histogram(np.log10(np.maximum(R, 1e-12)), bins=20)
RES["r_hist_log10"] = {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}
RES["kl_quantiles"] = {f"q{q:.2f}": float(v) for q, v in zip(QS, np.quantile(KL, QS))}
RES["dnll_quantiles"] = {f"q{q:.2f}": float(v) for q, v in zip(QS, np.quantile(A["nll1"] - A["nll2"], QS))}

BUCKETS = [(0, 16), (16, 64), (64, 256), (256, 512), (512, 1024), (1024, 2048)]
RES["r_by_position"] = {}
for lo, hi in BUCKETS:
    m = (POS >= lo) & (POS < hi)
    if m.sum():
        RES["r_by_position"][f"[{lo},{hi})"] = {"n": int(m.sum()), "mean_r": float(R[m].mean()),
                                                 "median_r": float(np.median(R[m])), "mean_kl": float(KL[m].mean()),
                                                 "mean_dnll": float(DN[m].mean())}

ORACLE = []
for q in (0.10, 0.25, 0.50, 0.75):
    tau = float(np.quantile(R, q))
    m = R < tau
    ORACLE.append({"quantile": q, "tau": tau, "on_r_below_tau": summary(m), "rest": summary(~m)})
RES["oracle_exit"] = ORACLE
RES["oracle_exit_note"] = ("Tokens with r_t < tau would use one loop pass. KL / dNLL / top-1 are the measured "
                           "per-token values of the 1-loop run, grouped by r. Per-token approximation, see caveat.")


def tok_row(i):
    s = tok.decode([int(A["tok"][i])])
    return {"token": repr(s), "id": int(A["tok"][i]), "r": float(R[i]), "kl": float(KL[i]),
            "dnll": float(DN[i]), "pos": int(POS[i]), "dataset": NAMES[int(DS[i])]}


order = np.argsort(R)
RES["top20_most_converged"] = [tok_row(i) for i in order[:20]]
RES["top20_least_converged"] = [tok_row(i) for i in order[::-1][:20]]

np.savez_compressed(f"{OUT}/per_token.npz", r=R, kl=KL, dnll=DN, agree=AG, pos=POS,
                    tok=A["tok"], ds=DS, names=np.array(NAMES))

# ------------------------------------------------------------------ report
log("CAVEAT:", CAVEAT)
o = RES["overall"]
log(f"overall: PPL 1-loop {o['ppl_1loop']:.4f} vs 2-loop {o['ppl_2loop']:.4f} ({o['ppl_delta_pct']:+.2f}%), "
    f"mean KL {o['mean_kl']:.4g}, top-1 agree {o['top1_agree']:.4f}")
for n in NAMES:
    p = RES["per_dataset"][n]
    log(f"  {n}: PPL 1-loop {p['ppl_1loop']:.4f} vs 2-loop {p['ppl_2loop']:.4f} ({p['ppl_delta_pct']:+.2f}%)")
log("r quantiles:", json.dumps({k: round(v, 5) for k, v in RES["r_quantiles"].items()}))
log("spearman:", json.dumps(RES["spearman"]))
for row in ORACLE:
    on, rest = row["on_r_below_tau"], row["rest"]
    log(f"oracle q{row['quantile']:.2f} tau={row['tau']:.4g}: frac {on['frac']:.3f} | "
        f"on KL {on['mean_kl']:.4g} dNLL {on['mean_dnll']:.4g} top1 {on['top1_agree']:.4f} | "
        f"rest KL {rest['mean_kl']:.4g} dNLL {rest['mean_dnll']:.4g} top1 {rest['top1_agree']:.4f}")
for k, v in RES["r_by_position"].items():
    log(f"pos {k}: n {v['n']} mean r {v['mean_r']:.4g} median r {v['median_r']:.4g}")
log("most converged:", [x["token"] for x in RES["top20_most_converged"]])
log("least converged:", [x["token"] for x in RES["top20_least_converged"]])
RES["wall_s"] = time.time() - T0
log("done")
save()

# Streaming-dataset worker threads can keep the interpreter alive after the script ends (seen on Kaggle:
# distill A printed "done" at 3.6 h, then the session stayed RUNNING for hours). Exit hard after flushing.
import sys as _sys, os as _os
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)
