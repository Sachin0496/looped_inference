# SenseNovaKV B2: layer-local distillation of a compressed latent KV cache (spec: kaggle/distill/SPEC.md).
# Every full-attention layer keeps its frozen teacher self_attn and gets a StudentAttn with the new params.
# The teacher residual stream always drives the model. Student outputs only feed the layer-local loss (train),
# a probe (init / final), or replace the teacher's attention output (eval).
# Config via SNKV_CONFIG (A, B, C). SNKV_TINY=<SenseNova-Loop checkout> runs a CPU smoke test on a random tiny model.
# Weights via SNKV_RELEASE=1 (RELEASE GGUF converted to HF, see release_hf_dir) or the Preview checkpoint (default).
import json, math, os, random, shutil, subprocess, sys, time
from contextlib import nullcontext

T0 = time.time()
CFG_NAME = os.environ.get("SNKV_CONFIG", "B")
CONFIGS = {
    "A": dict(r=512, res=0, share_loop=True, pair=False, rope="group", qat=True, hadamard=True),
    "B": dict(r=256, res=0, share_loop=True, pair=True, rope="shared", qat=True, hadamard=True),
    "C": dict(r=512, res=0, share_loop=False, pair=False, rope="group", qat=False, hadamard=False),
    # E: visit 1 of looped layers gets its own residual latent (res dims, FP4) on top of the visit-0 latent c0:
    # k/v = W_uk1 c0 + W_ukres c_res. Every visit owns its rope key (shared mode). No pair sharing.
    "E": dict(r=512, res=128, share_loop=True, pair=False, rope="shared", qat=True, hadamard=True),
}
CFG = dict(CONFIGS[CFG_NAME], name=CFG_NAME)
WEIGHTS = os.environ.get("SNKV_WEIGHTS", "sensenova/SenseNova-Loop-9B-Preview")
TINY = os.environ.get("SNKV_TINY")
# SNKV_RELEASE=1: train on the RELEASE weights, converted from the ungated BF16 GGUF (same steps as kv_release.py,
# converter embedded by render.py). Ignored in TINY mode.
RELEASE = os.environ.get("SNKV_RELEASE") == "1" and not TINY
RELEASE_GGUF_REPO = "sensenova/SenseNova-Loop-9B-GGUF"
RELEASE_GGUF_FILE = "SenseNova-Loop-9B-BF16.gguf"
PREVIEW_REPO = "sensenova/SenseNova-Loop-9B-Preview"
RELEASE_GGUF_DIR, RELEASE_META_DIR, RELEASE_HF_DIR = "/tmp/release_src", "/tmp/preview_meta", "/tmp/release_hf"
RELEASE_CONVERTER = "/tmp/gguf_to_hf.py"
CONVERTER_SRC = r'''__CONVERTER_SOURCE__'''
SEQ = 64 if TINY else 2048
N_CALIB = int(os.environ.get("SNKV_N_CALIB", 16 if TINY else 64))
N_EVAL = int(os.environ.get("SNKV_N_EVAL", 4 if TINY else 32))
N_MMLU = int(os.environ.get("SNKV_N_MMLU", 16 if TINY else 300))
GEN_TOK = 8 if TINY else 64
TRAIN_HOURS = float(os.environ.get("SNKV_TRAIN_HOURS", 3.0))
TRAIN_STEPS = int(os.environ.get("SNKV_STEPS", 4)) if TINY else 0
# SNKV_LR is a RELATIVE step: each tensor's lr = SNKV_LR * rms(its initial value). Adam's step size does not scale with
# the parameter, so one absolute lr over-steps small whitened tensors (config C drifted off its exact init at 3e-4).
LR_MAX, LR_MIN, WARM = float(os.environ.get("SNKV_LR", 2e-3)), 0.1 * float(os.environ.get("SNKV_LR", 2e-3)), 50
OUT = os.environ.get("SNKV_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else "./out")
os.makedirs(OUT, exist_ok=True)
RES = {"config": CFG_NAME, "tiny": bool(TINY), "seq": SEQ, "log": []}


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
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download

ON_CUDA = (not TINY) and torch.cuda.is_available()
log("torch", torch.__version__, "gpus", torch.cuda.device_count(), "config", CFG_NAME, CFG)

def release_hf_dir():
    """kv_release.py's conversion steps: ungated BF16 GGUF -> text-only HF safetensors + Preview config/tokenizer."""
    from huggingface_hub import HfApi, hf_hub_download
    sh(f"{sys.executable} -m pip install -q gguf huggingface_hub")
    os.makedirs(RELEASE_GGUF_DIR, exist_ok=True)
    os.makedirs(RELEASE_META_DIR, exist_ok=True)
    gguf = hf_hub_download(RELEASE_GGUF_REPO, RELEASE_GGUF_FILE, local_dir=RELEASE_GGUF_DIR)
    log("gguf at", gguf, f"{os.path.getsize(gguf) / 1e9:.2f} GB")
    names = [f for f in HfApi().list_repo_files(PREVIEW_REPO)
             if "/" not in f and f.endswith((".json", ".jinja", ".txt", ".model", ".tiktoken"))]
    for f in names:
        hf_hub_download(PREVIEW_REPO, f, local_dir=RELEASE_META_DIR)
    log("preview config/tokenizer files:", names)
    open(RELEASE_CONVERTER, "w").write(CONVERTER_SRC)
    conv_args = ["--gguf", gguf, "--config", os.path.join(RELEASE_META_DIR, "config.json"),
                 "--preview-index", os.path.join(RELEASE_META_DIR, "model.safetensors.index.json")]
    for extra in (["--check"], ["--text-only", "--out", RELEASE_HF_DIR]):
        cmd = [sys.executable, "-u", RELEASE_CONVERTER, *conv_args, *extra]
        log("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
    # The converter writes config.json and the weight index; add the Preview tokenizer and chat template files.
    for f in names:
        if f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(os.path.join(RELEASE_META_DIR, f), os.path.join(RELEASE_HF_DIR, f))
    cfg_path = os.path.join(RELEASE_HF_DIR, "config.json")
    raw_cfg = json.load(open(cfg_path))
    raw_cfg["text_config"]["mtp_num_hidden_layers"] = 0  # --text-only: no MTP head, no mtp.* tensors
    json.dump(raw_cfg, open(cfg_path, "w"), indent=2)
    wm = json.load(open(os.path.join(RELEASE_HF_DIR, "model.safetensors.index.json")))["weight_map"]
    dangling = sorted({f for f in wm.values() if not os.path.exists(os.path.join(RELEASE_HF_DIR, f))})
    assert not dangling, f"converted weight index points at absent files: {dangling}"
    log("converted dir:", sorted(os.listdir(RELEASE_HF_DIR)))
    os.remove(gguf)
    shutil.rmtree(RELEASE_GGUF_DIR, ignore_errors=True)
    log("deleted GGUF to free disk")
    return RELEASE_HF_DIR


if TINY:
    path = None
    RES["weights"] = "tiny-random"
elif RELEASE:
    path = release_hf_dir()
    RES["weights"] = "release-gguf-converted"
else:
    RES["weights"] = "preview"
    if os.path.isdir(WEIGHTS):
        path = WEIGHTS
    else:
        path = snapshot_download(WEIGHTS, local_dir="/tmp/weights",
                                 allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt", "vocab*", "merges*"])  # mtp.safetensors is in the weight index
log("weights used:", RES["weights"], "at", path)

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
    cfg.text_config.mtp_num_hidden_layers = 0  # MTP head not used; its companion file is not needed
    DTYPE = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        path, config=cfg, dtype=DTYPE, attn_implementation="sdpa", device_map="auto",
        max_memory={0: "11GiB", 1: "13GiB", "cpu": "24GiB"})
model.eval()
model.requires_grad_(False)  # teacher frozen; student params are created later with requires_grad=True

text_model = model.model.language_model
tcfg = text_model.config
FULL = [i for i, t_ in enumerate(tcfg.layer_types) if t_ == "full_attention"]
LOOP = (cfg.loop_settings["start"], cfg.loop_settings["end"])
LOOPED = [i for i in FULL if LOOP[0] <= i < LOOP[1]]
PAIRS = [(LOOPED[k], LOOPED[k + 1]) for k in range(0, len(LOOPED) - 1, 2)]
G, HD, H, HQ = tcfg.num_key_value_heads, tcfg.head_dim, tcfg.hidden_size, tcfg.num_attention_heads
ROT = int(HD * tcfg.rope_parameters.get("partial_rotary_factor", 1.0))  # rotary dims per head (64 of 256)
NOPE, HALF = HD - ROT, ROT // 2
RANK = CFG["r"] // 8 if TINY else CFG["r"]  # TINY: all dims scaled down by 8 (hidden 64 -> full rank = 64)
RRES = (CFG["res"] // 8 if TINY else CFG["res"]) if CFG["res"] else 0  # residual latent of looped visit 1 (config E)
RK = G * ROT if CFG["rope"] == "group" else ROT
REAL_G, REAL_ROT = 4, 64  # real-model dims, used for the byte accounting in both modes
log("full-attn", FULL, "looped", LOOPED, "pairs", PAIRS, "G", G, "HD", HD, "H", H, "ROT", ROT, "rank", RANK, "res", RRES, "RK", RK)
assert RANK % 16 == 0 and RANK <= H
assert RRES == 0 or (RRES % 16 == 0 and RRES <= H)
RES.update(full=FULL, looped=LOOPED, pairs=PAIRS, rank=RANK, res_rank=RRES, rot=ROT, rope_slots=RK)

ATT = {i: text_model.layers[i].self_attn for i in FULL}
for i in FULL:
    assert ATT[i].k_proj.bias is None and ATT[i].v_proj.bias is None
ROPE_FN = sys.modules[type(ATT[FULL[0]]).__module__].apply_rotary_pos_emb


def visits_of(L):
    return [0, 1] if L in LOOPED else [0]


def src_of(L, v):
    """Which stored latent slot (layer, visit) feeds this attention call."""
    if CFG["pair"]:
        for a, b in PAIRS:
            if L == b:
                return (a, 0)
    if CFG["share_loop"] and not CFG["res"] and L in LOOPED and v == 1:
        return (L, 0)
    return (L, v)


def lat_dim(L, v):
    """Width of the latent stored at slot (L, v)."""
    return RRES if (CFG["res"] and v == 1 and L in LOOPED) else RANK


SELF = [(L, v) for L in FULL for v in visits_of(L)]
SRC = {k: src_of(*k) for k in SELF}
OWNERS = [k for k in SELF if SRC[k] == k]
USERS = [k for k in SELF if SRC[k] != k]
# Residual visits (config E) own a latent AND read the visit-0 latent of the same layer as a base: k/v use [c0 ; c_res].
BASE = {(L, 1): (L, 0) for L in LOOPED} if CFG["res"] else {}
XREF = {**{u: SRC[u] for u in USERS}, **BASE}  # cross-covariance partners for calibration
log("slots (owners)", OWNERS, "users", USERS, "residual bases", BASE)

# ------------------------------------------------------------------ student modules

STATE = {"mode": "off", "stash": {}, "collect": {}, "losses": [], "probe": {}}
HAD = {}
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def hadamard(n, device):
    key = (n, str(device))
    if key not in HAD:
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        HAD[key] = (h / math.sqrt(n)).to(device)  # symmetric and orthogonal: H = H^T = H^-1
    return HAD[key]


def ste(x, q):
    return x + (q - x).detach()


@torch.no_grad()
def fp4_fake(x):
    """E2M1 values, one E4M3 scale per 16 along the last dim (NVFP4-style, no global scale)."""
    shp = x.shape
    g = x.reshape(-1, 16).float()
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float().clamp_min(2 ** -9)
    y = (g / scale).clamp(-6, 6)
    grid = E2M1.to(y.device)
    idx = (y.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    return (grid[idx] * torch.sign(y) * scale).reshape(shp)


@torch.no_grad()
def fp8_fake(x):
    """E4M3 with one scale per row (per token and group)."""
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 448.0).clamp_min(2 ** -9)
    return ((x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()) * scale


def quantize(c, kr):
    """c: [B,T,r] latent (r = RANK or RRES), kr: [B,T,RK] rope keys (pre-norm). Optional Hadamard + FP4 / FP8 fake quant (STE)."""
    if CFG["hadamard"]:
        Hm = hadamard(c.shape[-1], c.device)
        c = c @ Hm
    if CFG["qat"]:
        c = ste(c, fp4_fake(c))
        kr = ste(kr, fp8_fake(kr.reshape(*kr.shape[:-1], -1, ROT)).reshape(kr.shape))
    if CFG["hadamard"]:
        c = c @ Hm
    return c, kr


class StudentAttn(nn.Module):
    """New parameters of one full-attention layer: encoders for the slots it owns, up-projections per visit."""

    def __init__(self, L, dev):
        super().__init__()
        self.L = L
        self.enc_dkv = nn.ParameterDict()
        self.enc_kr = nn.ParameterDict()
        self.up_k = nn.ParameterDict()
        self.up_v = nn.ParameterDict()
        self.alpha = nn.ParameterDict()
        for v in visits_of(L):
            if SRC[(L, v)] == (L, v):
                self.enc_dkv[f"v{v}"] = nn.Parameter(torch.zeros(lat_dim(L, v), H, device=dev))
                self.enc_kr[f"v{v}"] = nn.Parameter(torch.zeros(RK, H, device=dev))
            cin = lat_dim(*SRC[(L, v)]) + (lat_dim(*BASE[(L, v)]) if (L, v) in BASE else 0)
            self.up_k[f"v{v}"] = nn.Parameter(torch.zeros(G, NOPE, cin, device=dev))
            self.up_v[f"v{v}"] = nn.Parameter(torch.zeros(G, HD, cin, device=dev))
            if CFG["rope"] == "shared":
                self.alpha[f"v{v}"] = nn.Parameter(torch.zeros(G, HALF, 2, device=dev))

    def run(self, x, cos, sin, attn, v, stash):
        """Student attention output (o_proj applied) for visit v; attn is the frozen teacher module."""
        B, T, _ = x.shape
        dev = x.device
        cos, sin = cos.to(dev), sin.to(dev)
        ac = torch.autocast("cuda", dtype=torch.float16) if ON_CUDA else nullcontext()
        with ac:
            src = SRC[(self.L, v)]
            if src == (self.L, v):
                c = F.linear(x, self.enc_dkv[f"v{v}"]).float()
                kr = F.linear(x, self.enc_kr[f"v{v}"]).float()
                c, kr = quantize(c, kr)
                stash[src] = (c, kr)
            else:
                c, kr = stash[src]
                c, kr = c.to(dev), kr.to(dev)
            if (self.L, v) in BASE:  # residual visit: up-project from [c0 ; c_res]
                c = torch.cat([stash[BASE[(self.L, v)]][0].to(dev), c], -1)
            cin = c.shape[-1]
            k_nope = F.linear(c, self.up_k[f"v{v}"].reshape(G * NOPE, cin)).float().view(B, T, G, NOPE)
            v_ = F.linear(c, self.up_v[f"v{v}"].reshape(G * HD, cin)).float().view(B, T, G, HD)
            if CFG["rope"] == "shared":
                al = self.alpha[f"v{v}"]
                a, b = al[..., 0], al[..., 1]  # complex scale per (group, frequency), pairs (i, i+HALF)
                kr4 = kr.reshape(B, T, 1, ROT)
                xr, xi = kr4[..., :HALF], kr4[..., HALF:]
                rope = torch.cat([a * xr - b * xi, b * xr + a * xi], -1)  # [B,T,G,ROT]
            else:
                rope = kr.reshape(B, T, G, ROT)
            # k_norm sees the teacher's layout: dims 0..ROT-1 rope part, ROT..HD-1 nope part
            k = attn.k_norm(torch.cat([rope, k_nope], -1)).transpose(1, 2)  # [B,G,T,HD]
            vv = v_.transpose(1, 2)
            qg = attn.q_proj(x).view(B, T, -1, 2 * HD)
            query, gate = torch.chunk(qg, 2, dim=-1)
            gate = gate.reshape(B, T, -1)
            query = attn.q_norm(query.reshape(B, T, -1, HD)).transpose(1, 2)  # [B,HQ,T,HD]
            query, k = ROPE_FN(query, k, cos, sin)
            dt = query.dtype
            k, vv = k.to(dt), vv.to(dt)
            rep = HQ // G
            k = k.repeat_interleave(rep, dim=1)
            vv = vv.repeat_interleave(rep, dim=1)
            o = F.scaled_dot_product_attention(query, k, vv, is_causal=True, scale=attn.scaling)
            o = o.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate)
            y = attn.o_proj(o)
        return y


class SplitAttn(nn.Module):
    """Replaces a layer's self_attn. The teacher output always goes into the residual stream (except in eval)."""

    def __init__(self, L, teacher, student):
        super().__init__()
        self.L = L
        self.teacher = teacher
        self.student = student
        self.counter = 0

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        v = self.counter
        self.counter += 1
        mode = STATE["mode"]
        if mode == "teacher":
            return self.teacher(hidden_states=hidden_states, position_embeddings=position_embeddings,
                                attention_mask=attention_mask, **kwargs)
        cos, sin = position_embeddings
        if mode == "eval":
            y = self.student.run(hidden_states, cos, sin, self.teacher, v, STATE["stash"])
            return y.to(hidden_states.dtype), None
        with torch.no_grad():
            yT, w = self.teacher(hidden_states=hidden_states, position_embeddings=position_embeddings,
                                 attention_mask=attention_mask, **kwargs)
        if mode == "collect":
            STATE["collect"][(self.L, v)] = hidden_states.detach()
            return yT, w
        yS = self.student.run(hidden_states.detach(), cos, sin, self.teacher, v, STATE["stash"]).float()
        yt = yT.float().to(yS.device)
        rel = (yS - yt).pow(2).sum() / yt.pow(2).sum().clamp_min(1e-12)
        if mode == "train":
            STATE["losses"].append(((self.L, v), rel))
        elif mode == "probe":
            STATE["probe"][(self.L, v)] = float(rel.detach().sqrt())
        return yT, w


# ------------------------------------------------------------------ data

class Chunks:
    """Cuts a stream of documents into SEQ-token chunks (EOS between documents)."""

    def __init__(self, docs):
        self.docs = docs
        self.buf = []

    def next(self):
        while len(self.buf) < SEQ:
            self.buf += tok(next(self.docs), add_special_tokens=False)["input_ids"] + [tok.eos_token_id or 0]
        out, self.buf = self.buf[:SEQ], self.buf[SEQ:]
        return torch.tensor(out)


def tiny_docs(alphabet, seed):
    rnd = random.Random(seed)
    while True:
        yield "".join(rnd.choice(alphabet) for _ in range(400))


def take(ch, n):
    return [ch.next() for _ in range(n)]


if TINY:
    WIKI = take(Chunks(tiny_docs("abcdefgh ij", 2)), N_EVAL)
    fw_docs = tiny_docs("klmnop qrs", 1)
    CALIB = take(Chunks(fw_docs), N_CALIB)
    FWEVAL = take(Chunks(fw_docs), N_EVAL)
    TRAIN = Chunks(fw_docs)
    rnd = random.Random(0)
    MMLU = [{"question": "".join(rnd.choice("abcdefg hij") for _ in range(30)),
             "choices": ["".join(rnd.choice("abcdefg hij") for _ in range(8)) for _ in range(4)],
             "answer": rnd.randrange(4)} for _ in range(N_MMLU)]
    PROMPTS = ["abc def ", "hij klm ", "nop qrs "]
else:
    from datasets import load_dataset
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    WIKI = take(Chunks(iter(["\n".join(wiki["text"])])), N_EVAL)
    fw_docs = (r["text"] for r in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
    CALIB = take(Chunks(fw_docs), N_CALIB)       # calibration: first documents
    FWEVAL = take(Chunks(fw_docs), N_EVAL)       # held-out fineweb-edu eval (disjoint from calibration)
    TRAIN = Chunks(fw_docs)                      # training stream continues after eval
    mm = load_dataset("cais/mmlu", "all", split="test")
    MMLU = [mm[i] for i in random.Random(0).sample(range(len(mm)), N_MMLU)]
    PROMPTS = ["The capital of France is", "def fibonacci(n):", "Photosynthesis is the process by which"]
log(f"data: calib {len(CALIB)}, wiki {len(WIKI)}, fineweb eval {len(FWEVAL)} x {SEQ}, mmlu {len(MMLU)}")
ANS_IDS = [tok(" " + L_, add_special_tokens=False)["input_ids"][-1] for L_ in "ABCD"]

dev0 = next(model.parameters()).device


def reset_visits(*_):
    for m in SPLITS.values():
        m.counter = 0
    STATE["stash"] = {}


SPLITS = {}
text_model.register_forward_pre_hook(lambda m, a: reset_visits())

STU = nn.ModuleDict()
for L in FULL:
    layer = text_model.layers[L]
    teacher = layer.self_attn
    dev = teacher.q_proj.weight.device
    stu = StudentAttn(L, dev)
    wrap = SplitAttn(L, teacher, stu)
    layer.self_attn = wrap
    SPLITS[L] = wrap
    STU[str(L)] = stu
    del teacher
for m in STU.parameters():
    m.requires_grad_(True)
N_STU = sum(p.numel() for p in STU.parameters())
log(f"student params {N_STU / 1e6:.1f}M, trainable tensors {len(list(STU.parameters()))}")


# ------------------------------------------------------------------ evaluation helpers

@torch.no_grad()
def nll(seqs):
    tot, cnt = 0.0, 0
    for s in seqs:
        ids = s.unsqueeze(0).to(dev0)
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
        tgt = ids[0, 1:].to(h.device)
        for c0 in range(0, h.shape[0], 256):
            logits = model.lm_head(h[c0:c0 + 256]).float()
            tot += F.cross_entropy(logits.to(tgt.device), tgt[c0:c0 + 256], reduction="sum").item()
        cnt += SEQ - 1
    return tot / cnt


def mmlu_prompt(it):
    s = ("The following is a multiple choice question. Answer with the letter of the correct choice.\n\n"
         + str(it["question"]).strip() + "\n")
    for L_, c in zip("ABCD", it["choices"]):
        s += f"{L_}. {str(c).strip()}\n"
    return s + "Answer:"


@torch.no_grad()
def mmlu(mode):
    STATE["mode"] = mode
    ok = 0
    for it in MMLU:
        ids = torch.tensor([tok(mmlu_prompt(it), add_special_tokens=False)["input_ids"][-1024:]]).to(dev0)
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
        logits = model.lm_head(h.to(model.lm_head.weight.device)).float()
        pred = int(logits[torch.tensor(ANS_IDS, device=logits.device)].argmax())
        ok += int(pred == int(it["answer"]))
    return ok / len(MMLU)


@torch.no_grad()
def greedy(mode, prompt):
    STATE["mode"] = mode
    ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]]).to(dev0)
    p = ids.shape[1]
    for _ in range(GEN_TOK):
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
        nxt = model.lm_head(h.to(model.lm_head.weight.device)).argmax(-1)
        ids = torch.cat([ids, nxt.view(1, 1).to(ids.device)], 1)
    new = ids[0, p:].tolist()
    return new, (tok.decode(new) if hasattr(tok, "decode") else new)


@torch.no_grad()
def probe(seqs):
    """Per (layer, visit) relative output error ||y_S - y_T|| / ||y_T|| (max over sequences)."""
    STATE["mode"] = "probe"
    out = {}
    for s in seqs:
        STATE["probe"] = {}
        model.model(input_ids=s.unsqueeze(0).to(dev0), use_cache=False)
        for k, e in STATE["probe"].items():
            out[k] = max(out.get(k, 0.0), e)
    STATE["mode"] = "off"
    return {f"{L}.{v}": e for (L, v), e in sorted(out.items())}


def kv_bytes_per_token():
    """Sum over stored slots (real-model dims): latent + its rope key. Each slot has its own width (E: 8 x 512, 6 x 128)."""
    rk_real = REAL_G * REAL_ROT if CFG["rope"] == "group" else REAL_ROT
    tot = 0.0
    for (L, v) in OWNERS:
        r_real = (CFG["res"] if (CFG["res"] and v == 1 and L in LOOPED) else CFG["r"])
        if CFG["qat"]:
            lat = r_real * 0.5 + r_real / 16              # FP4 values + E4M3 scale per 16
            rope = rk_real * 1 + (rk_real // REAL_ROT) * 2  # FP8 keys + 2-byte scale per (token, group)
        else:
            lat = 2 * r_real                              # bf16
            rope = 2 * rk_real
        tot += lat + rope
    return int(tot)


# ------------------------------------------------------------------ calibration statistics

COV = {k: torch.zeros(H, H, dtype=torch.float64) for k in SELF}
XC = {k: torch.zeros(H, H, dtype=torch.float64) for k in XREF}
NTOK = 0
STATE["mode"] = "collect"
t_cal = time.time()
for si, s in enumerate(CALIB):
    STATE["collect"] = {}
    with torch.no_grad():
        model.model(input_ids=s.unsqueeze(0).to(dev0), use_cache=False)
    X = {k: STATE["collect"][k].reshape(-1, H).float().to(dev0) for k in SELF}
    for k in SELF:
        COV[k] += (X[k].T @ X[k]).double().cpu()
    for u, ref in XREF.items():
        XC[u] += (X[u].T @ X[ref]).double().cpu()
    NTOK += X[SELF[0]].shape[0]
    del X
STATE["mode"] = "off"
STATE["collect"] = {}
for k in SELF:
    COV[k] /= NTOK
for u in XREF:
    XC[u] /= NTOK
log(f"calibration: {len(CALIB)} seqs, {NTOK} tokens, {time.time() - t_cal:.0f}s")


# ------------------------------------------------------------------ activation-aware SVD init

def sqrt_inv(C):
    """S = Q sqrt(L) Q^T and S^-1 with eigenvalue floor 1e-6 * max."""
    ev, Q = torch.linalg.eigh(C)
    lam = ev.clamp_min(1e-6 * ev.max())
    s = lam.sqrt()
    return (Q * s) @ Q.T, (Q / s) @ Q.T


def map_teacher(L):
    """Teacher map M = [nope rows of K per group ; V rows] (double, CPU) and the rope rows of K per group."""
    att = ATT[L]
    Wk = att.k_proj.weight.detach().double().cpu().reshape(G, HD, H)
    Wv = att.v_proj.weight.detach().double().cpu().reshape(G * HD, H)
    return torch.cat([Wk[:, ROT:, :].reshape(G * NOPE, H), Wv], 0), Wk[:, :ROT, :]


def fit_alpha(rows, Wsrc, Cxs, Cs):
    """Complex per-frequency scale alpha[g,i] minimising E|z_target - alpha z_src|^2 (closed form from covariances).
    rows: [G,ROT,H] target rope rows; Wsrc: [ROT,H] source rope rows; Cxs = E[x_target x_src^T]; Cs = E[x_src x_src^T]."""
    zt = torch.complex(rows[:, :HALF], rows[:, HALF:])
    zs = torch.complex(Wsrc[:HALF], Wsrc[HALF:])
    num = ((zt @ Cxs.to(torch.complex128)) * zs.conj()).sum(-1)
    den = ((zs @ Cs.to(torch.complex128)) * zs.conj()).sum(-1).real
    al = num / den
    return torch.stack([al.real, al.imag], -1)


def set_(p, t):
    with torch.no_grad():
        p.copy_(t.to(device=p.device, dtype=p.dtype))


INIT_METHOD = {}
t_init = time.time()
def ridge(A, eps=1e-6):
    return A + eps * A.diagonal().mean() * torch.eye(A.shape[0], dtype=A.dtype)


for (L, v) in OWNERS:
    S, Sinv = sqrt_inv(COV[(L, v)])
    M, rows = map_teacher(L)
    st = STU[str(L)]
    if (L, v) in BASE:
        # Residual visit (E). Target t = M x1. Stage 1: W1 = least squares from the visit-0 latent c0 = Wd0 x0,
        # W1 = E[t c0^T] E[c0 c0^T]^-1 with E[x1 x0^T] = XC. Stage 2: the best linear-in-x1 residual of that fit is
        # M_res = M - W1 Wd0 Q, Q = E[x0 x1^T] E[x1 x1^T]^-1; factor it by activation-aware SVD: c_res = Wdres x1,
        # W_ukres = U_r Sigma_r. Up-projection of [c0 ; c_res] is [W1 | W_ukres].
        b = BASE[(L, v)]
        Wd0 = st.enc_dkv[f"v{b[1]}"].detach().double().cpu()
        Cb, Cx, X10 = COV[b], COV[(L, v)], XC[(L, v)]
        W1 = torch.linalg.solve(ridge(Wd0 @ Cb @ Wd0.T), (M @ X10 @ Wd0.T).T).T
        Q = torch.linalg.solve(ridge(Cx), X10).T
        Mres = M - W1 @ Wd0 @ Q
        S1, S1inv = sqrt_inv(Cx)
        U, Sig, Vh = torch.linalg.svd(Mres @ S1, full_matrices=False)
        Wdkv = Vh[:RRES] @ S1inv
        Wup = torch.cat([W1, U[:, :RRES] * Sig[:RRES]], -1)
        method = f"residual: LS from base {b[0]}.{b[1]} + activation-aware SVD of residual"
    else:
        U, Sig, Vh = torch.linalg.svd(M @ S, full_matrices=False)
        Wdkv = Vh[:RANK] @ Sinv
        Wup = U[:, :RANK] * Sig[:RANK]
        method = "activation-aware SVD (owner)"
    if CFG["rope"] == "group":
        Wkr = rows.reshape(G * ROT, H)
        alpha = None
    else:
        U2, S2, Vh2 = torch.linalg.svd(rows.reshape(G * ROT, H) @ S, full_matrices=False)
        Wkr = Vh2[:ROT] @ Sinv
        Rrec = ((U2[:, :ROT] * S2[:ROT]) @ Wkr).reshape(G, ROT, H)
        alpha = fit_alpha(Rrec, Wkr, COV[(L, v)], COV[(L, v)])
    set_(st.enc_dkv[f"v{v}"], Wdkv)
    set_(st.enc_kr[f"v{v}"], Wkr)
    set_(st.up_k[f"v{v}"], Wup[:G * NOPE].reshape(G, NOPE, -1))
    set_(st.up_v[f"v{v}"], Wup[G * NOPE:].reshape(G, HD, -1))
    if alpha is not None:
        set_(st.alpha[f"v{v}"], alpha)
    INIT_METHOD[f"{L}.{v}"] = method
for (L, v) in USERS:
    Ls, vs = SRC[(L, v)]
    Cs, Cxs = COV[(Ls, vs)], XC[(L, v)]
    su = STU[str(Ls)]
    Wd = su.enc_dkv[f"v{vs}"].detach().double().cpu()
    M, rows = map_teacher(L)
    Gm = Wd @ Cs @ Wd.T
    Gm = Gm + 1e-6 * Gm.diagonal().mean() * torch.eye(RANK, dtype=Gm.dtype)
    Wup = torch.linalg.solve(Gm, (M @ Cxs @ Wd.T).T).T
    st = STU[str(L)]
    set_(st.up_k[f"v{v}"], Wup[:G * NOPE].reshape(G, NOPE, RANK))
    set_(st.up_v[f"v{v}"], Wup[G * NOPE:].reshape(G, HD, RANK))
    if CFG["rope"] == "shared":
        Wkr_src = su.enc_kr[f"v{vs}"].detach().double().cpu()
        set_(st.alpha[f"v{v}"], fit_alpha(rows, Wkr_src, Cxs, Cs))
    INIT_METHOD[f"{L}.{v}"] = f"least squares from slot {Ls}.{vs} (cross-covariance)"
log(f"init done in {time.time() - t_init:.0f}s")
RES["init_method"] = INIT_METHOD
RES["kv_bytes_per_token"] = kv_bytes_per_token()
RES["slots"] = len(OWNERS)
log("kv bytes/token (real dims):", RES["kv_bytes_per_token"], "slots", len(OWNERS))

init_probe = probe(FWEVAL[:2])
RES["init_probe_rel_err"] = init_probe
log("init per-layer relative output error:", json.dumps(init_probe))
if CFG_NAME == "C" and TINY:
    worst = max(init_probe.values())
    RES["exactness_C_max_rel_err"] = worst
    log(f"EXACTNESS CHECK config C (full rank, no QAT): max relative attention-output error {worst:.3e}")
    assert worst < 1e-4, "config C does not reproduce the teacher attention output"
save()

# ------------------------------------------------------------------ stage A: layer-local training

params = list(STU.parameters())
groups = [{"params": [q], "scale": max(q.detach().float().pow(2).mean().sqrt().item(), 1e-3)} for q in params]
opt = torch.optim.AdamW(groups, lr=LR_MAX, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", enabled=ON_CUDA)
TRAIN_BUDGET_S = TRAIN_HOURS * 3600
t_train0 = time.time()
step, tokens = 0, 0
run_loss, run_layer, loss_log, layer_hist = 0.0, {}, [], {}
while True:
    progress = (step / TRAIN_STEPS) if TINY else ((time.time() - t_train0) / TRAIN_BUDGET_S)
    if progress >= 1.0:
        break
    lr = min(1.0, (step + 1) / WARM) * (LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))
    for g_ in opt.param_groups:
        g_["lr"] = lr * g_["scale"]
    ids = TRAIN.next().unsqueeze(0).to(dev0)
    STATE["mode"], STATE["losses"] = "train", []
    model.model(input_ids=ids, use_cache=False)  # only the student branches build graphs
    loss = sum(r.to(dev0) for _, r in STATE["losses"])  # per-layer terms live on two GPUs
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    for k, r in STATE["losses"]:
        layer_hist.setdefault(f"{k[0]}.{k[1]}", []).append(float(r.detach()))
    run_loss += float(loss.detach())
    tokens += SEQ
    step += 1
    if step % 20 == 0:
        mean = run_loss / 20
        loss_log.append({"step": step, "tokens": tokens, "loss_mean_per_visit_sum": mean, "lr": lr,
                         "elapsed_s": time.time() - t_train0})
        log(f"train step {step} tokens {tokens} loss(sum over visits) {mean:.4f} lr {lr:.2e}")
        run_loss = 0.0
        save()
    STATE["losses"] = []
STATE["mode"] = "off"
log(f"training: {step} steps, {tokens} tokens, {time.time() - t_train0:.0f}s")
RES.update(train_steps=step, train_tokens=tokens, train_seconds=time.time() - t_train0, loss_log=loss_log,
           train_last20_per_layer_rel={k: sum(v_[-20:]) / len(v_[-20:]) for k, v_ in layer_hist.items() if v_})

final_probe = probe(FWEVAL[:2])
RES["final_probe_rel_err"] = final_probe
log("final per-layer relative output error:", json.dumps(final_probe))
torch.save({k: v_.detach().cpu() for k, v_ in STU.state_dict().items()}, f"{OUT}/student_{CFG_NAME}.pt")
save()

# ------------------------------------------------------------------ end-to-end evaluation

t_ev = time.time()
RES["ppl"] = {}
for mode, tag in (("teacher", "teacher"), ("eval", "student")):
    STATE["mode"] = mode
    with torch.no_grad():
        w = nll(WIKI)
        f = nll(FWEVAL)
    RES["ppl"][tag] = {"wiki_nll": w, "wiki_ppl": math.exp(w), "fineweb_nll": f, "fineweb_ppl": math.exp(f)}
    log(tag, "ppl", json.dumps(RES["ppl"][tag]))
    save()
RES["mmlu"] = {}
for mode, tag in (("teacher", "teacher"), ("eval", "student")):
    RES["mmlu"][tag] = {"n": len(MMLU), "acc": mmlu(mode)}
    log("mmlu", tag, RES["mmlu"][tag])
RES["generations"] = {}
for mode, tag in (("teacher", "teacher"), ("eval", "student")):
    RES["generations"][tag] = []
    for p in PROMPTS:
        ids, txt = greedy(mode, p)
        RES["generations"][tag].append({"prompt": p, "ids": ids, "text": txt})
        log("gen", tag, repr(p), "->", repr(txt)[:200])
STATE["mode"] = "off"
RES["eval_seconds"] = time.time() - t_ev
RES["total_seconds"] = time.time() - T0
save()
log("done")

# Streaming-dataset worker threads can keep the interpreter alive after the script ends (seen on Kaggle:
# distill A printed "done" at 3.6 h, then the session stayed RUNNING for hours). Exit hard after flushing.
import sys as _sys, os as _os
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)
