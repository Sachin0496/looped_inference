# SenseNovaKV stage B: END-TO-END logit distillation of the config-E compressed KV student on a Kaggle TPU v5e-8 (JAX).
# Teacher = SenseNova-Loop-9B release weights (BF16 GGUF converted in memory). Student = teacher with every full-attention
# K/V path replaced by the config-E latent (3,660 B/token), initialised from the layer-local result (student_E.pt,
# output of the sensenovakv-distill-e kernel). Loss = KL(p_teacher || p_student) over the full vocabulary, every token.
# Only the student KV tensors train (teacher weights frozen). Sources inlined by kaggle/render.py (job kd_tpu).
# SNKV_TINY=1: CPU smoke test on a tiny random model (no downloads beyond the Preview config), see tests/.
import sys as _sys0, os as _os0, traceback as _tb0
def _hard_exit_on_error(et, ev, tb):  # streaming-dataset threads keep the process alive after an exception
    _tb0.print_exception(et, ev, tb); _sys0.stdout.flush(); _sys0.stderr.flush(); _os0._exit(1)
_sys0.excepthook = _hard_exit_on_error

import glob, json, math, os, queue, random, subprocess, sys, threading, time

T0 = time.time()
TINY = os.environ.get("SNKV_TINY") == "1"
OUT = os.environ.get("SNKV_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else "./out_kd")
os.makedirs(OUT, exist_ok=True)
SEQ = 128 if TINY else 2048
BATCH = int(os.environ.get("SNKV_BATCH", 2 if TINY else 8))                 # global sequences per step (multiple of #devices)
TRAIN_HOURS = float(os.environ.get("SNKV_TRAIN_HOURS", 5.0))
TINY_STEPS = int(os.environ.get("SNKV_STEPS", 6))
LR_REL = float(os.environ.get("SNKV_LR", 1e-3))               # per-tensor RELATIVE lr: lr_t = LR_REL * rms(init_t)
WARM = int(os.environ.get("SNKV_WARM", 100))
N_EVAL = int(os.environ.get("SNKV_N_EVAL", 4 if TINY else 32))
N_MMLU = int(os.environ.get("SNKV_N_MMLU", 16 if TINY else 300))
N_CALIB_SKIP = 64   # distill.py used the first 64 fineweb chunks for calibration, then 32 for eval: keep the same eval set
CKPT_EVERY_S = 1800
RES = {"job": "kd_tpu", "config": "E", "tiny": TINY, "seq": SEQ, "batch": BATCH, "lr_rel": LR_REL, "log": []}


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


SRC_DIR = "/tmp/snkv_src"
os.makedirs(SRC_DIR, exist_ok=True)
for fname, src in (("gguf_to_hf.py", r'''__CONVERTER_SOURCE__'''), ("jax_model.py", r'''__JAX_MODEL_SOURCE__'''),
                   ("jax_student.py", r'''__JAX_STUDENT_SOURCE__''')):
    if not src.startswith("__"):
        open(os.path.join(SRC_DIR, fname), "w").write(src)
if TINY:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
else:
    sys.path.insert(0, SRC_DIR)
    sh("free -g | head -2; nproc; df -h / /tmp /kaggle/working 2>/dev/null; ls /kaggle/input 2>/dev/null")
    sh(f"{sys.executable} -m pip list 2>/dev/null | grep -i -E '^(jax|jaxlib|libtpu|torch|transformers|datasets) ' || true")
    sh(f"{sys.executable} -m pip install -q 'transformers==5.18.0' datasets gguf 'huggingface_hub>=0.30' 2>&1 | tail -2")
    _probe = subprocess.run([sys.executable, "-c", "import jax; print(jax.devices()[0].platform)"], capture_output=True, text=True)
    if "tpu" not in _probe.stdout:
        log("jax TPU backend missing:", _probe.stdout[-300:], _probe.stderr[-500:])
        sh(f"{sys.executable} -m pip install -q -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>&1 | tail -2")
    try:
        import torch  # noqa: F401  (only to read student_E.pt)
    except ImportError:
        sh(f"{sys.executable} -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2")

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from tpu import jax_model as JM, jax_student as JS  # local / TINY
except ImportError:
    import jax_model as JM
    import jax_student as JS

DEVS = jax.devices()
NDEV = len(DEVS)
MESH = Mesh(np.array(DEVS), ("d",))
REPL = NamedSharding(MESH, P())
BSHARD = NamedSharding(MESH, P("d"))
log("jax", jax.__version__, "devices", NDEV, DEVS[0].platform, DEVS[0].device_kind if hasattr(DEVS[0], "device_kind") else "")
assert BATCH % NDEV == 0, (BATCH, NDEV)
RES["devices"] = f"{NDEV} x {getattr(DEVS[0], 'device_kind', DEVS[0].platform)}"
CDTYPE = jnp.float32 if TINY else jnp.bfloat16


def constrain(x):
    return jax.lax.with_sharding_constraint(x, BSHARD) if x.shape[0] % NDEV == 0 else x


def shard_spec(x):
    """FSDP: shard the largest dimension divisible by the device count; small tensors are replicated."""
    if NDEV == 1 or x.size < (1 << 20):
        return REPL
    dims = [d for d in range(x.ndim) if x.shape[d] % NDEV == 0]
    if not dims:
        return REPL
    best = max(dims, key=lambda d: x.shape[d])
    spec = [None] * x.ndim
    spec[best] = "d"
    return NamedSharding(MESH, P(*spec))


# ------------------------------------------------------------------ weights

PREVIEW_REPO = "sensenova/SenseNova-Loop-9B-Preview"
GGUF_REPO, GGUF_FILE = "sensenova/SenseNova-Loop-9B-GGUF", "SenseNova-Loop-9B-BF16.gguf"
META_DIR = "/tmp/preview_meta"


def tiny_setup():
    """Tiny random model + numpy state dict (as in tests/test_jax_model.py)."""
    import torch
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, os.environ["SNKV_SN_SRC"])
    from sensenova_loop.models import register
    from sensenova_loop.models.modeling_sensenova_loop import SenseNovaLoopConfig
    from transformers import AutoModelForCausalLM
    register()
    raw = json.load(open(hf_hub_download(PREVIEW_REPO, "config.json")))
    t = raw["text_config"]
    t.update(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
             linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
             vocab_size=512, mtp_num_hidden_layers=0, pad_token_id=0, eos_token_id=1)
    raw.update(pad_token_id=0, eos_token_id=1, image_token_id=500, video_token_id=501, vision_start_token_id=502,
               vision_end_token_id=503)
    raw["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    raw.pop("auto_map", None)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(SenseNovaLoopConfig(**raw), dtype=torch.float32)
    with torch.no_grad():
        for p_ in m.parameters():
            p_.add_(0.05 * torch.randn_like(p_))
    sd = {k: v.detach().numpy().copy() for k, v in m.state_dict().items() if not k.startswith("model.visual.")}
    return raw, sd


def release_setup():
    """Release BF16 GGUF -> HF-named numpy tensors in memory (gguf_to_hf.py ops; no safetensors written)."""
    from huggingface_hub import HfApi, hf_hub_download
    import gguf_to_hf as C
    os.makedirs(META_DIR, exist_ok=True)
    names = [f for f in HfApi().list_repo_files(PREVIEW_REPO)
             if "/" not in f and f.endswith((".json", ".jinja", ".txt", ".model", ".tiktoken"))]
    for f in names:
        hf_hub_download(PREVIEW_REPO, f, local_dir=META_DIR)
    t = time.time()
    gpath = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir="/tmp/release_src")
    log(f"gguf {os.path.getsize(gpath) / 1e9:.2f} GB in {time.time() - t:.0f}s")
    header = C.read_gguf_header(gpath, require_data=True)
    config = json.load(open(os.path.join(META_DIR, "config.json")))
    index = C.load_preview_index(os.path.join(META_DIR, "model.safetensors.index.json"))
    preview = C.load_preview_shapes(None, index)
    outputs, consumed, problems = C.build_plan(header, config, preview, True)
    assert not problems, problems[:5]
    out_cfg, notes = C.build_out_config(config, header, True, False)
    log("config notes:", notes)
    gg, dims = C.GGUFData(gpath, header), C.Dims(config)
    import ml_dtypes
    sd = {}
    for spec in outputs:
        if spec.hf.startswith(("model.visual.", "mtp.")):
            continue
        a = C.produce(spec, gg, dims)
        sd[spec.hf] = a.view(ml_dtypes.bfloat16) if a.dtype == np.uint16 else a
    log(f"converted {len(sd)} tensors in {time.time() - t:.0f}s")
    return out_cfg, sd


if TINY:
    RAW_CFG, SD = tiny_setup()
else:
    RAW_CFG, SD = release_setup()
    RES["weights"] = "release-gguf-converted"
CFG = JM.ModelCfg.from_hf_config(RAW_CFG)
log("cfg", CFG)
t = time.time()
with jax.default_device(jax.devices("cpu")[0]):
    host_params = JM.params_from_state_dict(SD, CFG, CDTYPE)
del SD
PARAMS = jax.tree_util.tree_map(lambda x: jax.device_put(x, shard_spec(x)), host_params)
del host_params
jax.block_until_ready(PARAMS)
n_par = sum(x.size for x in jax.tree_util.tree_leaves(PARAMS))
log(f"params on device: {n_par / 1e9:.2f} B in {time.time() - t:.0f}s")

# ------------------------------------------------------------------ student (config E)

if TINY:
    STU_HOST = None  # random-small student built from the tiny teacher (see tests); here: teacher-exact init
else:
    cand = sorted(glob.glob("/kaggle/input/**/student_E.pt", recursive=True))
    assert cand, "student_E.pt not found under /kaggle/input (attach kernel sachin0496/sensenovakv-distill-e)"
    log("student init from", cand[0])
    STU_HOST = JS.student_tree(JS.load_student_pt(cand[0]))


def tiny_student():
    """Exact low-rank-free student for the tiny model: latent = full K/V input dims (smoke test only)."""
    rng = np.random.default_rng(0)
    H, G, HD, ROT = CFG.hidden, CFG.n_kv, CFG.head_dim, CFG.rot_dim
    NOPE, HALF = HD - ROT, ROT // 2

    def sp(cin, r):
        return {"enc_dkv": rng.standard_normal((r, H)).astype(np.float32) * 0.1,
                "enc_kr": rng.standard_normal((ROT, H)).astype(np.float32) * 0.1,
                "up_k": rng.standard_normal((G, NOPE, cin)).astype(np.float32) * 0.1,
                "up_v": rng.standard_normal((G, HD, cin)).astype(np.float32) * 0.1,
                "alpha": rng.standard_normal((G, HALF, 2)).astype(np.float32)}
    def st(cin, r):
        many = [sp(cin, r) for _ in range(6)]
        return {k: np.stack([m[k] for m in many]) for k in many[0]}
    return {"pre": sp(64, 64), "loop0": st(64, 64), "loop1": st(80, 16), "post": sp(64, 64)}


STU = jax.device_put(jax.tree_util.tree_map(jnp.asarray, tiny_student() if TINY else STU_HOST), REPL)
SCALES = jax.tree_util.tree_map(lambda x: float(jnp.sqrt(jnp.mean(jnp.square(x)))) + 1e-8, STU)
log(f"student params: {sum(x.size for x in jax.tree_util.tree_leaves(STU)) / 1e6:.1f}M")
RES["kv_bytes_per_token"] = JS.kv_bytes_per_token_E()


def kv_fn(x, sp, k_norm_w, base_c):
    return JS.student_kv(x, sp, k_norm_w, base_c)


def hidden(params, ids, stu=None, remat=False):
    if stu is None:
        return JM.forward_hidden(params, ids, CFG, remat=remat, constrain=constrain)
    return JM.forward_hidden(params, ids, CFG, student=stu, kv_fn=kv_fn, remat=remat, constrain=constrain)


# ------------------------------------------------------------------ losses (vocab-sharded logits, T-chunked)

CH = 128 if TINY else 256
VSHARD = NamedSharding(MESH, P(None, None, "d"))


def _chunks(x):
    B, T = x.shape[:2]
    return jnp.swapaxes(x.reshape(B, T // CH, CH, *x.shape[2:]), 0, 1)


def kd_stats(W, hT, hS, labels, mask):
    """Sums over tokens of KL(pT||pS), student NLL, teacher NLL; and the token count. Logits never fully materialised."""
    hT = jax.lax.with_sharding_constraint(hT, REPL)
    hS = jax.lax.with_sharding_constraint(hS, REPL)
    V = W.shape[0]

    @jax.checkpoint
    def body(carry, xs):
        ht, hs, lab, m = xs
        lt = jax.lax.with_sharding_constraint(jnp.einsum("bth,vh->btv", ht, W, preferred_element_type=jnp.float32), VSHARD)
        ls = jax.lax.with_sharding_constraint(jnp.einsum("bth,vh->btv", hs, W, preferred_element_type=jnp.float32), VSHARD)
        lpt = jax.nn.log_softmax(lt, -1)
        lps = jax.nn.log_softmax(ls, -1)
        kl = jnp.sum(jnp.exp(lpt) * (lpt - lps), -1)
        hit = jnp.arange(V)[None, None, :] == lab[..., None]
        nll_s = -jnp.sum(jnp.where(hit, lps, 0.0), -1)
        nll_t = -jnp.sum(jnp.where(hit, lpt, 0.0), -1)
        c = carry
        return (c[0] + jnp.sum(kl), c[1] + jnp.sum(nll_s * m), c[2] + jnp.sum(nll_t * m), c[3] + jnp.sum(m)), None

    z = jnp.zeros((), jnp.float32)
    (kl, ns, nt, cnt), _ = jax.lax.scan(body, (z, z, z, z), (_chunks(hT), _chunks(hS), _chunks(labels), _chunks(mask)))
    return kl, ns, nt, cnt


def labels_of(ids):
    lab = jnp.concatenate([ids[:, 1:], jnp.zeros_like(ids[:, :1])], 1)
    mask = jnp.concatenate([jnp.ones_like(ids[:, 1:], jnp.float32), jnp.zeros_like(ids[:, :1], jnp.float32)], 1)
    return lab, mask


def loss_fn(stu, params, ids):
    hT = jax.lax.stop_gradient(hidden(params, ids))
    hS = hidden(params, ids, stu, remat=True)
    lab, mask = labels_of(ids)
    kl, ns, nt, cnt = kd_stats(params["lm_head"], hT, hS, lab, mask)
    ntok = ids.shape[0] * ids.shape[1]
    return kl / ntok, (kl / ntok, ns / cnt, nt / cnt)


B1, B2, EPS = 0.9, 0.999, 1e-8


@jax.jit
def train_step(stu, opt, params, ids, lr):
    (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(stu, params, ids)
    gn = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(g)))
    g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, 1.0 / (gn + 1e-6)), g)
    t = opt["t"] + 1
    m = jax.tree_util.tree_map(lambda a, b: B1 * a + (1 - B1) * b, opt["m"], g)
    v = jax.tree_util.tree_map(lambda a, b: B2 * a + (1 - B2) * b * b, opt["v"], g)
    c1, c2 = 1 - B1 ** t, 1 - B2 ** t
    stu = jax.tree_util.tree_map(lambda p, a, b, s: p - lr * s * (a / c1) / (jnp.sqrt(b / c2) + EPS), stu, m, v, SCALES)
    return stu, {"t": t, "m": m, "v": v}, aux, gn


@jax.jit
def eval_nll(params, stu, ids):
    """(student NLL sum, teacher NLL sum, KL sum, count) on a batch; stu=None not allowed under jit, see eval_teacher."""
    hT = hidden(params, ids)
    hS = hidden(params, ids, stu)
    lab, mask = labels_of(ids)
    kl, ns, nt, cnt = kd_stats(params["lm_head"], hT, hS, lab, mask)
    return ns, nt, kl, cnt


@jax.jit
def last_logits(params, stu, ids, pos, ans):
    """Teacher and student logits over the answer tokens at position pos[b] (MMLU)."""
    hT = hidden(params, ids)
    hS = hidden(params, ids, stu)
    W = params["lm_head"][ans]
    b = jnp.arange(ids.shape[0])
    return (jnp.einsum("bh,vh->bv", hT[b, pos], W, preferred_element_type=jnp.float32),
            jnp.einsum("bh,vh->bv", hS[b, pos], W, preferred_element_type=jnp.float32))


# ------------------------------------------------------------------ data

if TINY:
    class _Tok:
        eos_token_id = 1

        def __call__(self, t, add_special_tokens=False):
            return {"input_ids": [ord(ch) % 500 + 2 for ch in t]}
    TOK = _Tok()
else:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(META_DIR)


class Chunks:
    def __init__(self, docs):
        self.docs, self.buf = docs, []

    def next(self):
        while len(self.buf) < SEQ:
            self.buf += TOK(next(self.docs), add_special_tokens=False)["input_ids"] + [TOK.eos_token_id or 0]
        out, self.buf = self.buf[:SEQ], self.buf[SEQ:]
        return np.asarray(out, np.int32)


def take(ch, n):
    return [ch.next() for _ in range(n)]


def tiny_docs(alphabet, seed):
    rnd = random.Random(seed)
    while True:
        yield "".join(rnd.choice(alphabet) for _ in range(400))


if TINY:
    WIKI = take(Chunks(tiny_docs("abcdefgh ij", 2)), N_EVAL)
    fw = Chunks(tiny_docs("klmnop qrs", 1))
    take(fw, 4)
    FWEVAL = take(fw, N_EVAL)
    rnd = random.Random(0)
    MMLU = [{"question": "".join(rnd.choice("abcdefg hij") for _ in range(30)),
             "choices": ["".join(rnd.choice("abcdefg hij") for _ in range(8)) for _ in range(4)],
             "answer": rnd.randrange(4)} for _ in range(N_MMLU)]
else:
    from datasets import load_dataset
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    WIKI = take(Chunks(iter(["\n".join(wiki["text"])])), N_EVAL)
    fw = Chunks(r["text"] for r in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
    take(fw, N_CALIB_SKIP)                 # distill.py calibration chunks (not used here)
    FWEVAL = take(fw, N_EVAL)              # same held-out fineweb-edu eval chunks as distill.py
    mm = load_dataset("cais/mmlu", "all", split="test")
    MMLU = [mm[i] for i in random.Random(0).sample(range(len(mm)), N_MMLU)]
ANS = np.asarray([TOK(" " + L_, add_special_tokens=False)["input_ids"][-1] for L_ in "ABCD"], np.int32)
log(f"data: wiki {len(WIKI)}, fineweb eval {len(FWEVAL)} x {SEQ}, mmlu {len(MMLU)}")

BATCHQ = queue.Queue(maxsize=8)


def producer():
    while True:
        BATCHQ.put(np.stack(take(fw, BATCH)))


threading.Thread(target=producer, daemon=True).start()


def put(x):
    return jax.device_put(x, BSHARD)


def ppl_eval(seqs):
    ns = nt = kl = cnt = 0.0
    for i in range(0, len(seqs), BATCH):
        blk = seqs[i:i + BATCH]
        assert len(blk) == BATCH, "eval set size must be a multiple of the batch"
        out = eval_nll(PARAMS, STU, put(np.stack(blk)))
        ns, nt, kl, cnt = ns + float(out[0]), nt + float(out[1]), kl + float(out[2]), cnt + float(out[3])
    return {"student_nll": ns / cnt, "student_ppl": math.exp(ns / cnt), "teacher_nll": nt / cnt,
            "teacher_ppl": math.exp(nt / cnt), "kl_per_token": kl / (cnt * SEQ / (SEQ - 1))}


def mmlu_prompt(it):
    s = ("The following is a multiple choice question. Answer with the letter of the correct choice.\n\n"
         + str(it["question"]).strip() + "\n")
    for L_, c in zip("ABCD", it["choices"]):
        s += f"{L_}. {str(c).strip()}\n"
    return s + "Answer:"


def mmlu_eval():
    L = 128 if TINY else 1024
    okT = okS = 0
    items = list(MMLU)
    for i in range(0, len(items), BATCH):
        blk = items[i:i + BATCH]
        ids = np.zeros((BATCH, L), np.int32)
        pos = np.zeros(BATCH, np.int32)
        for b, it in enumerate(blk):
            tk = TOK(mmlu_prompt(it), add_special_tokens=False)["input_ids"][-L:]
            ids[b, :len(tk)] = tk
            pos[b] = len(tk) - 1
        lt, ls = last_logits(PARAMS, STU, put(ids), jnp.asarray(pos), jnp.asarray(ANS))
        lt, ls = np.asarray(lt), np.asarray(ls)
        for b, it in enumerate(blk):
            okT += int(lt[b].argmax() == int(it["answer"]))
            okS += int(ls[b].argmax() == int(it["answer"]))
    return {"n": len(items), "teacher_acc": okT / len(items), "student_acc": okS / len(items)}


def evaluate(tag):
    t = time.time()
    r = {"wiki": ppl_eval(WIKI), "fineweb": ppl_eval(FWEVAL), "mmlu": mmlu_eval(), "seconds": None}
    r["seconds"] = time.time() - t
    RES[f"eval_{tag}"] = r
    log(f"eval {tag}:", json.dumps(r))
    save()
    return r


def save_student(name):
    flat = {}
    for path, x in jax.tree_util.tree_flatten_with_path(STU)[0]:
        flat["/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)] = np.asarray(x, np.float32)
    np.savez(f"{OUT}/{name}.npz", **flat)


# ------------------------------------------------------------------ run

t = time.time()
r0 = evaluate("init")
log(f"init eval (includes compile) {time.time() - t:.0f}s")
if not TINY and r0["wiki"]["teacher_ppl"] > 25:
    log("ABORT: JAX teacher wiki ppl", r0["wiki"]["teacher_ppl"], "is far from the torch reference 19.80")
    save()
    sys.stdout.flush(); os._exit(2)

OPT = {"t": jnp.zeros((), jnp.int32), "m": jax.tree_util.tree_map(jnp.zeros_like, STU),
       "v": jax.tree_util.tree_map(jnp.zeros_like, STU)}
budget = TRAIN_HOURS * 3600
t_train0 = last_ck = time.time()
step = 0
hist = []
while True:
    prog = (step / TINY_STEPS) if TINY else (time.time() - t_train0) / budget
    if prog >= 1.0:
        break
    lr = LR_REL * min(1.0, (step + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))
    ids = put(BATCHQ.get())
    ts = time.time()
    STU, OPT, aux, gn = train_step(STU, OPT, PARAMS, ids, jnp.float32(lr))
    step += 1
    if step <= 3 or step % 10 == 0:
        kl, ns, nt = (float(a) for a in aux)
        hist.append({"step": step, "kl": kl, "nll_s": ns, "nll_t": nt, "gnorm": float(gn), "lr": lr,
                     "step_s": time.time() - ts, "elapsed_s": time.time() - t_train0})
        log(f"step {step} kl {kl:.4f} nll_s {ns:.4f} nll_t {nt:.4f} gn {float(gn):.3f} lr {lr:.2e} "
            f"{time.time() - ts:.2f}s/step tok/s {BATCH * SEQ / (time.time() - ts):.0f}")
        RES["train_log"] = hist
    if time.time() - last_ck > CKPT_EVERY_S:
        save_student("student_E_kd")
        save()
        last_ck = time.time()
RES.update(train_steps=step, train_tokens=step * BATCH * SEQ, train_seconds=time.time() - t_train0)
log(f"training done: {step} steps, {step * BATCH * SEQ / 1e6:.1f}M tokens")
save_student("student_E_kd")
evaluate("final")
RES["total_seconds"] = time.time() - T0
save()
log("done")

import sys as _sys, os as _os
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)
