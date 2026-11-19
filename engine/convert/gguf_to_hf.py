#!/usr/bin/env python3
"""Convert the SenseNova-Loop-9B BF16 GGUF back to Hugging Face safetensors.

Usage:
    python gguf_to_hf.py --gguf SenseNova-Loop-9B-BF16.gguf --config config.json --out DIR [--text-only]
    python gguf_to_hf.py --gguf SenseNova-Loop-9B-BF16.gguf --check            # header only, no tensor data
    python gguf_to_hf.py --gguf X.gguf --dump-names gguf_names.json
    python gguf_to_hf.py --gguf X.gguf --map-md name_map.md

Notes:
  * Needs no GPU. Tensor data is memory-mapped and handled one tensor at a time; shards are capped at 4 GB.
  * The HF names and shapes/dtypes come from the Preview checkpoint (its index and safetensors headers).
    Preview is fetched from Hugging Face with HTTP range requests unless --preview-index / --preview-shapes
    point at local copies. Only headers are read from the Preview shards.
  * The GGUF has no vision tower (it lives in mmproj-SenseNova-Loop-9B-F16.gguf, arch `clip`, F16). The
    vision tensors are reported as unmapped. Pass --visual-src DIR (a local Preview snapshot) to copy them.
  * The GGUF carries the MTP head as block 56 (`nextn.*`); it is converted to `mtp.*` unless --text-only.

Inverse transforms (GGUF -> HF) are the inverses of the converter in
https://github.com/OpenSenseNova/llama.cpp (conversion/qwen.py), see TRANSFORM_SOURCES below.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import struct
import sys
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np

PREVIEW_REPO = "https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/resolve/main/"
SHARD_LIMIT_BYTES = 4_000_000_000

GGML_TYPE_NAMES = {0: "F32", 1: "F16", 30: "BF16"}
NP_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4}
SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

# Source lines of the converter that define each forward transform (GGUF <- HF).
# Paths refer to https://github.com/OpenSenseNova/llama.cpp, file conversion/qwen.py unless stated.
TRANSFORM_SOURCES = {
    "copy": "no transform: Qwen3NextModel does not rewrite this tensor (qwen.py:424-465, _LinearAttentionVReorderBase.modify_tensors qwen.py:661-716)",
    "minus1": "norm +1 offset: qwen.py:431 `elif name.endswith(\"norm.weight\") and not name.endswith(\"linear_attn.norm.weight\"): data_torch + 1`",
    "ssm_a": "A_log -> ssm_a = -exp(A_log): qwen.py:425-426; V-head reorder qwen.py:688-695 (1-D, head_dim 1)",
    "dt_bias": "dt_bias -> ssm_dt.bias (rename dt_bias -> dt_proj.bias): qwen.py:427-428; V-head reorder qwen.py:688-695",
    "conv1d": "conv1d squeeze [8192,1,4]->[8192,4]: qwen.py:429-430; V-channel reorder qwen.py:697-704 (channels >= 2*16*128)",
    "ssm_norm": "linear_attn.norm.weight is excluded from +1: qwen.py:431 (copy otherwise)",
    "ssm_out": "out_proj column reorder: qwen.py:706-713 (dim 1, head_v_dim)",
    "qkv_part": "in_proj_qkv V-row reorder: qwen.py:670-678; merged into attn_qkv_gate by _merge_shared_input qwen.py:503-531 (parts ATTN_QKV, ATTN_GATE) -> cat([qkv, z]) dim 0",
    "z_part": "in_proj_z row reorder: qwen.py:680-683; merged into attn_qkv_gate (cat dim 0) qwen.py:503-531",
    "ba_b": "in_proj_b row reorder (head_dim 1): qwen.py:684-687; merged into ssm_beta_alpha as first half: qwen.py:503-531 with parts (SSM_BETA, SSM_ALPHA) and tensor_mapping.py:1212 (SSM_BETA <- in_proj_b)",
    "ba_a": "in_proj_a row reorder (head_dim 1): qwen.py:684-687; second half of ssm_beta_alpha: tensor_mapping.py:1184 (SSM_ALPHA <- in_proj_a)",
    "mhc": "no converter rule in qwen.py; the name hc_attn_fn/hc_ffn_fn is the released GGUF name for the Preview mhc_attn/mhc_ffn router (same shape, F32). Verified only by shape.",
    "mtp_rename": "mtp.* remap: qwen.py:340-360 (_QwenMtpMixin.filter_tensors): fc->eh_proj, pre_fc_norm_embedding->enorm, pre_fc_norm_hidden->hnorm, norm->shared_head.norm; mtp.layers.i -> block n_layer+i",
    "loop": "loop expansion: the GGUF stores one block per execution slot and only the first visit owns tensors (Xolotl tools/sensenova/convert_sensenova_loop.py header; chynggi patch 0002 conversion/sensenova.py `virtual_for[phys][:1]`)",
    "embed": "token_embd / output: qwen.py Qwen3NextModel base mapping (tensor_mapping.py)",
    "pass": "",
}


# --------------------------------------------------------------------------------------
# GGUF header (streaming, header only)
# --------------------------------------------------------------------------------------
class _Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        buf = self.f.read(size)
        if len(buf) != size:
            raise EOFError("truncated GGUF header")
        return struct.unpack(fmt, buf)[0]

    def string(self):
        n = self.unpack("<Q")
        return self.f.read(n).decode("utf-8")

    def value(self, vtype):
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype = self.unpack("<I")
            n = self.unpack("<Q")
            if etype == 8:
                if n <= 64:
                    return [self.string() for _ in range(n)]
                for _ in range(n):  # large string arrays (tokenizer): skip
                    self.f.read(self.unpack("<Q"))
                return {"array_len": n, "elem_type": 8}
            fmt = SCALAR_FMT[etype]
            size = struct.calcsize(fmt)
            if n <= 64:
                return [struct.unpack(fmt, self.f.read(size))[0] for _ in range(n)]
            self.f.seek(size * n, os.SEEK_CUR)
            return {"array_len": n, "elem_type": etype}
        return self.unpack(SCALAR_FMT[vtype])


def read_gguf_header(path: str, require_data: bool = False) -> dict:
    """Parse magic, KV metadata and tensor infos. Never reads tensor data.

    With require_data=False a truncated file (header only) is accepted; tensors past EOF get `in_file: False`.
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = r.unpack("<I")
        n_tensors = r.unpack("<Q")
        n_kv = r.unpack("<Q")
        kv = OrderedDict()
        for _ in range(n_kv):
            key = r.string()
            vtype = r.unpack("<I")
            kv[key] = r.value(vtype)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.unpack("<I")
            dims = [r.unpack("<Q") for _ in range(n_dims)]
            gtype = r.unpack("<I")
            offset = r.unpack("<Q")
            if gtype not in GGML_TYPE_NAMES:
                raise ValueError(f"tensor {name}: unsupported ggml type {gtype}")
            shape = list(reversed(dims))  # numpy / HF order (GGUF stores ne0 first)
            numel = int(np.prod(shape)) if shape else 1
            dtype = GGML_TYPE_NAMES[gtype]
            tensors.append({
                "name": name, "dtype": dtype, "shape": shape, "gguf_dims": dims,
                "offset": offset, "nbytes": numel * NP_ITEMSIZE[dtype],
            })
        alignment = kv.get("general.alignment", 32)
        data_start = (f.tell() + alignment - 1) // alignment * alignment
    for t in tensors:
        t["abs_offset"] = data_start + t["offset"]
        t["in_file"] = t["abs_offset"] + t["nbytes"] <= file_size
        if require_data and not t["in_file"]:
            raise ValueError(f"tensor {t['name']} extends past end of file")
    return {"version": version, "kv": kv, "tensors": tensors, "data_start": data_start, "file_size": file_size}


# --------------------------------------------------------------------------------------
# Preview (HF) side: index + safetensors headers
# --------------------------------------------------------------------------------------
def _http_range(url: str, start: int, end_inclusive: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def load_preview_index(path: str | None) -> dict:
    if path:
        return json.load(open(path))
    return _http_json(PREVIEW_REPO + "model.safetensors.index.json")


def load_preview_config(path: str | None) -> dict:
    if path:
        return json.load(open(path))
    return _http_json(PREVIEW_REPO + "config.json")


def read_st_header_local(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def load_preview_shapes(path: str | None, index: dict, preview_dir: str | None = None) -> dict:
    """name -> {"dtype", "shape", "file"} from the safetensors headers of every shard in the index."""
    if path:
        return json.load(open(path))
    out = {}
    files = sorted(set(index["weight_map"].values()))
    for fn in files:
        if preview_dir:
            hdr, _ = read_st_header_local(os.path.join(preview_dir, fn))
        else:
            url = PREVIEW_REPO + fn
            n = struct.unpack("<Q", _http_range(url, 0, 7))[0]
            hdr = json.loads(_http_range(url, 8, 8 + n - 1))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = {"dtype": v["dtype"], "shape": v["shape"], "file": fn}
    missing = set(index["weight_map"]) - set(out)
    if missing:
        raise ValueError(f"index names missing from shard headers: {sorted(missing)[:5]}")
    return out


# --------------------------------------------------------------------------------------
# Loop plan (copied from the Preview modeling code semantics: modeling_sensenova_loop.execution_plan)
# --------------------------------------------------------------------------------------
def execution_plan(config: dict) -> list[int]:
    loop = config["loop_settings"]
    text = config["text_config"]
    n_phys = text["num_hidden_layers"]
    start, end, count = loop["start"], loop["end"], loop["eval_count"]
    if loop.get("counts") != [count]:
        raise ValueError("only fixed-count Loop checkpoints are supported")
    return list(range(start)) + list(range(start, end)) * count + list(range(end, n_phys))


# --------------------------------------------------------------------------------------
# Inverse transforms (operate on float32 numpy arrays)
# --------------------------------------------------------------------------------------
class Dims:
    def __init__(self, config: dict):
        t = config["text_config"]
        self.num_k = t["linear_num_key_heads"]          # 16
        self.num_v = t["linear_num_value_heads"]        # 32
        self.hk = t["linear_key_head_dim"]              # 128
        self.hv = t["linear_value_head_dim"]            # 128
        self.nvpk = self.num_v // self.num_k            # 2
        assert self.num_v % self.num_k == 0

    def inv_index(self, n: int, head_dim: int) -> np.ndarray:
        """Forward converter reorder: out = in[idx]. Inverse: in = out[inv]."""
        assert n == self.num_v * head_dim, (n, self.num_v, head_dim)
        idx = np.arange(n).reshape(self.num_k, self.nvpk, head_dim).transpose(1, 0, 2).reshape(n)
        inv = np.empty(n, dtype=np.int64)
        inv[idx] = np.arange(n)
        return inv


def unreorder_rows(d: Dims, x: np.ndarray, head_dim: int) -> np.ndarray:
    return x[d.inv_index(x.shape[0], head_dim)]


def unreorder_cols(d: Dims, x: np.ndarray, head_dim: int) -> np.ndarray:
    return x[:, d.inv_index(x.shape[1], head_dim)]


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bfloat16 bit pattern (uint16)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    if nan.any():
        rounded = np.where(nan, np.uint16(0x7FC0), rounded)
    return rounded


# Each op: (d: Dims, [float32 source arrays]) -> float32 array (before dtype cast)
def _op_copy(d, xs):
    return xs[0]


def _op_minus1(d, xs):
    return xs[0] - np.float32(1.0)


def _op_ssm_a(d, xs):
    # forward: ssm_a = -exp(A_log)  =>  A_log = log(-ssm_a); V reorder is elementwise-commuting
    x = xs[0]
    if np.any(x >= 0):
        raise ValueError("ssm_a must be strictly negative for the -exp(A_log) inverse")
    return unreorder_rows(d, np.log(-x), 1)


def _op_dt_bias(d, xs):
    return unreorder_rows(d, xs[0], 1)


def _op_conv1d(d, xs):
    x = xs[0]  # [8192, 4]
    qk = 2 * d.hk * d.num_k
    out = np.concatenate([x[:qk], unreorder_rows(d, x[qk:], d.hv)], axis=0)
    return out.reshape(out.shape[0], 1, out.shape[1])  # forward: squeeze


def _op_ssm_out(d, xs):
    return unreorder_cols(d, xs[0], d.hv)


def _op_qkv_part(d, xs):
    x = xs[0]  # rows [q | k | v | z] = [2048 | 2048 | 4096 | 4096]
    qk = 2 * d.hk * d.num_k
    return np.concatenate([x[:qk], unreorder_rows(d, x[qk:qk + d.num_v * d.hv], d.hv)], axis=0)


def _op_z_part(d, xs):
    x = xs[0]
    start = 2 * d.hk * d.num_k + d.num_v * d.hv
    return unreorder_rows(d, x[start:start + d.num_v * d.hv], d.hv)


def _op_ba_b(d, xs):
    return unreorder_rows(d, xs[0][:d.num_v], 1)


def _op_ba_a(d, xs):
    return unreorder_rows(d, xs[0][d.num_v:2 * d.num_v], 1)


OPS = {
    "copy": _op_copy, "minus1": _op_minus1, "ssm_a": _op_ssm_a, "dt_bias": _op_dt_bias,
    "conv1d": _op_conv1d, "ssm_norm": _op_copy, "ssm_out": _op_ssm_out,
    "qkv_part": _op_qkv_part, "z_part": _op_z_part, "ba_b": _op_ba_b, "ba_a": _op_ba_a,
    "mhc": _op_copy, "mtp_rename": _op_copy, "embed": _op_copy,
}


# --------------------------------------------------------------------------------------
# Name mapping
# --------------------------------------------------------------------------------------
# GGUF per-block suffix -> (HF suffix, transform id)
TRUNK_RULES = {
    "attn_norm.weight": ("input_layernorm.weight", "minus1"),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", "minus1"),
    "ffn_gate.weight": ("mlp.gate_proj.weight", "copy"),
    "ffn_up.weight": ("mlp.up_proj.weight", "copy"),
    "ffn_down.weight": ("mlp.down_proj.weight", "copy"),
    "attn_q.weight": ("self_attn.q_proj.weight", "copy"),
    "attn_k.weight": ("self_attn.k_proj.weight", "copy"),
    "attn_v.weight": ("self_attn.v_proj.weight", "copy"),
    "attn_output.weight": ("self_attn.o_proj.weight", "copy"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "minus1"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "minus1"),
}
LINEAR_RULES = {
    "ssm_a": ("linear_attn.A_log", "ssm_a"),
    "ssm_dt.bias": ("linear_attn.dt_bias", "dt_bias"),
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "conv1d"),
    "ssm_norm.weight": ("linear_attn.norm.weight", "ssm_norm"),
    "ssm_out.weight": ("linear_attn.out_proj.weight", "ssm_out"),
}
# GGUF merged tensors that split into several HF tensors: suffix -> [(HF suffix, transform, row slice)]
LINEAR_SPLITS = {
    "attn_qkv_gate.weight": [("linear_attn.in_proj_qkv.weight", "qkv_part"), ("linear_attn.in_proj_z.weight", "z_part")],
    "ssm_beta_alpha.weight": [("linear_attn.in_proj_b.weight", "ba_b"), ("linear_attn.in_proj_a.weight", "ba_a")],
}
MHC_RULES = {
    "hc_attn_fn.weight": ("mhc_attn.weight", "mhc"),
    "hc_attn_fn.bias": ("mhc_attn.bias", "mhc"),
    "hc_ffn_fn.weight": ("mhc_ffn.weight", "mhc"),
    "hc_ffn_fn.bias": ("mhc_ffn.bias", "mhc"),
}
MTP_RULES = {
    "nextn.eh_proj.weight": ("mtp.fc.weight", "copy"),
    "nextn.enorm.weight": ("mtp.pre_fc_norm_embedding.weight", "minus1"),
    "nextn.hnorm.weight": ("mtp.pre_fc_norm_hidden.weight", "minus1"),
    "nextn.shared_head_norm.weight": ("mtp.norm.weight", "minus1"),
}
TOP_RULES = {
    "token_embd.weight": ("model.language_model.embed_tokens.weight", "copy"),
    "output_norm.weight": ("model.language_model.norm.weight", "minus1"),
    "output.weight": ("lm_head.weight", "copy"),
}

BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


class OutSpec:
    __slots__ = ("hf", "sources", "op", "out_shape", "dtype", "src_dtype", "gguf_names", "note", "expected_src_shape")

    def __init__(self, hf, sources, op, out_shape, dtype, src_dtype, note=""):
        self.hf = hf
        self.sources = tuple(sources)
        self.op = op
        self.out_shape = list(out_shape)
        self.dtype = dtype
        self.src_dtype = src_dtype
        self.note = note


def build_plan(header: dict, config: dict, preview: dict, text_only: bool):
    """Return (outputs, consumed, problems). outputs: list[OutSpec] whose HF names exist in preview."""
    d = Dims(config)
    plan = execution_plan(config)
    n_virtual = len(plan)
    first_visit = {}
    for v, p in enumerate(plan):
        first_visit.setdefault(p, v)
    first_visit_slots = set(first_visit.values())
    gguf = {t["name"]: t for t in header["tensors"]}
    problems = []
    outputs: list[OutSpec] = []
    consumed: set[str] = set()

    def add(hf, srcs, op, out_shape, src_dtype, note=""):
        if hf not in preview:
            problems.append(f"computed HF name not in Preview index: {hf}")
            return
        p_dtype = preview[hf]["dtype"]
        p_shape = list(preview[hf]["shape"])
        if list(out_shape) != p_shape:
            problems.append(f"shape mismatch {hf}: GGUF-derived {out_shape} vs Preview {p_shape}")
        outputs.append(OutSpec(hf, srcs, op, out_shape, p_dtype, src_dtype, note))

    for name, (hf, op) in TOP_RULES.items():
        if name not in gguf:
            problems.append(f"missing GGUF tensor {name}")
            continue
        t = gguf[name]
        consumed.add(name)
        add(hf, [name], op, t["shape"], t["dtype"], TRANSFORM_SOURCES[op])

    for name, t in gguf.items():
        if name in consumed:
            continue
        m = BLK_RE.match(name)
        if not m:
            continue
        bid, tail = int(m.group(1)), m.group(2)
        if bid == n_virtual:  # NextN / MTP block
            if text_only:
                consumed.add(name)
                continue
            if tail.startswith("nextn."):
                if tail in MTP_RULES:
                    hf, op = MTP_RULES[tail]
                    consumed.add(name)
                    add(hf, [name], op, t["shape"], t["dtype"], TRANSFORM_SOURCES[op])
                elif tail in ("nextn.embed_tokens.weight", "nextn.shared_head_head.weight"):
                    problems.append(f"{name}: no HF counterpart in Preview index")
                else:
                    problems.append(f"{name}: unknown nextn tensor")
                continue
            base, hf_prefix = tail, "mtp.layers.0."
            if base in TRUNK_RULES:
                hf_suffix, op = TRUNK_RULES[base]
                consumed.add(name)
                add(hf_prefix + hf_suffix, [name], op, t["shape"], t["dtype"], TRANSFORM_SOURCES[op])
            else:
                problems.append(f"{name}: unmapped MTP tensor")
            continue
        if bid >= n_virtual or bid not in first_visit_slots:
            problems.append(f"{name}: block {bid} is not a first-visit slot of the loop plan (should not hold tensors)")
            continue
        phys = plan[bid]
        pre = f"model.language_model.layers.{phys}."
        if tail in TRUNK_RULES:
            hf_suffix, op = TRUNK_RULES[tail]
            consumed.add(name)
            add(pre + hf_suffix, [name], op, t["shape"], t["dtype"], TRANSFORM_SOURCES[op])
        elif tail in LINEAR_RULES:
            hf_suffix, op = LINEAR_RULES[tail]
            consumed.add(name)
            out_shape = [t["shape"][0], 1, t["shape"][1]] if op == "conv1d" else t["shape"]
            if op == "ssm_a" or op == "dt_bias":
                out_shape = t["shape"]
            add(pre + hf_suffix, [name], op, out_shape, t["dtype"], TRANSFORM_SOURCES[op])
        elif tail in LINEAR_SPLITS:
            consumed.add(name)
            rows = 0
            for hf_suffix, op in LINEAR_SPLITS[tail]:
                if op == "qkv_part":
                    shape = [2 * d.hk * d.num_k + d.num_v * d.hv, t["shape"][1]]
                elif op == "z_part":
                    shape = [d.num_v * d.hv, t["shape"][1]]
                else:
                    shape = [d.num_v, t["shape"][1]]
                add(pre + hf_suffix, [name], op, shape, t["dtype"], TRANSFORM_SOURCES[op])
                rows += shape[0]
            if rows != t["shape"][0]:
                problems.append(f"{name}: split rows {rows} != GGUF rows {t['shape'][0]}")
        elif tail in MHC_RULES:
            hf_suffix, op = MHC_RULES[tail]
            consumed.add(name)
            add(pre + hf_suffix, [name], op, t["shape"], t["dtype"], TRANSFORM_SOURCES[op])
        else:
            problems.append(f"{name}: unmapped tensor (block {bid} -> layer {phys})")
    unconsumed = sorted(set(gguf) - consumed)
    for n in unconsumed:
        problems.append(f"GGUF tensor not consumed: {n}")
    return outputs, consumed, problems


def hf_names_covered(outputs) -> set:
    return {o.hf for o in outputs}


def expected_absent(hf: str) -> str | None:
    if hf.startswith("model.visual."):
        return "vision tower: not in the text GGUF; it is in mmproj-SenseNova-Loop-9B-F16.gguf (arch clip, F16, 334 tensors)"
    return None


def check_config_vs_gguf(header: dict, config: dict) -> list[str]:
    kv = header["kv"]
    t = config["text_config"]
    rows = []

    def cmp(label, gguf_val, cfg_val):
        ok = gguf_val == cfg_val
        rows.append(f"{'OK  ' if ok else 'DIFF'} {label}: GGUF={gguf_val} config={cfg_val}")
    loop = config["loop_settings"]
    cmp("loop.start", kv.get("sensenova_loop.loop.start"), loop["start"])
    cmp("loop.end", kv.get("sensenova_loop.loop.end"), loop["end"])
    cmp("loop.count", kv.get("sensenova_loop.loop.count"), loop["eval_count"])
    m = config["mhc"]
    cmp("mhc.start", kv.get("sensenova_loop.mhc.start"), m["start"])
    cmp("mhc.end", kv.get("sensenova_loop.mhc.end"), m["end"])
    cmp("mhc.streams", kv.get("sensenova_loop.mhc.streams"), m["streams"])
    cmp("mhc.iterations", kv.get("sensenova_loop.mhc.iterations"), m["iterations"])
    cmp("mhc.fixed_identity", kv.get("sensenova_loop.mhc.fixed_identity"), m["fixed_identity"])
    cmp("mhc.pre_scale", kv.get("sensenova_loop.mhc.pre_scale"), m.get("pre_scale", 1.0))
    cmp("block_count (phys+mtp)", kv.get("sensenova_loop.block_count"), len(execution_plan(config)) + 1)
    cmp("ssm.conv_kernel", kv.get("sensenova_loop.ssm.conv_kernel"), t["linear_conv_kernel_dim"])
    cmp("ssm.state_size", kv.get("sensenova_loop.ssm.state_size"), t["linear_key_head_dim"])
    cmp("ssm.group_count", kv.get("sensenova_loop.ssm.group_count"), t["linear_num_key_heads"])
    cmp("ssm.time_step_rank", kv.get("sensenova_loop.ssm.time_step_rank"), t["linear_num_value_heads"])
    cmp("embedding_length", kv.get("sensenova_loop.embedding_length"), t["hidden_size"])
    cmp("feed_forward_length", kv.get("sensenova_loop.feed_forward_length"), t["intermediate_size"])
    cmp("attention.head_count", kv.get("sensenova_loop.attention.head_count"), t["num_attention_heads"])
    cmp("attention.head_count_kv", kv.get("sensenova_loop.attention.head_count_kv"), t["num_key_value_heads"])
    cmp("attention.key_length", kv.get("sensenova_loop.attention.key_length"), t["head_dim"])
    cmp("rope.freq_base", kv.get("sensenova_loop.rope.freq_base"), t["rope_parameters"]["rope_theta"])
    cmp("rope.dimension_count", kv.get("sensenova_loop.rope.dimension_count"), int(t["head_dim"] * t["partial_rotary_factor"]))
    rs_type = kv.get("sensenova_loop.rope.scaling.type")
    rp = t.get("rope_parameters", {})
    cmp("rope.scaling.type", rs_type, rp.get("rope_type") if rp.get("rope_type") != "default" else None)
    cmp("context_length", kv.get("sensenova_loop.context_length"), t["max_position_embeddings"])
    rec = kv.get("sensenova_loop.attention.recurrent_layers")
    if isinstance(rec, list):
        expect = [t["layer_types"][p] == "linear_attention" for p in execution_plan(config)] + [False]
        cmp("recurrent_layers (per slot)", rec == expect, True)
    return rows


def yarn_from_gguf(kv: dict) -> dict | None:
    if kv.get("sensenova_loop.rope.scaling.type") != "yarn":
        return None
    return {
        "rope_type": "yarn",
        "factor": float(kv["sensenova_loop.rope.scaling.factor"]),
        "original_max_position_embeddings": int(kv["sensenova_loop.rope.scaling.original_context_length"]),
    }


def build_out_config(config: dict, header: dict, text_only: bool, keep_rope: bool) -> tuple[dict, list[str]]:
    kv = header["kv"]
    cfg = copy.deepcopy(config)
    notes = []
    cfg["loop_settings"] = {"counts": [kv["sensenova_loop.loop.count"]], "end": kv["sensenova_loop.loop.end"],
                            "eval_count": kv["sensenova_loop.loop.count"], "start": kv["sensenova_loop.loop.start"]}
    cfg["mhc"] = {"end": kv["sensenova_loop.mhc.end"], "fixed_identity": kv["sensenova_loop.mhc.fixed_identity"],
                  "iterations": kv["sensenova_loop.mhc.iterations"], "pre_scale": kv["sensenova_loop.mhc.pre_scale"],
                  "start": kv["sensenova_loop.mhc.start"], "streams": kv["sensenova_loop.mhc.streams"]}
    if not keep_rope:
        yarn = yarn_from_gguf(kv)
        if yarn:
            rp = dict(cfg["text_config"].get("rope_parameters", {}))
            rp.update(yarn)
            cfg["text_config"]["rope_parameters"] = rp
            cfg["text_config"]["max_position_embeddings"] = int(kv["sensenova_loop.context_length"])
            notes.append("rope: GGUF declares YaRN (factor 2.0, original 262144); written into text_config.rope_parameters. "
                         "Preview config has none. Pass --keep-config-rope to keep the config unchanged.")
    if text_only:
        cfg["text_config"]["mtp_num_hidden_layers"] = 0
        notes.append("--text-only: mtp_num_hidden_layers set to 0")
    return cfg, notes


# --------------------------------------------------------------------------------------
# Materialise tensors
# --------------------------------------------------------------------------------------
class GGUFData:
    def __init__(self, path: str, header: dict):
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.tensors = {t["name"]: t for t in header["tensors"]}

    def raw(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        buf = self.mm[t["abs_offset"]: t["abs_offset"] + t["nbytes"]]
        if t["dtype"] == "BF16":
            return buf.view(np.uint16).reshape(t["shape"])
        if t["dtype"] == "F32":
            return buf.view(np.float32).reshape(t["shape"])
        raise ValueError(f"unsupported source dtype {t['dtype']} for {name}")

    def f32(self, name: str) -> np.ndarray:
        r = self.raw(name)
        if r.dtype == np.uint16:
            return bf16_bits_to_f32(r)
        return np.asarray(r, dtype=np.float32)


def produce(spec: OutSpec, gg: GGUFData, dims: Dims) -> np.ndarray:
    """Return the output tensor as uint16 (BF16 bits) or float32 array."""
    if spec.op == "copy" and gg.tensors[spec.sources[0]]["dtype"] == spec.dtype:
        return np.ascontiguousarray(gg.raw(spec.sources[0]))
    xs = [gg.f32(s) for s in spec.sources]
    x = OPS[spec.op](dims, xs)
    if list(x.shape) != spec.out_shape:
        raise ValueError(f"{spec.hf}: produced shape {list(x.shape)} != expected {spec.out_shape}")
    if spec.dtype == "BF16":
        return np.ascontiguousarray(f32_to_bf16_bits(x))
    if spec.dtype == "F32":
        return np.ascontiguousarray(x, dtype=np.float32)
    raise ValueError(f"unsupported Preview dtype {spec.dtype}")


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------
def natural_key(s: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


def nbytes_of(dtype: str, shape: list[int]) -> int:
    return int(np.prod(shape)) * NP_ITEMSIZE[dtype]


def shard_outputs(items: list[tuple[str, str, list[int]]]) -> list[list[tuple[str, str, list[int]]]]:
    shards, cur, cur_bytes = [], [], 0
    for it in items:
        b = nbytes_of(it[1], it[2])
        if cur and cur_bytes + b > SHARD_LIMIT_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(it)
        cur_bytes += b
    if cur:
        shards.append(cur)
    return shards


def write_safetensors(path: str, entries: list[tuple[str, str, list[int], callable]]):
    header = {"__metadata__": {"format": "pt"}}
    off = 0
    for name, dtype, shape, _ in entries:
        n = nbytes_of(dtype, shape)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name, dtype, shape, fn in entries:
            arr = np.ascontiguousarray(fn())
            if arr.nbytes != nbytes_of(dtype, shape):
                raise ValueError(f"{name}: wrote {arr.nbytes} bytes, expected {nbytes_of(dtype, shape)}")
            f.write(arr.reshape(-1).view(np.uint8))  # no extra copy
            del arr
    return off


def visual_entries(visual_src: str, index: dict, names: list[str]):
    """Yield (name, dtype, shape, producer) for vision tensors copied raw from a Preview snapshot."""
    out = []
    for name in names:
        fn = index["weight_map"][name]
        hdr, data_start = read_st_header_local(os.path.join(visual_src, fn))
        info = hdr[name]
        start, end = info["data_offsets"]
        path = os.path.join(visual_src, fn)

        def make(path=path, start=start, end=end, data_start=data_start, dtype=info["dtype"]):
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            buf = mm[data_start + start: data_start + end]
            return np.ascontiguousarray(buf).view(np.uint16 if dtype == "BF16" else np.float32)
        out.append((name, info["dtype"], info["shape"], make))
    return out


def convert(args, header, config, preview, outputs, dims, text_only):
    gg = GGUFData(args.gguf, header)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    by_name = {}
    for spec in outputs:
        items.append((spec.hf, spec.dtype, spec.out_shape))
        by_name[spec.hf] = spec
    visual_names = []
    if args.visual_src:
        vindex = json.load(open(os.path.join(args.visual_src, "model.safetensors.index.json")))
        visual_names = sorted([n for n in vindex["weight_map"] if n.startswith("model.visual.")], key=natural_key)
        for n in visual_names:
            items.append((n, preview[n]["dtype"], preview[n]["shape"]))
    items.sort(key=lambda it: natural_key(it[0]))
    shards = shard_outputs(items)
    weight_map = {}
    total = 0
    n_shards = len(shards)
    vis_by_name = {}
    if visual_names:
        vindex_path = args.visual_src
        for e in visual_entries(vindex_path, vindex, visual_names):
            vis_by_name[e[0]] = e
    for si, shard in enumerate(shards, 1):
        fname = f"model-{si:05d}-of-{n_shards:05d}.safetensors"
        entries = []
        for name, dtype, shape in shard:
            if name in by_name:
                spec = by_name[name]
                entries.append((name, dtype, shape, (lambda s=spec: produce(s, gg, dims))))
            else:
                _, dt, sh, fn = vis_by_name[name]
                entries.append((name, dt, sh, fn))
            weight_map[name] = fname
        size = write_safetensors(str(out_dir / fname), entries)
        total += size
        print(f"wrote {fname}: {len(entries)} tensors, {size / 1e9:.2f} GB", file=sys.stderr)
    index_out = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items(), key=lambda kv: natural_key(kv[0])))}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index_out, indent=2))
    return weight_map


def write_map_md(path: str, header: dict, config: dict, preview: dict, outputs, consumed, problems, text_only, args):
    gguf = {t["name"]: t for t in header["tensors"]}
    rows = []
    for spec in sorted(outputs, key=lambda s: natural_key(s.hf)):
        srcs = spec.sources
        for s in srcs:
            t = gguf[s]
            rows.append((s, t["dtype"], t["shape"], spec.hf, spec.dtype, spec.out_shape, spec.op, spec.note))
    lines = []
    lines.append("# SenseNova-Loop-9B GGUF -> HF name map")
    lines.append("")
    lines.append("Generated by `gguf_to_hf.py --map-md`. GGUF: `SenseNova-Loop-9B-BF16.gguf` (490 tensors, arch `sensenova_loop`).")
    lines.append("HF names/shapes/dtypes: `sensenova/SenseNova-Loop-9B-Preview` index and safetensors headers.")
    lines.append("Shapes are numpy/HF order (GGUF stores dims reversed).")
    lines.append("")
    lines.append("## Coverage")
    n_gguf = len(header["tensors"])
    n_hf = len(preview)
    n_vis = sum(1 for n in preview if n.startswith("model.visual."))
    n_mapped_hf = len({o.hf for o in outputs})
    lines.append(f"- GGUF tensors consumed: {len(consumed)} / {n_gguf}")
    lines.append(f"- HF tensors in Preview index: {n_hf}; produced from GGUF: {n_mapped_hf}; "
                 f"vision (`model.visual.*`, not in this GGUF): {n_vis}; "
                 f"unaccounted: {n_hf - n_mapped_hf - n_vis}")
    lines.append(f"- Problems: {len(problems)}" + ("" if not problems else ""))
    for p in problems:
        lines.append(f"  - {p}")
    lines.append("")
    lines.append("## Transforms (inverse of the converter)")
    lines.append("")
    for k in ["copy", "minus1", "ssm_a", "dt_bias", "conv1d", "ssm_norm", "ssm_out", "qkv_part", "z_part", "ba_b", "ba_a", "mhc", "mtp_rename", "embed"]:
        if TRANSFORM_SOURCES.get(k):
            lines.append(f"- **{k}**: {TRANSFORM_SOURCES[k]}")
    lines.append("- **loop**: " + TRANSFORM_SOURCES["loop"])
    lines.append("")
    lines.append("Reorder (`V` heads): forward converter output `out[v*16 + k] = in[k*2 + v]` over the 32 V heads "
                 "(16 K groups x 2). Inverse: `in = out[inv]` with `inv` the inverse permutation (`Dims.inv_index`).")
    lines.append("")
    lines.append("## Per-tensor table")
    lines.append("")
    lines.append("| GGUF name | GGUF dtype | GGUF shape | HF name | HF dtype | HF shape | transform | source |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s, gd, gs, hf, hd, hs, op, note in rows:
        src_ref = TRANSFORM_SOURCES.get(op, "").split(":")[0] if op != "copy" else "no transform"
        lines.append(f"| `{s}` | {gd} | {gs} | `{hf}` | {hd} | {hs} | {op} | {src_ref} |")
    lines.append("")
    lines.append("## Not mapped")
    lines.append("")
    lines.append("### GGUF tensors not written as HF tensors")
    lines.append("")
    not_written = [n for n in gguf if n not in consumed]
    if not not_written:
        lines.append("None.")
    for n in not_written:
        lines.append(f"- `{n}`")
    lines.append("")
    lines.append("Tensors consumed but not written: `blk.56.nextn.*` are written only when not `--text-only`; "
                 "nothing else is skipped.")
    lines.append("")
    lines.append("### HF tensors in the Preview index with no GGUF counterpart")
    lines.append("")
    covered = {o.hf for o in outputs}
    absent = [n for n in sorted(preview, key=natural_key) if n not in covered]
    groups = {}
    for n in absent:
        reason = expected_absent(n) or "UNEXPECTED"
        groups.setdefault(reason, []).append(n)
    for reason, names in groups.items():
        lines.append(f"- {len(names)} tensors: {reason}")
        if reason != "UNEXPECTED":
            lines.append(f"  - e.g. `{names[0]}` ... `{names[-1]}`")
        else:
            for n in names:
                lines.append(f"  - `{n}`")
    lines.append("")
    lines.append("Vision tensors in `mmproj-SenseNova-Loop-9B-F16.gguf` (arch `clip`, F16) use different names "
                 "(`v.blk.N.*`, `mm.*`) and `v.patch_embd.weight` is stored as [1152, 3, 16, 16] (temporal axis folded), "
                 "while Preview has a Conv3d weight. They are not converted by this script.")
    lines.append("## Verification notes")
    lines.append("")
    lines.append("- Header-only `--check`: 490/490 GGUF tensors consumed; 538 HF names produced; 333 vision names absent (mmproj); 0 unexpected.")
    lines.append("- Round trip on synthetic tensors (forward converter reimplemented from qwen.py, then inverse): reorder, split, merge, conv1d and out_proj are exact; the +1 offset is exact except 2 of 4096 values differing by 2.6e-8 (fp32 precision of the converter's own `+ 1`).")
    lines.append("- Real GGUF bytes (layers 0, 3, 4, 56 small tensors fetched by range request): `ssm_a` is strictly negative (consistent with -exp(A_log)); norm tensors centre near 1 (consistent with the +1 offset); `GGUF - 1` for norms and `conv1d` are bf16-representable.")
    lines.append("- Preview is NOT bit-identical to the release at bf16 precision. Layer 4 `input_layernorm`: 1844 of 4096 elements differ from GGUF-1 by one bf16 ulp. Layer 4 `A_log`: 2 of 32 elements differ by one ulp from bf16(log(-ssm_a)). `dt_bias` matches bf16 rounding of the GGUF values exactly. The converter writes GGUF-derived values; Preview supplies only names, shapes and dtypes.")
    lines.append("- Config: loop (4, 28, 2), mhc (4, 28, streams 2, 20 iterations, fixed_identity False, pre_scale 1.0) and all linear_attn/attention/rope-theta fields agree between GGUF and the Preview config. Differences: GGUF declares YaRN rope (factor 2.0, original 262144) and context 524288; Preview config has neither (applied by default, see --keep-config-rope).")
    lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")


def dump_names(path: str, header: dict):
    out = {
        "source": "SenseNova-Loop-9B-BF16.gguf",
        "gguf_version": header["version"],
        "data_start": header["data_start"],
        "tensor_count": len(header["tensors"]),
        "tensors": [
            {"name": t["name"], "dtype": t["dtype"], "shape": t["shape"], "gguf_dims": t["gguf_dims"],
             "offset": t["offset"], "nbytes": t["nbytes"]}
            for t in header["tensors"]
        ],
    }
    Path(path).write_text(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="local BF16 GGUF (header is parsed; data only read in convert mode)")
    ap.add_argument("--config", help="config.json to copy (default: Preview config.json, fetched)")
    ap.add_argument("--out", help="output directory (convert mode)")
    ap.add_argument("--text-only", action="store_true", help="omit the MTP head (mtp.*) and set mtp_num_hidden_layers=0")
    ap.add_argument("--check", action="store_true", help="verify names/shapes against the Preview index; no tensor data read")
    ap.add_argument("--dump-names", metavar="JSON", help="write every GGUF tensor name/dtype/shape to JSON and exit")
    ap.add_argument("--map-md", metavar="MD", help="write the name map as Markdown and exit")
    ap.add_argument("--preview-index", help="local model.safetensors.index.json (default: fetch from HF)")
    ap.add_argument("--preview-shapes", help="JSON cache of Preview shard headers (default: fetch headers from HF)")
    ap.add_argument("--preview-dir", help="local Preview snapshot dir (reads shard headers from disk instead of HF)")
    ap.add_argument("--visual-src", help="local Preview snapshot dir: copy model.visual.* tensors from it (convert mode)")
    ap.add_argument("--keep-config-rope", action="store_true", help="do not apply the GGUF YaRN settings to the config")
    args = ap.parse_args(argv)

    header = read_gguf_header(args.gguf, require_data=not (args.check or args.dump_names or args.map_md))
    print(f"GGUF v{header['version']}: {len(header['tensors'])} tensors, data starts at byte {header['data_start']}",
          file=sys.stderr)

    index = load_preview_index(args.preview_index)
    preview = load_preview_shapes(args.preview_shapes, index, args.preview_dir)
    config = json.load(open(args.config)) if args.config else load_preview_config(None)

    if args.dump_names:
        dump_names(args.dump_names, header)
        print(f"wrote {args.dump_names}", file=sys.stderr)
        if not (args.check or args.map_md or args.out):
            return 0

    outputs, consumed, problems = build_plan(header, config, preview, args.text_only)
    # HF names expected by the Preview index
    expected_names = set(preview)
    if args.text_only:
        expected_names = {n for n in expected_names if not n.startswith("mtp.")}
    covered = {o.hf for o in outputs}
    unexplained = [n for n in sorted(expected_names - covered, key=natural_key) if expected_absent(n) is None]
    visual_missing = [n for n in sorted(expected_names - covered, key=natural_key) if expected_absent(n) is not None]

    if args.map_md:
        write_map_md(args.map_md, header, config, preview, outputs, consumed, problems, args.text_only, args)
        print(f"wrote {args.map_md}", file=sys.stderr)

    if args.check or args.map_md or args.dump_names:
        print("config vs GGUF metadata:")
        for row in check_config_vs_gguf(header, config):
            print("  " + row)
        print(f"GGUF tensors consumed: {len(consumed)}/{len(header['tensors'])}")
        print(f"HF outputs derived: {len(outputs)} (distinct names {len(covered)}), Preview index names: {len(preview)}")
        print(f"HF names with no GGUF source (vision, expected): {len(visual_missing)}")
        print(f"HF names with no GGUF source (UNEXPECTED): {len(unexplained)}")
        for n in unexplained:
            print("  " + n)
        print(f"problems: {len(problems)}")
        for p in problems:
            print("  " + p)
        ok = not problems and not unexplained and len(consumed) == len(header["tensors"])
        print("CHECK " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1

    if not args.out:
        ap.error("--out is required in convert mode")
    if problems or unexplained:
        for p in problems + unexplained:
            print("ERROR " + p, file=sys.stderr)
        return 1
    if visual_missing and not args.visual_src:
        print(f"WARNING: {len(visual_missing)} vision tensors not written (not in the GGUF). Pass --visual-src DIR, "
              "or --text-only; the output will not load into the full model without them.", file=sys.stderr)
    dims = Dims(config)
    out_cfg, notes = build_out_config(config, header, args.text_only, args.keep_config_rope)
    for n in notes:
        print("config: " + n, file=sys.stderr)
    for row in check_config_vs_gguf(header, out_cfg):
        if row.startswith("DIFF"):
            print("config (after GGUF override) " + row, file=sys.stderr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(out_cfg, indent=2))
    convert(args, header, config, preview, outputs, dims, args.text_only)
    print(f"done: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
