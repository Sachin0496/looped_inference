"""Render a Kaggle script kernel with sources inlined. Usage: python3 kaggle/render.py mhc_bench"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
JOBS = {
    "mhc_bench": {
        "template": "kaggle/mhc_bench/bench_template.py",
        "subs": {"__CU_SOURCE__": "engine/kernels/mhc_fused.cu", "__REF_SOURCE__": "engine/ref/mhc_ref.py"},
        "id": "sachin0496/sensenovakv-mhc-bench",
        "gpu": True,
    },
    "gemv_bench": {
        "template": "kaggle/gemv_bench/bench_template.py",
        "subs": {"__CU_SOURCE__": "engine/kernels/w4_gemv.cu", "__REF_SOURCE__": "engine/ref/w4_ref.py"},
        "id": "sachin0496/sensenovakv-gemv-bench",
        "gpu": True,
    },
    "deltanet_bench": {
        "template": "kaggle/deltanet_bench/bench_template.py",
        "subs": {"__CU_SOURCE__": "engine/kernels/deltanet_fused.cu", "__REF_SOURCE__": "engine/ref/deltanet_ref.py"},
        "id": "sachin0496/sensenovakv-deltanet-bench",
        "gpu": True,
        "internet": False,
    },
    "kv_analysis": {
        "template": "kaggle/kv_analysis/kv_analysis.py",
        "subs": {},
        "id": "sachin0496/sensenovakv-kv-analysis",
        "gpu": True,
        "internet": True,
    },
    "attn_bench": {
        "template": "kaggle/attn_bench/bench_template.py",
        "subs": {"__CU_SOURCE__": "engine/kernels/kv4_attn.cu", "__REF_SOURCE__": "engine/ref/kv4_attn_ref.py"},
        "id": "sachin0496/sensenovakv-attn-bench",
        "gpu": True,
        "internet": False,
    },
    "distill_B": {
        "template": "kaggle/distill/distill.py",
        "subs": {"__CONVERTER_SOURCE__": "engine/convert/gguf_to_hf.py"},
        "prelude": 'import os\nos.environ["SNKV_CONFIG"] = "B"\nos.environ["SNKV_RELEASE"] = "1"\n',
        "id": "sachin0496/sensenovakv-distill-b",
        "gpu": True,
        "internet": True,
    },
    "distill_A": {
        "template": "kaggle/distill/distill.py",
        "subs": {"__CONVERTER_SOURCE__": "engine/convert/gguf_to_hf.py"},
        "prelude": 'import os\nos.environ["SNKV_CONFIG"] = "A"\nos.environ["SNKV_RELEASE"] = "1"\n',
        "id": "sachin0496/sensenovakv-distill-a",
        "gpu": True,
        "internet": True,
    },
    "distill_E": {
        "template": "kaggle/distill/distill.py",
        "subs": {"__CONVERTER_SOURCE__": "engine/convert/gguf_to_hf.py"},
        "prelude": 'import os\nos.environ["SNKV_CONFIG"] = "E"\nos.environ["SNKV_RELEASE"] = "1"\n',
        "id": "sachin0496/sensenovakv-distill-e",
        "gpu": True,
        "internet": True,
    },
    "engine_e2e": {
        "template": "kaggle/engine_e2e/engine_e2e.py",
        "subs": {
            "__ENGINE_SOURCE__": "engine/runtime/snkv_engine.py",
            "__W4_CU_SOURCE__": "engine/kernels/w4_gemv.cu",
            "__MHC_CU_SOURCE__": "engine/kernels/mhc_fused.cu",
            "__DN_CU_SOURCE__": "engine/kernels/deltanet_fused.cu",
            "__CONVERTER_SOURCE__": "engine/convert/gguf_to_hf.py",
        },
        "id": "sachin0496/sensenovakv-engine-e2e",
        "gpu": True,
        "internet": True,
    },
    "kv_release": {
        "template": "kaggle/kv_release/kv_release.py",
        "subs": {"__CONVERTER_SOURCE__": "engine/convert/gguf_to_hf.py", "__KV_ANALYSIS_SOURCE__": "kaggle/kv_analysis/kv_analysis.py"},
        "id": "sachin0496/sensenovakv-kv-release",
        "gpu": True,
        "internet": True,
    },
    "loop_conv": {
        "template": "kaggle/loop_conv/loop_conv.py",
        "subs": {},
        "id": "sachin0496/sensenovakv-loop-conv",
        "gpu": True,
        "internet": True,
    },
}

name = sys.argv[1]
job = JOBS[name]
text = (ROOT / job["template"]).read_text()
if "prelude" in job:
    text = job["prelude"] + text
for key, path in job["subs"].items():
    src = (ROOT / path).read_text()
    assert "'''" not in src, f"{path} contains triple quotes"
    text = text.replace(key, src)
out = ROOT / "kaggle" / "build" / name
out.mkdir(parents=True, exist_ok=True)
(out / f"{name}.py").write_text(text)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": job["id"], "title": job["id"].split("/")[1], "code_file": f"{name}.py",
    "language": "python", "kernel_type": "script", "is_private": "true",
    "enable_gpu": str(job["gpu"]).lower(), "enable_tpu": "false", "enable_internet": str(job.get("internet", False)).lower(),
    "competition_sources": [], "dataset_sources": [], "kernel_sources": [], "model_sources": [],
}, indent=2))
print(out)
