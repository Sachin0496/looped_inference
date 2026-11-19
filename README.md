# SenseNovaKV

Research code for running a looped transformer with a compressed KV cache on small GPUs.

The target is SenseNova-Loop-9B, a Qwen3.5-style backbone (Gated DeltaNet linear attention plus gated full
attention, 3:1) whose layers 4-27 run twice per token. This repo covers four pieces:

- **Compressed KV design.** Each full-attention layer caches a DeepSeek-style latent (FP4 with per-16 scales)
  instead of full K/V. Loop-shared and layer-paired variants shrink the cache further.
- **Layer-local distillation.** The latent K/V path is trained to match each teacher attention output, with all
  FFN and DeltaNet weights frozen.
- **Custom CUDA kernels for Tesla T4.** Fused mHC step, 4-bit GEMV, fused Gated DeltaNet decode, and FP4 latent
  attention, each checked against a PyTorch reference and benchmarked.
- **GGUF to HF converter** for the release weights.

Full details, measurements and open problems are in [REPORT.md](REPORT.md). The plan is in [PLAN.md](PLAN.md).

## Measured results

All numbers are copied from [REPORT.md](REPORT.md). Kernel numbers are on a Kaggle T4. KV numbers are
training-free and use Preview weights.

**Kernels**

| Kernel | Result (from REPORT.md) |
|---|---|
| mHC fused step | Per decoded token (96 steps): **1.7 ms** vs 25 ms (torch.compile + graph) vs 550 ms (eager) |
| 4-bit GEMV, 4096x4096, T=1 | **3.29x** vs cuBLAS fp16, 98% of copy bandwidth |
| 4-bit GEMV, 248320x4096 (lm_head), T=1 | **6.89x** vs cuBLAS fp16, 82% of copy bandwidth |
| Gated DeltaNet decode | Per token (42 visits): **0.61 ms** vs 13.9 ms (CUDA-graph reference) vs 45.5 ms (eager) |
| FP4 latent attention v2, 131,072 context | **737 us** vs 13,773 us SDPA fp16 GQA, **18.7x** |

**KV compression without training** (Preview weights; `fineweb` column is the more reliable one)

| Variant | KV B/token | Smaller by | fineweb PPL delta |
|---|---|---|---|
| base | 57,344 | 1x | 0 |
| pv512_fp4h | 4,032 | 14x | +2.2% |
| share512_fp4h | 2,304 | 25x | +5.4% |
| share256_fp4h | 1,152 | 50x | +13.0% |
| share256_pair_fp4h | 720 | 80x | +18.2% |

These are the measured results only. Distillation is what should close the perplexity gap.

## Status

- Kernels: done, correct against references, benchmarked.
- KV analysis: done (above).
- Layer-local distillation: **results pending.** No distilled model is measured yet, so "knowledge preserved
  at baseline" is not yet shown.
- Decode engine (`engine/runtime/snkv_engine.py`): CPU-tested on a tiny model. The GPU end-to-end run is
  queued.

## Repository layout

| Path | Contents |
|---|---|
| `engine/kernels/` | CUDA kernels: `w4_gemv.cu`, `mhc_fused.cu`, `deltanet_fused.cu`, `kv4_attn.cu` |
| `engine/ref/` | PyTorch references for the kernels |
| `engine/runtime/` | Decode engine (single file, torch and stdlib only) |
| `engine/convert/` | GGUF to HF converter and name map |
| `kaggle/` | Kaggle jobs: benchmarks, KV analysis and release, distillation (`distill/SPEC.md`), engine e2e |
| `kaggle/render.py` | Renders a job into `kaggle/build/<job>/` with sources inlined |
| `results/` | Raw outputs (`results.json`, logs) and `*_summary.md` per benchmark |
| `research/` | Notes on the model, KV literature and design |
| `tests/` | Kernel emulation tests and engine CPU tests |

## Running a Kaggle job

Jobs are defined in `JOBS` in `kaggle/render.py`. Each job runs on Kaggle as a private script kernel.

```bash
python3 kaggle/render.py <job>                  # e.g. mhc_bench, kv_analysis
kaggle kernels push -p kaggle/build/<job>
```

Kaggle allows only two concurrent GPU sessions. The jobs need the Kaggle CLI configured with your own account.

## Weights

Model weights are not included in this repository. SenseNova-Loop-9B is Apache-2.0 and published on
Hugging Face by SenseNova. Download it from there and accept the terms on the model page.

## License

License: TBD by owner.
