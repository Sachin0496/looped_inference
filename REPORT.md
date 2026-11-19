# SenseNovaKV — morning report (2026-11-19, 07:15 IST)

Every number below comes from a results file named next to it, or is marked **estimate** / **pending**.
Weights: SenseNova-Loop-9B (Preview where noted; release weights via our validated GGUF converter).

## 1. Summary
- **Done and measured on a Kaggle T4:** four custom CUDA kernels (fused mHC, 4-bit GEMV, fused Gated DeltaNet decode,
  FP4 latent attention v1 + tensor-core v2), each verified against a PyTorch reference and faster than the PyTorch /
  cuBLAS / SDPA baseline. A release-weights converter (ungated GGUF → HF), validated.
- **KV design measured without training:** a DeepSeek-style compressed latent cuts the KV cache 14× at +2.2% fineweb
  perplexity, 25× at +5.4%, and reaches the DeepSeek-V4.1 class (720–1,152 B/token, 50–80×) at +13–18%.
- **Pending (running on Kaggle now):** layer-local distillation runs that should close that gap (config A on release
  weights, config B on Preview), then config B on release weights, then the end-to-end engine validation.
  **No distilled model is measured yet**, so "knowledge preserved at baseline" is not yet shown.
- **Honest headline:** the kernels are real and fast; the ~1 KB/token KV design is real but costs 13–18% PPL
  until distillation results say otherwise.

## 2. Model facts that matter (from the source code)
- Qwen3.5 backbone: Gated DeltaNet linear attention + gated full attention, 3:1. 32 physical layers; layers 4–27 run
  twice (56 executions). Full-attention layers 3, 7, 11, 15, 19, 23, 27, 31 → **14 KV visits per token**.
- Each loop visit has its own cache slot → **57,344 B/token** (fp16). DeltaNet states also duplicated per visit
  (~87–91 MB per sequence, constant).
- mHC: 2 residual streams; per layer two router calls (RMS over 8,192, 8-output fp32 GEMV, 20-step Sinkhorn on 2×2)
  + merge → 96 mHC steps per decoded token. MTP head available for speculative decoding.

## 3. Kernels on Tesla T4 (measured)

### mHC fused step — results/mhc_bench_v3/results.json
Correct: 0 ulp vs reference up to T=16, ≤1 ulp at T=128.
| T | fused µs | PyTorch eager | PyTorch CUDA graph | torch.compile + graph |
|---|---|---|---|---|
| 1 | 18.1 | 5,733 | 1,488 | 261 |
| 4 | 16.2 | 6,239 | 919 | 291 |
| 16 | 22.5 | 5,884 | 1,078 | 330 |
Per decoded token (96 steps): **1.7 ms** vs 25 ms (compile+graph) vs 550 ms (eager). Still latency-bound at T=1
(7% of bandwidth); next step folds its second launch into the following GEMV.

### 4-bit GEMV (own format, group 64) — results/gemv_bench_summary.md
1,152 correctness checks pass (worst relative error 4.6e-4). Weights rotated past L2 (no cache flattery).
| N×K | T=1 speedup vs cuBLAS fp16 | T=1 % copy BW | T=8 vs cuBLAS |
|---|---|---|---|
| 4096×4096 | 3.29× | 98% | 0.88× |
| 8192×4096 | 3.47× | 105% | 0.95× |
| 12288×4096 | 3.57× | 108% | 0.94× |
| 4096×12288 | 3.32× | 100% | 0.88× |
| 248320×4096 (lm_head) | 6.89× | 82% | 1.08× |
Weak spot: at T=8 (MTP verify batches) it only ties cuBLAS (fp32 FMA bound) → tensor-core path needed.

### Gated DeltaNet decode step — results/deltanet_bench_summary.md
Correct over 64 carried steps: state error 1.5e-7 relative, ≥99.99% of outputs within 1 bf16 ulp.
Per token (42 visits): **0.61 ms** vs 13.9 ms CUDA-graph reference vs 45.5 ms eager. Caveat: T=1 is L2-assisted
(2 MiB state fits the 4 MiB L2); with DRAM-resident states expect ~0.8 ms/token (**estimate**, T=4/16 show 83–89% BW).

### FP4 latent attention — results/attn_bench_summary.md, results/attn_bench_v2/results.json
Cache: 416 B/token/layer (latent 256) or 560 (latent 512) vs 4,096 for the original fp16 GQA cache (7–10× smaller).
Correct: ≤2.3e-6 (v1), ≤5.8e-4 (v2, fp16 tensor cores) relative vs the reference on the dequantized cache.
| context | SDPA fp16 GQA µs | v1 (latent 256) | v2 tensor-core (latent 256) | v2 vs SDPA |
|---|---|---|---|---|
| 32,768 | 3,488 | 372 | 221 | 15.7× |
| 131,072 | 13,773 | 1,462 | 737 | 18.7× |
| 524,288 | 55,010 | 6,097 | 2,651 | 20.7× |
v2 reaches only 14–34% of bandwidth (suspect: FP4→fp16 dequant into shared memory; not profiled). v2's fp16 query
rounding exceeds the 5e-3 gate only at logits ±20–30 (CPU emulation) — check on real-model logits.

## 4. KV compression without training — results/kv_analysis_v4/results.json
Preview weights; 64 WikiText-2 + 64 fineweb-edu sequences × 2,048 tokens per variant. `pv` = latent per visit,
`share` = loop visit 2 rebuilt from visit 1's latent (least-squares map), `pair` = layer pairs share a latent,
`fp4` = FP4 latent (E2M1 + E4M3 scale per 16), `h` = Hadamard. Control pv2048 = baseline (hooks exact).
| variant | KV B/token | smaller by | wiki PPL Δ | fineweb PPL Δ |
|---|---|---|---|---|
| base | 57,344 | 1× | 0 (19.86) | 0 (17.33) |
| pv512_fp4h | 4,032 | 14× | −3.2% | **+2.2%** |
| share512_fp4h | 2,304 | 25× | −0.5% | **+5.4%** |
| pv256_fp4 | 2,016 | 28× | +5.6% | +7.8% |
| share256_fp4h | 1,152 | 50× | +12.3% | +13.0% |
| share256_pair_fp4h | 720 | 80× | +31.4% | +18.2% |
| pv128 | 3,584 | 16× | +62.8% | +24.9% |
Takeaways: (1) FP4 on the latent is ~free (+0.05%); Hadamard does not help on PCA coordinates. (2) The KV is not very
low-rank: rank 256 keeps 70–89% of the variance, 512 keeps 84–97%. (3) Loop visit 2's KV is only partly predictable
from visit 1's latent (linear R² 0.58–0.78) → the DeepSeek-class sizes need training. (Negative wiki deltas at 512 are
unexplained; fineweb is the more reliable column.)

## 5. Distillation (layer-local, Kaggle 2×T4) — kaggle/distill/SPEC.md
Student = latent K/V path per full-attention layer (activation-aware SVD init, per-visit up-projections, FP4 latent +
FP8 RoPE key with QAT), trained to match each teacher attention output; teacher q/o/gate/norms and all FFN/DeltaNet
weights frozen (that is where knowledge lives). Eval end-to-end: PPL + 300-question MMLU + greedy samples vs teacher.
| run | config | KV B/token (formula) | weights | status |
|---|---|---|---|---|
| distill_A_v1 | latent 512, loop-shared, FP4 | 4,416 | release | **running** (since ~01:10) |
| distill_B_v1 | latent 256, loop + pair shared, FP4 | 1,050 | Preview, old absolute lr | **running** (since 22:52, slower than planned) |
| distill_B_v2 | same as B | 1,050 | release | queued (auto-push when B v1 ends) |
Results: **pending**. Both runs exceed the planned ~4.5 h; cause unknown without live logs (CLI cannot stream them).
Kaggle's 12 h cap ends B v1 by ~10:50 and A by ~13:10 at the latest.

## 6. Release-weight converter — engine/convert/gguf_to_hf.py
490/490 GGUF tensors mapped; 0 missing / 0 unexpected / 0 shape mismatches. On the same 2 WikiText sequences the
release model scores NLL 2.811 (PPL 16.6) vs Preview 2.914 (PPL 18.4). Vision tower is not in the GGUF (text-only).

## 7. What went wrong
- kv-analysis failed 3 times before succeeding: MTP file excluded from the download (twice, two code paths), then a
  cross-GPU device mismatch. Each failed within minutes. kv-release v1 hit the same cross-GPU bug (fixed build not yet re-run).
- distill B v1 was launched with a known loss bug and an absolute learning rate that drifted the exact-init control;
  both fixed for later runs (per-tensor relative lr). B v1 is kept as a Preview data point.
- Kaggle allows only 2 concurrent GPU sessions; that, not compute, paced the night. The engine end-to-end job
  (engine/runtime/snkv_engine.py, CPU-tested: 8/8 top-1 vs reference on a tiny model) has not reached a GPU yet.

## 8. Next steps (proposals)
1. Read distill A/B results; if the gap closes, run end-to-end KD (TPU v5e-8, JAX port) with the full gate suite
   (MMLU-Pro, TriviaQA, GSM8K, BBH, RULER) — the "knowledge at baseline" claim needs it.
2. Wire the FP4 latent attention into the engine and validate the whole decode loop under an 8 GB memory cap.
3. Tensor-core GEMV for T=4–8 (MTP verification), then MTP self-speculative decoding.
4. Profile attention v2's dequant path; re-measure DeltaNet with rotated states.

## Files
PLAN.md, OVERNIGHT.md (full log + Kaggle ledger), engine/kernels/*.cu, engine/ref/*.py, engine/runtime/snkv_engine.py,
engine/convert/gguf_to_hf.py, kaggle/*/ (all jobs), results/ (all raw outputs + *_summary.md), research/.
Nothing committed or pushed anywhere.
