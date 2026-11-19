# SenseNovaKV — plan

Goal: run SenseNova-Loop-9B on an **8 GB NVIDIA GPU** with full context, using
(A) a from-scratch inference engine with fused CUDA kernels, and
(B) an architecture change to its attention/KV design, trained by distillation from the
original model so knowledge and capacity are kept. Target KV: DeepSeek-V4.1-Flash class
(they publish 890 B/token; arXiv 2609.19969).

## Facts pinned from the source (OpenSenseNova/SenseNova-Loop, modeling_sensenova_loop.py)
- Qwen3.5 backbone: Gated DeltaNet linear attention + gated full attention, 3:1.
- 32 physical layers; layers 4–27 run twice (56 executions). Each visit gets its **own** cache
  slot, so full-attention KV = 14 × 2 × 4 heads × 256 × 2 B = **57,344 B/token** (bf16), and
  DeltaNet states are duplicated per visit.
- mHC: 2 residual streams, per layer two router calls (RMS over 8192, 8-output fp32 GEMV,
  sigmoids, 20-iteration log-space Sinkhorn on 2×2) + merge. Many tiny launches in PyTorch.
- Built-in MTP head (usable for self-speculative decoding).
- Weights repo is gated (accept terms on HF; HF token needed as a Kaggle secret).

## Track A — engine (CUDA, tested on Kaggle T4 capped at 8 GB)
1. Fused mHC step (merge + router + Sinkhorn + branch + RMSNorm). DONE (results/mhc_bench_v3):
   ≤1 ulp vs reference; T=1 decode 18 µs/call vs 261 µs torch.compile+CUDA graph, 5.7 ms eager
   (96 calls/token: 1.7 ms vs 25 ms vs 550 ms). Still latency-bound at T=1 (7% of bandwidth):
   next step folds kernel B into the following GEMV prologue.
2. Fused 4-bit weight GEMV with in-register dequant (own format, not GGUF).
3. Fused Gated DeltaNet decode step (conv update, l2norm, gated delta rule, gated norm).
4. Latent attention decode directly over the compressed (FP4) cache; sparse top-k variant.
5. Whole decode step under CUDA Graphs; MTP draft + batched verify so looped weights are
   read once per k tokens.
Every kernel: bit-level or ulp-level check vs a PyTorch reference, and speed vs HF eager,
CUDA-graph eager and torch.compile.

## Track B — architecture change, distilled from the original
Teacher = original SenseNova-Loop-9B. Freeze FFN + DeltaNet (knowledge); train attention
projections and small adapters only.
1. GQA → MLA by the exact (lossless) TransMLA construction, then shrink latent rank stepwise.
2. Cross-loop latent sharing: visit 2 reads visit 1's latent + small learned per-visit residual.
3. Cross-layer latent sharing between full-attention layers; FP4 cache via QAT.
4. Optional top-k indexer (FP4) for sparse attention at long context.
Training: layer-local output matching first (cheap), then short end-to-end KL to teacher.
Compute: Kaggle TPU v5e-8 (needs a JAX port) — decide after the training-free analysis.
Gates vs teacher at every step: MMLU-Pro, TriviaQA (knowledge); GSM8K, BBH subset (reasoning);
RULER (long context). A step is kept only if it passes.

| Step | B/token |
|---|---|
| per-visit cache (today) | 57,344 |
| shared across loop visits | 32,768 |
| MLA latent 256 + 64 rope | 5,120 |
| FP4 latent | 1,440 |
| cross-layer pairs | 720 |
| + FP4 indexer | ≈864 |

## Compute
- Kaggle GPU (T4, sm_75: no bf16 tensor cores, has INT8/INT4 tensor cores) — kernels.
- Kaggle TPU v5e-8 — distillation training.
- InstaBox `first` (4 vCPU / 8 GiB) — CPU-side tooling, conversion scripts, tests.

## Related work to cite and beat
LoopKV (2611.04137), ResidualQuant (2610.10381), Ragged KV (2610.09013), Ouro (2510.25741),
MoR (2507.10524), MHA2MLA (2502.14837), TransMLA (2502.07864). Notes in research/.
