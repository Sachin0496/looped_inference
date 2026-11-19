# Efficient KV Handling in Looped / Recurrent-Depth Transformers (2024-2026)

Scope: looped transformers (Huginn, Ouro, MoR, etc.), adaptive depth, and KV compression applicable to small models.
Verification key: **[V]** = read via alphaXiv full text or PDF query in this session. **[A]** = abstract or search snippet only. **[U]** = not verified in this session (from memory or a search title only).

Note: one alphaXiv lookup (arXiv 2402.06732, cited as KIVI) returned an unrelated math paper, so KIVI details below come from how LoopKV and MiniCache describe it, not from the KIVI paper itself.

---

## 1. Looped transformer KV handling

| Paper | Finding on KV across loops | Quality cost | Tag |
|---|---|---|---|
| Huginn / Geiping et al. 2025, [arXiv 2502.05171](https://arxiv.org/abs/2502.05171) | Zero-shot KV sharing: keep a fixed budget and write/read slot `i mod k`. Budget 4 gives MT-Bench 5.86 vs 5.69 baseline (not significant). Early exit attends to the deepest available KV of earlier tokens. | None measured at small budgets. Trained for cache compatibility across depths. | [V] |
| Ouro / LoopLM, ByteDance 2025, [arXiv 2510.25741](https://arxiv.org/abs/2510.25741) | Prefill needs all 4 loop caches (>10 pt GSM8K loss if shared). Decode: last-step reuse gives 78.85 vs 78.92 GSM8K and 80.40 vs 82.40 MATH-500, a 4x cut. First-step-only collapses to 18.73. | Small on the paper's own benchmarks. Conflicts with later measurements (see below). | [V] |
| Ragged KV / Ouro early exit, [arXiv 2610.09013](https://arxiv.org/abs/2610.09013) | Exited tokens alias deeper KV to their last computed entry (page table). Works on Ouro without training; up to 30% FLOPs and KV savings. Also reports that last-KV-only decode drops to ~0 on BBH frontier tasks. | Ragged-aware training fixes mismatch at no extra cost. vLLM throughput gain not realized at batch 32 (773 vs 808 tok/s). | [V] |
| MoR, Bae et al. 2025, [arXiv 2507.10524](https://arxiv.org/abs/2507.10524) | Recursion-wise caching: KV memory (N_r+1)/2N_r. Recursive KV sharing (first recursion only): 1/N_r. | Expert-choice, N_r=3: 42.6 (own cache) vs 41.9 (shared). Text says token-choice benefits from sharing, but Table 15 shows 39.1 to 38.6. Treat as unresolved. | [V] |
| Relaxed Recursive Transformers, Bae et al. 2024 | LoRA per-recursion adapters over a shared block. KV handling not checked. | n/a | [U] |
| Parallel Loop Transformer (PLT), Wu et al. 2025, [arXiv 2510.24824](https://arxiv.org/abs/2510.24824) | First loop's global KV shared with gated local-window KV in later loops; loops run in parallel. | Cited as baseline; LPT beats it at comparable cache. | [V, via citing papers] |
| Shared memory LPT, Monea et al. 2026, [arXiv 2610.02383](https://arxiv.org/abs/2610.02383) | Pretrain with only the first recursion writing full KV; later recursions read it plus a 64-token window. | Improves PPL by 0.4 to 1.1 at matched size; 2 to 4.7x less KV. Hybrid H-LPT: 76-79% less context memory, better PPL than same-size standard transformer. Trained from scratch. | [V] |
| BeNice!, Prairie et al. 2026, [arXiv 2610.31633](https://arxiv.org/abs/2610.31633) | "KV-shareability" depends on stochastic-depth training. Parcae 770M/1.3B fail under naive sharing (+9.75 PPL). BeNice! post-training fixes it, also with MLA and KDA latents. | Post-training fix matches unshared PPL. Up to 23.6% throughput gain and 31.4% hit-rate gain under simulated load. | [V] |
| MELT, Vendrell et al. 2026, [arXiv 2605.07721](https://arxiv.org/abs/2605.07721) | One gated KV cache per layer, updated across loops. Targets Ouro. | Cited in LPT and LoopKV; not read. | [A] |
| Looped Latent Attention, O'Neill & Reid 2026, [arXiv 2607.15456](https://arxiv.org/abs/2607.15456) | Per-loop cache is low-rank but not safely collapsible to one shared state. Learns a low-rank cross-loop codec by distillation. | Frozen model. | [A] |
| Architectures and KV Replay, Licakova et al. 2026, [arXiv 2611.19446](https://arxiv.org/abs/2611.19446) | Train a 6-iteration looped Transformer with KV replay (writer/reader iterations). Stride 3 replay: 2 KV groups, 192 KiB/token vs 576 KiB per-iteration. | CORE 18.71 vs 18.96 per-iteration. Inference-only compression (no training) loses more. GSM8K absolute scores are very low. | [V] |
| LoopKV, Zhou et al. 2026, [arXiv 2611.04137](https://arxiv.org/abs/2611.04137) | Per-(loop, layer) cell choice: INT4, INT2, or reuse final loop's KV. Early loops cannot share; late loops often can. | LoopKV-M: 35% less KV than KIVI-4 and 1.7-2.2x goodput, no significant loss on suite averages. Naive last-loop reuse for decode loses 44 pts on MATH at 8K in their engine, contradicting Ouro's own report. | [V] |
| ResidualQuant, Kim et al. 2026, [arXiv 2610.10381](https://arxiv.org/abs/2610.10381) | Store final-loop KV as INT4 anchor, earlier loops as INT2 residuals with least-squares scaling. | Ouro-1.4B average 62.30 vs BF16 62.58 at 80.7% theoretical KV cut. Up to 5.38x peak throughput on RTX 5090. | [V] |
| LoopVQ, SNU 2026, [arXiv 2611.05304](https://arxiv.org/abs/2611.05304) | Sub-1-bit KV quantization for looped models. | Not read. | [A] |
| Recency is the wrong axis, 2026, [arXiv 2610.34436](https://arxiv.org/abs/2610.34436) | Negative result: keep deepest plane plus a recent window (RIPE). The window fixes replay tails on Huginn but not decoding. On Ouro it fails on prompt KV. | Shows the cost of sharing depends on how converged late loops are (ripeness), not recency. | [V] |
| T-LoopFormer, Yu et al. 2026, [arXiv 2609.15160](https://arxiv.org/abs/2609.15160) | Token-choice router plus exact recursion-wise KV cache (only active tokens per loop). Not shared. | Removing recursion-wise KV: TPOT +0.196 ms; removing router: PPL +2.1. | [V] |
| FlashLoop, 2026, [arXiv 2609.29812](https://arxiv.org/abs/2609.29812) | Lazy updates for sparse KV and residual quantization across loops. | Its official code loses 12.75 pts on RULER-16K vs its dense path (per LoopKV). | [A] |
| Dory-4B, NVIDIA 2026, [arXiv 2611.03998](https://arxiv.org/abs/2611.03998) | Looped model with elastic depth; served in last-loop mode per LoopKV. | Not read. | [A] |

Other 2026 looped-decoding work seen only as abstracts: LoopSpec (2609.17184), LoopAhead (2610.21594), LoopSkim (2611.19896), LoopFill (2611.19649), CopyLoop (2610.38822), LSTM-style shared-KV looped model (2610.33360), Enabling Dynamic Computation (above), Depth-adaptive continuous batching (2608.09444).

---

## 2. Adaptive loop count / early exit and KV caching

- **Ouro** exit gate with threshold on a cumulative distribution, trained with entropy regularization to a uniform prior [V, 2510.25741]. Ragged KV paper argues that this objective makes exits behave like a fixed lower depth (most tokens exit at depth 2 or 3) [V, 2610.09013].
- **Ragged KV** [V, 2610.09013]: difficulty-aware exits via a global prior, plus ragged-aware training. Works with page-table aliasing; the physical KV ratio equals the average exit depth over U.
- **T-LoopFormer** [V, 2609.15160]: exact per-loop KV cache containing only active tokens, gather-scatter for training and inference. Average depth 3.69 on FineWeb-Edu.
- **MoR** [V, 2507.10524]: token routing by expert-choice (causality leak) or token-choice; attention restricted to tokens still active at that depth, which reduces KV and attention cost together.
- **Huginn** [V, 2502.05171]: KL-between-steps exit (5e-4) on MMLU; deepest available KV used for skipped positions.
- **Interaction summary**: early exit saves KV only if (a) skipped positions alias deeper KV (Ragged KV, Huginn) or (b) they are excluded from the cache (T-LoopFormer, MoR). Naive exit breaks causality for later tokens. Results on Ouro disagree with its own report, so treat last-loop reuse as model-dependent.

---

## 3. Extreme KV compression compatible with small models

- **MHA2MLA**, Ji et al. 2025, [arXiv 2502.14837](https://arxiv.org/abs/2502.14837) [V]: converts MHA/GQA to MLA with partial-RoPE removal and joint SVD, using 0.6-1% of pretraining tokens. Llama2-7B: d_kv=16 (87.5% cut) costs 2.3 LongBench pts; with Int4 (HQQ) the combined cut is 96.87% for -2.4 pts. Int2 alone (-87.5%) costs 6.2 pts.
- **TransMLA**, [arXiv 2502.07864](https://arxiv.org/abs/2502.07864) [A]: converts GQA models (Llama, Qwen, Mistral) to MLA with full DeepSeek compatibility; NeurIPS 2025.
- **MLAfication**, [arXiv 2610.17202](https://arxiv.org/abs/2610.17202) [A]: improves MHA2MLA/TransMLA for QK-Norm models.
- **MLA low-bit caches** [A]: RoLaQ [2610.20686] (2-bit latent), QuantMLA [2609.36760], RAQ-MLA [2610.33625], NVFP4-MLA [2611.18972].
- **KIVI** (2-bit asymmetric KV, per-channel keys and per-token values). Details [U] from the paper; LoopKV describes it as INT4/INT2 codes with groups of 32 and fp16 scale and zero point [V, 2611.04137]. Arxiv ID 2402.06732 returned an unrelated paper; the commonly cited ID is 2402.02750 [U].
- **KVQuant** (Hooper et al. 2024) [U]: not checked in this session.
- **MiniCache**, Liu et al. 2024, [arXiv 2405.14366](https://arxiv.org/abs/2405.14366) [V]: cross-layer merging from layer L/2 onward using SLERP on normalized directions with stored magnitudes; token retention for outliers (gamma=0.05); t=0.6. 4-bit MiniCache + KIVI reaches 5.02x compression. LongBench (Llama-2-7B-Chat) 35.44 vs baseline 36.41, against KIVI-2 at 31.51 (3.95x). ~41% memory cut and ~5x throughput vs FP16 at batch 128 on A100. Training-free.
- **CLA** (Brandon et al. 2024), [arXiv 2405.12981](https://arxiv.org/abs/2405.12981) [A/V via citing papers]: adjacent-layer KV sharing with near-lossless results when trained; MoR cites it as the trained analogue.
- **DeepSeek-V3.2 / DSA**, [arXiv 2512.02556](https://arxiv.org/abs/2512.02556) [A]: lightning indexer scores preceding tokens; top-k selection (k around 2048 per NVIDIA's TensorRT-LLM blog [A]). Added to V3.1-Terminus with continued training. The indexer has its own cache. Only search snippets were read.
- **LatentIndex**, [arXiv 2610.04635](https://arxiv.org/abs/2610.04635) [A]: reuses sparse indices across layers.

---

## 4. Gaps for a possible novel paper

1. **Trained cross-loop latent KV with exit-aware aliasing.** BeNice! trains KV-sharing post hoc and only for full depth. Ragged KV aliases exits with no training, and its ragged-aware training only uses dense KV. Nobody trains a looped model with a shared low-rank (MLA-style) latent across loops and per-token exit aliasing at the same time. Risk: LoopKV shows per-loop KV carries information for early loops, so sharing may need a per-loop residual.
2. **Quantization-aware training on loop residuals.** ResidualQuant, LoopKV, and LoopVQ all quantize a model that was not trained for it. A looped model trained with INT2 residual codes against a final-loop anchor (per-cell bit allocation learned, not profiled) is untested.
3. **Cross-loop index reuse for sparse attention.** DSA and LatentIndex reuse indexers across layers, and FlashLoop does sparse updates within loops. No paper reuses a top-k index from one loop for other loops. Risk: LoopKV measures Jaccard 0.83 between loops 3 and 4 for top tokens, so index reuse may fail. A small test would settle it.

---

## Unverified or conflicting items

- Relaxed Recursive Transformers KV details: not read.
- KIVI and KVQuant numbers: not read directly (see section 3).
- DSA details beyond snippets: not read.
- MoR token-choice KV sharing: text and Table 15 disagree.
- Last-loop KV reuse for decoding: Ouro reports near-lossless; Ragged KV and LoopKV report large losses on other benchmarks or with longer outputs.
- Huginn KV-sharing budget numbers come from MT-Bench, a small single-turn benchmark.
