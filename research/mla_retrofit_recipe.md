# Retrofitting SenseNova-Loop-9B full attention to compressed latent KV

Scope: convert the 8 full-attention layers (physical 3, 7, 11, 15, 19, 23, 27, 31) from GQA (16 q-heads, 4 kv-heads, head_dim 256, partial RoPE 64 of 256, theta 1e7, q/k RMSNorm, sigmoid output gate) into an MLA-style latent cache, keeping the teacher's quality. Gated DeltaNet layers, MLPs, embeddings, and the mHC residual wrappers stay frozen.

## 0. Reading status and target facts

Read in full (answer_pdf_queries on the PDF): TransMLA 2502.07864, MHA2MLA 2502.14837, MLAfication 2610.17202, CARE 2603.17946, X-EcoMLA 2503.11132, Attention Editing 2604.05688, KL-guided layer selection 2512.20569, OPAL 2609.31947, GenDistill 2603.26556, MiniLLM 2306.08543, QuaRot 2404.00456, CLA 2405.12981, LoopKV 2611.04137, ResidualQuant 2610.10381, DeepSeek-V4.1-Flash 2609.19969, NVFP4-MLA 2611.18972, DCKV 2610.36950.

Abstract or search-snippet only (not read in full): Taylor-Calibrate 2606.16429, Translating KV caches 2611.12062, YouZhi 2606.05868, GQLA 2605.15250, FlashLoop 2609.29812, Looped Latent Attention (O'Neill & Reid 2026, cited in LoopKV; learns a cross-loop codec by distillation with a frozen model, which is our setting), YOCO and MiniCache (described only through citing papers). X-EcoMLA was located by title search and read through its arXiv ID.

Verified against the local Preview decode spec (`/Users/sachin/Development/SenseNovaKV/research/qwen35_decode_spec.md`, from `modeling_sensenova_loop.py`):
- Loop visits do NOT share KV or DeltaNet state. Physical layers 4..27 are visited twice, so the full-attention KV entries per token are 2 (layers 3, 31) + 6 × 2 (layers 7, 11, 15, 19, 23, 27) = 14. This matches your premise.
- Per entry, bf16: 2 × 4 × 256 × 2 B = 4 KiB. Total 56 KiB/token (28 GiB at 512K context).
- Partial RoPE: 64 rotary dims per head, theta 1e7.
- The output gate comes from the q_proj output (chunk 2 of q_proj reshaped to 2×256 per head), so it is a weight you must preserve exactly.
- Caveat: this is the Preview config. The release config.json returned HTTP 401 earlier. Verify every number against the release before the first run.

## 1. GQA to MLA conversion methods

| Method | Init (what is SVD'd) | RoPE handling | Absorb | Data / tokens | Trainable | Quality evidence |
|---|---|---|---|---|---|---|
| TransMLA 2502.07864 | Merge all KV heads into one latent (identity W_UK/W_UV blocks), then PCA of [K_nope; V] after balancing norms (BKV: scale K_nope by alpha = E‖K_nope x‖/E‖V x‖) | RoRoPE: PCA across heads, rotate so positional content lands in one head; FreqFold to shrink further. Shared K_rope | Yes (W_UK into W_Q, W_UV into W_O) | SmolLM corpus subset; 6B tokens recovers most | Recovery fine-tune | Llama-2-7B: 68.75% KV cut, −1.65 pts training-free (MHA2MLA −21.85 at the same cut). 93% cut still coherent. Does not support QK-norm (per MLAfication). |
| MHA2MLA 2502.14837 | Joint SVD of [W_K; W_V] (SVD_joint beats split by +0.91 on 135M, +1.38 on 1B7) | Partial RoPE: keep r subspaces chosen by head-wise 2-norm (S_2-norm). Default r = d_h/16. Best vs high-freq/low-freq/uniform | Yes | 0.6–1% of pretraining tokens (6B for 1B7, 12B for Llama-2-7B). Data: smollm-corpus (FineWeb-Edu-dedup, Cosmopedia-v2, python-edu, open-web-math, StackOverflow) | Full FT | Llama-2-7B, 68.75% cut: −0.29; 81.25%: −0.30; 87.5%: −1.02 (commonsense avg). LongBench (Llama-2-7B, BF16 27.4): d_kv=64 26.7, d_kv=32 26.0, d_kv=16 25.1 |
| X-EcoMLA 2503.11132 | Static SVD of W_Q, joint SVD of [W_K; W_V] (Alg. 1). W_KR = mean-head K projection's last d_r cols | Shared key-RoPE vector across heads (DeepSeek-style) | Yes | 3.4–7B tokens; 8B teacher; CE + KL (KD dominant) | Full | Llama3.2-1B-Inst (base 52.77): 53.1% KV, +DPO, 7B tok: 53.42; 15.6% KV with 8B teacher, 3.6B tok: same avg (abstract); 9.4% KV, 7B tok: <0.1 pt drop (abstract) |
| MLAfication 2610.17202 | Activation-aware joint low-rank (SVDLLM-V2): eigendecompose XXᵀ, SVD of W_kv·U·Λ^½, unwhiten | Contribution-aware partial RoPE (from MHA2MLA-VLM); d_r = 64 fixed | Yes, plus calibrated KNorm: constant α_h per KV head absorbed into W_uk | SmolLM subset; 300M–2B tokens | QKV + attn norms only (1.6–17.6% of params); o_proj, FFN, routers, embeddings frozen | Qwen3-8B (QK-norm) −81.25% KV: CS −3.35, LB −2.12 vs CPT; Qwen3-30B-A3B (MoE) −81.25%: CS −4.05, LB −2.04. Frozen-router check shows no MoE drift |
| CARE 2603.17946 | Covariance-aware: SVD of √C·W, unwhiten; adjusted (water-filling) rank per layer | Partial RoPE as in MHA2MLA | Yes | 256 calibration samples, seq 32 (saturates by 512). Healing 1B–3B tokens, LR 2e-6 | Healing run | One-shot Llama-3.1-8B: at rank 256, CARE-E avg 52.62 vs SVD-LLM-V2 50.13 vs MHA2MLA 32.08 (GQA original 58.24). Reports above-original scores after healing (63.27 @3B) that I would not trust without an eval audit |
| Attention Editing 2604.05688 | None: MLA randomly initialized, o_proj copied | MLA d_c=512, d_r=64 (FlashMLA shape), query down-proj dropped | Yes (MQA-style decode) | Pretraining-format mix 40% general, 35% math/code, 25% Chinese; <0.1% of Qwen3's pretraining | All attention params | Qualitative only in what I read; no numeric table captured |

Key takeaways for us:
1. TransMLA and MHA2MLA do not handle QK-norm. Qwen3.5 has q/k RMSNorm, so MLAfication's calibrated KNorm is the path. Use it.
2. The positional-channel choice and the activation-aware SVD are the two levers that matter most at high compression. MLAfication Table 6: at 68.75% cut on Qwen2.5-7B, attention-output MSE drops 482 → 87 with activation-aware SVD. Measure this per layer before any training.
3. Learned initialization with random W_DKV/W_UK (Attention Editing) works but needs far more compute. Use it only as a fallback.

## 2. Losses and data for knowledge preservation

| Question | Evidence |
|---|---|
| Logit KD vs NTP | MLAfication Table 5: logit distillation beats NTP on LongBench in all 12 matched settings. Example: Qwen3-8B −81.25% LB 30.29 (KD) vs 26.15 (NTP). X-EcoMLA Table 12 (Llama3.2-1B): KL-only 50.84, CE 1 + KL 0.1 → 50.98, CE-only 48.54. KD carries the signal; a small CE term is harmless. |
| Forward vs reverse KL | MiniLLM 2306.08543: reverse KL with on-policy policy-gradient fixes exposure bias and calibration (SST2 ECE: KD 0.191, SeqKD 0.243, MiniLLM 0.099, teacher 0.025). Needs 5000 PG steps and teacher-mixed sampling. Too costly for us to run every night. |
| On-policy vs off-policy | OPAL 2609.31947: at 400M tokens, MATH-500 87.0 (on-policy student rollouts) vs 78.1 (off-policy dataset) vs 80.0 (teacher traces). AIME'24: 33.3 vs 17.1 vs 17.5. Horizon doubling from 512 to 32K tokens after KL plateaus. |
| Layer-local vs end-to-end | Attention Editing 4.2: block-wise teacher-forced MSE on the post-o_proj output (normalized, eq. 23) first, then end-to-end KD. Without the first stage, output-level distillation stalls. OPAL and GenDistill use the same two-stage pattern. |
| Completion-only masking, teacher-aligned data | GenDistill 2603.26556: KD beats SFT on generation (not on perplexity). Completion-only masking helps generation. Teacher-aligned data (AM-Qwen3-Distilled) beats broader instruction corpora by about 20 pp. Freezing attention during instruction tuning: +5.2 pp generation, <1 pp perplexity. |
| Pretraining-like vs self-generated | MHA2MLA: pretraining-like text recovers the base. GenDistill: teacher-generated text wins for instruction and reasoning. Our plan uses both. |

Perplexity hides generation regressions. GenDistill measured an ~0.2 pp perplexity gap against a 20.8 pp generation gap on QRWKV6-7B. Gate every checkpoint on generation.

## 3. KV quantization at 4 bits and below

| Source | Result |
|---|---|
| QuaRot 2404.00456 | Randomized Hadamard rotation fused into weights removes outliers without changing the model. 4-bit KV asym group-128, LLaMA-2-7B WikiText PPL: K4V4 5.51 (base 5.47), K4V3 5.54, K3V4 5.65, K3V3 5.68, K4V2 5.75, K2V2 9.23. Keys are more sensitive than values. |
| NVFP4-MLA 2611.18972 (NVIDIA, Nov 2026) | Shared MLA latent in NVFP4 with 2D 16×16 blocks (so both BMMs read the same values/scales). The RoPE channels (64 of 576) carry the outliers: a NVFP4 residual on RoPE only cuts RoPE NMSE 115×. FP8 RoPE leaves 9.5× more error than that residual. DeepSeek-R1 avg 59.82 vs BF16 59.80; vanilla NVFP4 drops 5.71 pts (Kimi-K3 drops 16.48). |
| DeepSeek-V4.1-Flash 2609.19969 | Global KV in FP4: E2M1 with one E4M3 scale per 16 channels (NVFP4 without its global scale). Quantize after RoPE. QAT in post-training. SWA stays FP8. Cuts global KV to 890 B/token. The paper says "MXFP4" but describes NVFP4 without the global scale. |
| ResidualQuant 2610.10381 | Looped model (Ouro-1.4B, 4 loops). Final-loop KV as INT4 anchor, earlier loops as INT2 residuals against it, least-squares scale per residual, learned Hadamard-init rotation (OptR-H). Bits per loop [2,2,2,4]. MATH500 75.0 (BF16) vs 76.0 at 80.7% KV reduction. Naive INT2 gives 27.8. |
| LoopKV 2611.04137 | Keys per-channel, values per-token in groups of 32 (KIVI style). Values stored as two 2-bit planes; the attention score decides who reads the low plane. Knapsack over a per-(loop,layer) damage profile (about 3 GPU-hours for 72 cells). LoopKV-M stores 35% less KV than uniform INT4 with no significant suite-average loss. |
| MHA2MLA 2502.14837 LongBench | d_kv=32 latent + INT4 HQQ: 25.8 (BF16 27.4). INT2 on the same latent collapses. |

Practical recipe: rotate (Hadamard) the latent and k_rope before quantizing; per-token scales for latent; RoPE key in FP8 or NVFP4 + residual (per NVFP4-MLA); latent in NVFP4 2D blocks if both attention matmuls run in FP4.

## 4. Cross-layer and cross-loop sharing

| Source | Result |
|---|---|
| CLA 2405.12981 | Trained from scratch (1B and 3B, SlimPajama). MQA+CLA2 halves KV at about equal PPL: H128 MQA 10240 B/tok, PPL 13.54; H128 MQA-CLA2 5120 B/tok, PPL 13.60. At 3B, CLA2 beats its MQA baseline. CLA3/CLA4 are worse than CLA2 at the same memory. Non-uniform sharing patterns are worse. CLA does not reduce core attention bandwidth. No retrofit experiment. |
| DeepSeek-V4.1 CSA2 2609.19969 | Pretrained from scratch. Modes Full / Reindex / Reuse statically assigned. Decoder: 1 Full + 3 Reuse, then 4 × (Reindex + 3 Reuse). Attention-sharing training is in its section 3.1.2 (not read). |
| LoopKV P2 | Loop-3 reading loop-4 KV is near-free in 21 of 24 layers. Sharing into an earlier loop costs about 4× more per cell. Naive last-loop reuse loses 44.4 pts on MATH500 for the Thinking model. |
| ResidualQuant | Anchor = final loop; earlier loops as low-precision residuals. This works for quantization. |

For a 2-visit model, the earlier visit is the costly direction to share. Our candidates:
- (a) Separate latent per visit. 14 entries/token. Safe baseline.
- (b) Visit 2 stores the anchor latent; visit 1 stores a residual against it. ResidualQuant-style, untested for MLA latents.
- (c) Visit 2 reads visit 1's latent through a learned map. Expected to be the worst option by LoopKV's P2.

Decide per cell with a KL profile, as LoopKV does.

## 5. Hybrid and looped models

- DCKV 2610.36950 on Qwen3.5-2B (6 of 24 layers full attention; 12,288 B/token bf16; GDN state fixed at 9.84 MiB/request): a learned, end-to-end-trained codec for the stored prompt cache, keys taken pre-RoPE. About 210 B/token at equal RULER score. Storage-only (not decode), because the decode path needs fixed-width cache. Relevant for prefix reuse.
- OPAL and GenDistill both distill Qwen3-class models into hybrids with the retained full-attention layers kept trainable (OPAL, lr 10× lower than GDN) or frozen (GenDistill, for instruction tuning).
- KL-guided layer selection 2512.20569: "which layers keep softmax" matters a lot at low budget. Not needed for us because all 8 full layers are kept; relevant only if we later convert some of them.
- Looped: LoopKV and ResidualQuant serve Ouro and Huginn with KV per loop, which is our situation. Both are inference-only. Cross-loop codec learning (Looped Latent Attention) is the closest training-based precedent, but I did not read it.

## 6. Recipe for SenseNova-Loop-9B

### 6.1 Target design (per full-attention physical layer)

Dimensions: D = 4096. Query heads h = 16, kv heads g = 4, head_dim 256, rotary dims per head 64.

- Latent: c_t = W_DKV x_t ∈ R^{r}. Start with r = 256.
- Keys: NoPE part K_nope = W_UK c_t for 4 kv heads × 192 dims (after removing the 64 rotary dims per head). RoPE part k_rope = RoPE(W_KR x_t) ∈ R^{64}, shared across kv heads. This needs the RoRoPE/FreqFold step from TransMLA or the MHA2MLA partial-RoPE selection to move positional content into one shared 64-dim key. Use MHA2MLA's 2-norm selection with r_subspaces = 16 (32 rotary dims) as the ablation, and 64 as the default.
- Values: V = W_UV c_t, 4 × 256.
- Storage per token = 14 entries × (r + 64) values. r = 256 gives 8.75 KiB bf16 (6.4× smaller than 56 KiB). r = 128 gives 5.25 KiB (10.7×). With NVFP4 latent, r = 256 gives 2.46 KiB (22.8×).

Calibrated KNorm (MLAfication): replace per-token RMS in k_norm with a constant α_h per kv head (calibrated on 64 sequences; coefficient of variation about 0.5% across context lengths), folded into W_UK.

Output gate: keep the gate rows of q_proj exactly. Apply g ⊙ attn_out before o_proj as in the original.

### 6.2 Initialization math

Calibration: 128–256 sequences of 2K–4K tokens (the teacher's own generated text plus pretraining-like text). Collect X = attention input (after input RMSNorm), pre-RoPE K_nope and V.

1. Balance: α = E‖K_nope x‖ / E‖V x‖. Set W_K^nope ← W_K^nope / α, and W_UK ← α · W_UK (TransMLA BKV).
2. Activation covariance: C = XXᵀ/N = U_x Λ_x U_xᵀ (keep nonzero eigen-space, with damping 0.01·mean eigenvalue, per CARE).
3. Stacked weight W_kv = [W_K^nope; W_V] ∈ R^{(4·192+4·256)×D}.
4. D = W_kv U_x Λ_x^{½}; truncated SVD D ≈ U_r Σ_r V_rᵀ with rank r.
5. Initialize: W_DKV = Σ_r^{½} V_rᵀ Λ_x^{−½} U_xᵀ; [W_UK; W_UV] = U_r Σ_r^{½}. This minimizes ‖W_kv X − Ŵ_kv X‖ over rank r (MLAfication eq. 7–8).
6. Absorption: W_UK folds into W_Q's NoPE part per query head (score path only). W_UV cannot fold into W_O because the output gate is elementwise on the value output. Apply W_UV per query head at decode (16 × r × 256 MACs per token, negligible against KV reads). This is a consequence of the gate that the papers do not discuss; verify numerically.
7. Copy o_proj, MLP, norms, embeddings, mHC wrappers, and all DeltaNet layers exactly.

### 6.3 Trainable parameters

Per full-attention layer, KV side: W_DKV (4096×256 ≈ 1.05M), W_UK (256×768 ≈ 0.20M), W_UV (256×1024 ≈ 0.26M), W_KR (4096×64 ≈ 0.26M), k_norm gains. About 1.8M per layer, about 14M for 8 layers (0.16% of 9B).

Optional stage B: q_proj query rows only (gate rows frozen). Per layer, 4096×4096 ≈ 16.8M. Include only if KV-only recovery plateaus (MLAfication found QKV-only matches full FT).

Frozen: o_proj, MLPs, embeddings, LM head, mHC routers, all DeltaNet layers, and the 16 non-full layers.

### 6.4 Losses

- Stage A (layer-local): per layer ℓ in {3,7,...,31}, teacher input H^{T,ℓ−1}, loss L_blk = ‖Û − U‖²_F / (‖U‖²_F + ε) on post-o_proj outputs (Attention Editing eq. 23). Also cosine on intermediate hidden states at small weight λ_cos ≈ 0.1 (optional).
- Stage B (end-to-end KD): forward KL from teacher to student logits on completion tokens only, temperature τ = 1 (sweep τ ∈ {1, 2}), loss KL(p_T ‖ p_S). Add 0.1 × CE on ground-truth tokens (X-EcoMLA: harmless).
- Stage C (optional, if Stage B leaves a generation gap): on-policy reverse KL on student rollouts, rollout horizon 512 doubling after KL plateau to 8K–16K (OPAL), with teacher-mixed sampling (MiniLLM α = 0.2).
- Important: keep the teacher and student on identical loop plumbing (mHC split reset at loop boundary, mean-pool re-expansion). Otherwise the KD signal is wrong.

### 6.5 Data

Ungated datasets to try (I did not verify each dataset page on the Hub in this session; check gating before pulling):
- Pretraining-like: HuggingFaceFW/fineweb-edu (sample-10BT); HuggingFaceTB/smollm-corpus subsets (cosmopedia-v2, python-edu); open-web-math/open-web-math.
- Teacher-generated (generate with SenseNova-Loop itself, Apache-2.0): prompts from HuggingFaceTB/smoltalk, teknium/OpenHermes-2.5, databricks/databricks-dolly-15k, HuggingFaceH4/ultrachat_200k.
- Reasoning and math prompts: open-thoughts/OpenThoughts-114k, open-r1/OpenR1-Math-220k, BytedTsinghua-SIA/DAPO-Math-17k.
- Long context: synthetic RULER-style needle and multi-key data, generated locally, at 4K, 8K, 16K, 32K.

Mix: Stage A, 60% pretraining-like, 25% math/code, 15% long-context synthetic. Stage B, 40% pretraining-like, 40% teacher-generated (thinking mode on, as in deployment), 20% long-context.

### 6.6 Token budgets, schedule, and learning rates

| Stage | Tokens | Seq len | LR | Schedule | Notes |
|---|---|---|---|---|---|
| Calibration | 0.5M (256 × 2K) | 2K | n/a | n/a | Covariance, α, PCA for RoPE selection |
| A: layer-local | 100M | 4K | 1e-4 (KV params) | cosine, 3% warmup | About 1 night per 25–30M at the stated throughput |
| B: KD end-to-end | 600M–1B | 4K → 16K at last 20% | 5e-5 KV params; 5e-6 for any q_proj rows | cosine to 10% | MLAfication 300M–1.2B; X-EcoMLA 3.4–7B; MHA2MLA ~6–12B |
| C: on-policy (optional) | 100M–200M | rollouts to 16K | 2e-5 | constant + decay | Only if B leaves a generation gap |
| Q: FP4 QAT | 50M–100M | 4K | 1e-5 | cosine | After MLA is stable |

AdamW, β = (0.9, 0.95), weight decay 0.1, grad clip 1.0. Batch: 32 sequences of 4K (MLAfication); 2–4M tokens per step.

### 6.7 Compute and hardware

Per-token FLOPs for full-model KD ≈ 8N with N ≈ 9e9, so about 7.2e10 FLOP/token. Estimates at 30–35% MFU:
- 2×T4 fp16: about 39 TFLOP/s effective, about 540 tok/s, about 15M tokens per 8 hours at the absolute best. T4 has no bf16. A 9B model's full training state will not fit in 2×16 GB without offloading. Use T4 for calibration, evaluation, and generation of teacher text only. Do not run full-model KD on T4.
- TPU v5e-8 bf16: about 550 TFLOP/s effective, about 7,600 tok/s theoretical. Realistic at 10–30M tokens per night with real overheads. Do all KD training here.

The 10–30M per night figure you gave is consistent with this only on TPU. Verify the Kaggle TPU session limits before planning the nightly schedule.

Layer-local Stage A is cheap: each step needs only the attention block of one layer, so it fits on T4 if the teacher activations are cached.

### 6.8 Quantized target

After Stage B, quantize the latent: NVFP4 with one E4M3 scale per 16 channels, Hadamard rotation first (QuaRot), QAT for 50–100M tokens. RoPE key in FP8, or NVFP4 with a residual (NVFP4-MLA). Keep the visit-specific cache layout (14 entries/token). Target 2.5–3.5 KiB/token; about 16–22× vs bf16 GQA.

### 6.9 Visit-sharing ablation

After Stage B, measure per visit pair (layer ℓ, visit v) the KL damage of options (a), (b), (c) from section 4, as LoopKV does. Pick the cheapest per-cell option that holds the KL budget. This is cheap to run: one forward per option.

### 6.10 Eval suite for "no loss vs teacher"

Run teacher and student side by side on identical prompts and seeds. Report paired differences with 95% CI and a sign test (DCKV/LoopKV style).

- Next-token: KL(teacher ‖ student) on held-out text (mean and p95), top-1 agreement (LoopKV reports 99.6% on Ouro).
- Perplexity on held-out web and code.
- Generation (not likelihood only): MMLU-Redux (generative), GSM8K, MATH-500, GPQA-Diamond, HumanEval, IFEval, BBH.
- Long context: RULER at 4K, 32K, 128K; NIAH single/multi-key; LongBench v2.
- Long generation: 8K–16K thinking-mode outputs; fraction of degenerate rollouts (OPAL Fig. 2); exposure-bias error vs length (MiniLLM ExAccErr).
- Systems: measured KV bytes/token, decode tok/s at 16K and 128K, peak memory.

Gate (proposed): non-inferiority margin of 1.0 pt on reasoning averages and 2.0 pts on RULER-32K; any suite below that fails. Report what the 6.4× vs 10× vs 22× settings cost.

Literature expectation: at ~6× KV reduction with an activation-aware init and 100M–1B tokens of KD, expect <1 pt loss. Beyond ~80% reduction, residual gaps of 1–4 pts remained even after distillation in MLAfication and MHA2MLA. Plan the "no loss" claim around 6× in bf16 and treat FP4 as the next step.

## 7. Risks and open questions

1. Preview vs release config. All dimensions above come from the Preview config and code. Re-verify before the first run.
2. fp16 on T4. Qwen-family and DeltaNet kernels can overflow in fp16. Do not train in fp16 on T4.
3. Output gate forces per-query W_UV. Verify numerically that the absorbed decode path matches the reference to within bf16 noise.
4. Visit sharing. Options (b) and (c) are untested for MLA latents. Only (a) is established.
5. Gated DeltaNet state is fixed-size and per visit, so it is not affected by this work. Still, DCKV reports it as about 9.84 MiB/request for Qwen3.5-2B, which dominates the KV at short prompts.
6. Generation vs likelihood. Do not sign off on likelihood-only numbers.
7. Dataset gating and licenses. Check each dataset before pulling.

## 8. Sources

- TransMLA: https://arxiv.org/abs/2502.07864
- MHA2MLA: https://arxiv.org/abs/2502.14837
- X-EcoMLA: https://arxiv.org/abs/2503.11132
- MLAfication: https://arxiv.org/abs/2610.17202
- CARE: https://arxiv.org/abs/2603.17946
- Attention Editing: https://arxiv.org/abs/2604.05688
- KL-guided layer selection: https://arxiv.org/abs/2512.20569
- OPAL: https://arxiv.org/abs/2609.31947
- GenDistill: https://arxiv.org/abs/2603.26556
- MiniLLM: https://arxiv.org/abs/2306.08543
- QuaRot: https://arxiv.org/abs/2404.00456
- CLA: https://arxiv.org/abs/2405.12981
- LoopKV: https://arxiv.org/abs/2611.04137
- ResidualQuant: https://arxiv.org/abs/2610.10381
- DeepSeek-V4.1-Flash: https://arxiv.org/abs/2609.19969
- NVFP4-MLA: https://arxiv.org/abs/2611.18972
- DCKV: https://arxiv.org/abs/2610.36950
- Local Preview spec: /Users/sachin/Development/SenseNovaKV/research/qwen35_decode_spec.md
