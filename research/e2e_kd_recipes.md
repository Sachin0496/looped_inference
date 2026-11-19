# End-to-end logit KD recipes for retrofitting latent (MLA-style) KV caches

Scope: papers read through alphaXiv full text, except where marked "abstract/search only". Numbers are as the authors report them. "not stated" means the value was not in the text I could extract. Eval harnesses differ across papers, so MMLU deltas are only rough comparisons.

## 1. Per-paper recipes

### MHA2MLA (Ji et al.), arXiv 2502.14837, https://arxiv.org/abs/2502.14837
Same paper as "Towards Economical Inference: Enabling DeepSeek's MLA in Any Transformer-based LLMs".
- Loss: LM cross-entropy on pretraining-format text. KD is not mentioned in the sections read.
- Trainable: all parameters in the main recipe. Freezing FFN and RoPE-kept QK is listed as future work.
- Optimizer: LR 1e-4 for all model sizes, 10% warmup, 1-sqrt decay. Batch 64x4 GPUs for 135M/360M/1.7B, seq 2048 (4096 for 7B/13B), 12k steps.
- Token budget: the abstract says 0.6-1% of data. At face value, 12k steps x 256 seqs x 2048 is about 6B tokens, so check the repo.
- Data: SmolLM corpus mix (FineWeb-Edu-dedup, Cosmopedia-v2, Python-Edu, OpenWebMath, StackOverflow).
- Recovery: SmolLM-1.7B at 68.75% KV reduction, MMLU 27.73 untrained, 38.11 after 6B tokens (base 39.27). That is about 11 of 11.5 points recovered, leaving about -1.2.

### TransMLA (Meng et al.), arXiv 2502.07864, https://arxiv.org/abs/2502.07864
- Loss: LM training. KD is not stated in the sections read.
- Trainable: not stated explicitly (implied full model).
- SmolLM-1.7B: LR 1e-4, constant, batch 64, seq 2048, 300M to 1B tokens.
- LLaMA-2-7B: LR 2e-5 (two-stage 1e-4 then 2e-5 for the 93% case), batch 64/256, seq 4096, cosine, 0.5B to 6B tokens.
- Data: same SmolLM mix as MHA2MLA (Table 2 of v5 gives the weights: FineWeb-Edu-dedup 0.70, Cosmopedia-v2 0.15, Python-Edu 0.06, OpenWebMath 0.08, StackOverflow 0.01).
- Recovery: SmolLM-1.7B at 68.75% KV, MMLU 38.60 after 300M tokens (base 39.27, so -0.7). This is the strongest small-budget MMLU number in the set.

### X-EcoMLA (Li et al., AMD), arXiv 2503.11132, https://arxiv.org/abs/2503.11132
Closest analog to our setup: teacher-KL, SVD init, then DPO.
- Loss: per-token KL(teacher || student) on full vocab, no temperature stated.
- Ablation (App. A.5.1, Llama-3.2-1B, 20% of data): CE only gives 48.54 avg vs KL only 50.84 (base 52.77). CE+KL mixes gave 50.7-51.0, no better than KL alone. Conclusion: do not use CE as the main signal.
- Trainable: all parameters ("unfreeze all"). Intermediate LayerNorms between down/up projections are removed (A.5.2 shows this helps).
- Optimizer: AdamW (0.9, 0.98), batch 96, one epoch. LR not stated in the extracted text.
- Data: OpenHermes-2.5, GenQA, Infinity-Instruct (instruction data, not web). Then DPO on UltraFeedback-style pairs at batch 64.
- Tokens: 3.4B to 7B. Llama-3.2-1B at 53.1% KV: MMLU 46.05 base, 42.03 after KD (-4.0), 41.58 with DPO.
- Llama-3.1-8B at 9.4% KV (reproduced by Zebra-Llama): MMLU about 58.6 vs 68.1 base (-9.5).

### EliteKV (Zhou et al., Fudan), arXiv 2503.01586, https://arxiv.org/abs/2503.01586
- Loss: not stated in the text read (uptraining on LM data).
- Trainable: full uptraining (implied).
- Optimizer: AdamW (0.9, 0.95), weight decay 0.1, constant LR set to the pretraining's final minimum LR (value not extracted). Batch 512 seqs x 4096 tokens.
- Data: RefinedWeb. Budget is under 0.6% of the original pretraining tokens (about 12B if LLaMA-2's 2T, my arithmetic).
- Result: LLaMA2-7B at 25% KV, 6-task avg 63.9 vs 64.5 base. MMLU was not in their eval suite.

### MLAfication (Wang, Xie, Ji; Fudan), arXiv 2610.17202, https://arxiv.org/abs/2610.17202
Most directly relevant: KD + QKV-only training + QKNorm handling.
- Loss: forward KL(teacher || student) on logits. Ablation shows next-token CE is worse on both CS and LongBench, by 3-4 pts at the strongest compression.
- Trainable: only Q/K/V-side parameters (latent down/up, RoPE-subspace projections, QKNorm gains), 1.6-17.6% of params. O-proj, embeddings, LM head, FFN and MoE routers/experts are frozen. QKV-only matches full fine-tuning (mean difference -0.015).
- Optimizer: GBS 32, warmup 3%, LR 1e-4 (5e-4 for SmolLM at 81-88% KV). Seq 2K (SmolLM) and 4K (Qwen). Optimizer and schedule not extracted.
- Tokens: SmolLM-1.7B 300M/700M/1B; Qwen3-8B 300M/600M/1.2B; Qwen3-30B-A3B 300M/1.2B/2B.
- Data: SmolLM mix, the same as MHA2MLA/TransMLA.
- Controls: they run CPT on the same data without conversion. CPT alone beats the original Qwen models in 17 of 21 settings, so compare against CPT, not the base model.
- Qwen3-8B (MMLU base 50.14, CPT 600M 50.58): MLAfication at 68.75% KV, 600M tokens, 48.68 (-1.5 vs base). At 81.25% KV, 1.2B tokens, 46.75 (-3.4).
- Qwen3-30B-A3B (MMLU base 53.48): at 68.75% KV, 1.2B tokens, 49.91 (-3.6). At 81.25%, 2B tokens, 47.59 (-5.9).
- SmolLM-1.7B (MMLU base 39.27): at 68.75% KV, 300M tokens, 38.14 (-1.1).
- KNorm handling: KNorm's token-wise RMS denominator breaks matrix absorption. They replace it with a per-layer, per-head calibrated constant (folded into W_uk), keep QNorm, and train QKNorm gains. Removing KNorm outright was worse (MoE load-balance violations up 25-63%).

### CARE (Zhou et al.), arXiv 2603.17946, https://arxiv.org/abs/2603.17946 (ICLR 2026)
- Mainly an initialization method (covariance-weighted SVD, adjusted per-layer rank) followed by a short "healing" SFT.
- Healing: LR 2e-6, AdamW (wd 0.01), cosine, 1B and 3B tokens. Loss not stated in the extracted text.
- Llama-3.1-8B-Instruct at MLA rank 512: 3B tokens gives 63.3 avg vs 58.2 for the original model.
- Caution: their base MMLU is 48.05 while the healed model reports 71.6. That is an eval-setup mismatch, so do not use CARE's MMLU to calibrate gaps.

### Attention Editing (Cheng et al., China Merchants Bank), arXiv 2604.05688, https://arxiv.org/abs/2604.05688
Closest in spirit to "train new KV maps with KD on a frozen teacher": the new attention is randomly initialized and trained with progressive distillation.
- Stage I (layer-wise teacher forcing): each edited layer gets the teacher hidden state as input and is trained with a normalized MSE on the post-o_proj attention output, ||U_S - U_T||^2 / ||U_T||^2. 2B tokens at 8K seq.
- Stage II (model-level): KL(teacher || student) on next-token distributions with temperature tau (scaled by tau^2), over all non-pad positions. Plus a low-weight cosine loss on selected intermediate hidden states. 6B tokens in three 2B curriculum segments.
- Trainable: all attention parameters except o_proj (kept frozen and reused). FFN frozen. LR and batch not extracted.
- Data: Stage I is 40% general, 35% math/code, 25% Chinese. Stage II moves toward harder math/code/reasoning. Pretraining format, no prompt/answer split.
- Qwen3-8B few-shot MMLU: 74.47 base, 71.77 MLA (-2.7). Qwen3-30B-A3B: 78.68 base, 76.99 MLA (-1.7). C-Eval (thinking): 83.8 base, 79.2 MLA (-4.6).
- Target: MLA with d_c=512, d_r=64, full-rank Q (no q down-projection), d_NoPE=64.

### Zebra-Llama (Yang et al., AMD), arXiv 2505.17272, https://arxiv.org/abs/2505.17272
Hybrid Mamba2 + MLA composition. Closest published setup to a hybrid base.
- Stage 1 (ILD): per-layer MSE between the attention output and the MLA/Mamba2 student output, all layer parameters trainable. 1.36B tokens, LR 8e-5 (1B, 3B) or 4e-5 (8B), batch 96/48.
- Stage 2 (end-to-end): KL distillation on instruction data, 5.4B to 9.4B tokens, LR 8e-5 (1B) or 4e-5 (3B/8B), AdamW (0.9, 0.8), 1% warmup, cosine, seq 2048 packed.
- Stage 3: DPO, 0.2B tokens, LR 5e-7.
- Data: OpenHermes-2.5, GenQA, Infinity-Instruct (instruction-only).
- MMLU: 1B model with 8 MLA layers, 38.9 vs 46.1 base (-7.1). 8B with 16 MLA layers, 58.2 vs 68.1 base (-10.0). Average accuracy stays within about 1 pt.
- Their own data ablation (App. F, Table 15): Llamba trained on this instruction mix drops MMLU from 38.1 to 27.4. Re-training with FineWeb-Edu-filtered data restores it. Data mix is the main MMLU lever in this family.

## 2. Side-by-side

| Paper | Loss | Unfrozen | LR | Tokens | Seq | Data | MMLU gap |
|---|---|---|---|---|---|---|---|
| MHA2MLA 2502.14837 | CE | all | 1e-4 | ~0.6-1% (ambiguous, ~6B at face value) | 2K/4K | SmolLM mix | -1.2 at 68.75% KV |
| TransMLA 2502.07864 | LM (CE) | not stated | 1e-4 / 2e-5 | 0.3B-1B (1.7B), 0.5-6B (7B) | 2K/4K | SmolLM mix | -0.7 at 300M |
| X-EcoMLA 2503.11132 | KL, CE hurts | all | not stated | 3.4-7B | not stated | instruction | -4 (1B, 53% KV), ~-9.5 (8B, 9%) |
| EliteKV 2503.01586 | LM | full | final pretrain LR | <0.6% of pretrain | 4K | RefinedWeb | not measured |
| MLAfication 2610.17202 | KL > NTP | QKV + QK-norm only | 1e-4 (5e-4 at high compression) | 0.3-2B | 2K/4K | SmolLM mix | -1.5 (8B, 69%), -3.6 (30B-MoE, 69%) |
| CARE 2603.17946 | not stated | not stated | 2e-6 | 1-3B | not stated | Alpaca/C4 | not comparable |
| Attention Editing 2604.05688 | KL + cos | attn except o_proj | not stated | 2B + 6B | 8K | general/math/code/Chinese | -2.7 (8B), -1.7 (30B-A3B) |
| Zebra-Llama 2505.17272 | KL (+ILD MSE) | all MLA/M2 | 8e-5 / 4e-5 | 1.4B + 5-9B | 2K | instruction | -7.1 (1B), -10.0 (8B) |

## 3. Recommendation for our setting

1. Loss: pure forward KL(teacher || student) over the full vocab, T=1 as the default, with no CE term. X-EcoMLA showed CE alone hurts and mixing it in adds nothing. If you want a temperature sweep, use {1, 2}, scaling the loss by T^2 if T != 1 (Attention Editing does this). Keep the layer-local MSE only as a short warm-start or a tiny regularizer on attention outputs. Do not make intermediate losses the main objective; Attention Editing keeps the cosine term low-weight.

2. Trainable set: latent down/up projections, the K/V-side tensors, the q_proj that absorbs W_uk, and any QK-norm gains that survive the rewrite. Keep o_proj frozen (Attention Editing and MLAfication both do; X-EcoMLA trains everything). Keep FFN, embeddings, LM head, and all gated-DeltaNet layers frozen. If k_norm is present, follow MLAfication: replace it with a calibrated per-layer, per-head constant so W_uk can be absorbed. Otherwise the latent cache must be materialized.

3. Relative LR: no paper uses per-tensor relative scaling, so this is my own mapping. The absolute LRs in the literature are 1e-4 for 1-8B models (MLAfication, MHA2MLA, TransMLA-1.7B) and 2e-5 for 7B (TransMLA). With weight RMS around 0.01-0.03, 1e-4 absolute corresponds to about 3e-3 to 1e-2 relative per step. Sweep relative eta in {1e-3, 3e-3, 1e-2}, with 3% warmup and cosine or 1-sqrt decay. Check the actual weight RMS first, since that is what determines the mapping.

4. Tokens: 50-100M is at the bottom of the reported range. TransMLA gets to -0.7 MMLU at 300M tokens on a 1.7B model, and MLAfication's smallest runs are 300M. Plan a gated schedule: stop at 100M, measure MMLU, and extend to 300M if the gap is still above about 1.5 points. Log MMLU every 25M tokens. Also run a CPT control on the same data with no teacher, as MLAfication does, so you can separate the effect of the data from the effect of KD.

5. Sequence length 2K-4K, batch 32-64 sequences.

6. Data: use a FineWeb-Edu-dominant mix with the SmolLM ratios (about 70% FineWeb-Edu, 15% Cosmopedia-like synthetic, 8% math, 6% code, 1% StackOverflow). This is the only mix replicated across several papers on MMLU-type metrics, and Zebra's Table 15 points to FineWeb-Edu as the MMLU lever. Avoid instruction-only data (X-EcoMLA and Zebra both lost 4-10 MMLU points with it). If you add MCQ-formatted data, use only the MMLU auxiliary-train split to limit test contamination (my suggestion, not from the papers).

7. Eval: measure MMLU with the same harness settings as the teacher, 0-shot and 5-shot. CARE's base-vs-healed mismatch shows how easily an MMLU comparison breaks.

Expected outcome: the literature suggests about -1 to -1.5 MMLU residual after 300M-600M tokens at 68-70% KV for 1.7-8B dense models. Your 3.7-point gap is consistent with the upper end of that range, so target under 1.5 points.

## 4. Gaps and what I did not verify

- Not found: a paper with the exact "frozen gated-DeltaNet + full-attention hybrid, train only new KV maps with KD" recipe. Zebra-Llama (hybrid composition) and Attention Editing (progressive KD onto frozen weights) are the closest.
- Not found: a LoRA-style KD paper for compressed attention in this pass.
- Read at abstract or search level only: QK-Normed MLA (https://arxiv.org/abs/2606.16310), GaugeRoPE (https://arxiv.org/abs/2611.20524), RoLaQ (https://arxiv.org/abs/2610.20686), CARE's training details beyond Appendix H, Palu (https://arxiv.org/abs/2407.21118). For Palu, the search summary said it is post-training and does not depend on fine-tuning, but I did not confirm this in the full text.
- Learning rates, optimizers, and temperatures are missing for several papers because they were not in the extracted text. Check the repos before relying on any of them.
