# B2 spec — layer-local distillation of a compressed-latent KV design (Kaggle 2xT4)

Goal: replace each full-attention layer's K/V path with a DeepSeek-style compressed latent, trained so each layer's
attention output matches the original (teacher) model, then measure end-to-end quality vs the teacher.

## Model facts (SenseNova-Loop-9B, Preview weights for now)
Full-attention physical layers FULL = [3,7,11,15,19,23,27,31]; looped (run twice, visits 0 and 1) = [7,...,27].
Per layer: q_proj (4096 → 16 heads × 256 × 2: query and output gate, interleaved per head: view(..., 16, 512) then chunk),
k_proj, v_proj (4096 → 4 × 256), q_norm/k_norm = Qwen3_5RMSNorm over head_dim 256 with (1 + w), partial RoPE:
the first 64 of the 256 dims rotate (rotate_half convention, apply_rotary_pos_emb from transformers qwen3_5), GQA 16/4,
scaling 1/16, output = o_proj(attn_out * sigmoid(gate)). See transformers Qwen3_5Attention.forward (quoted in
research/qwen35_decode_spec.md) — reuse its functions, do not reimplement RoPE.

## Student attention (per physical layer L; q path, o_proj, gate, q_norm, k_norm are the teacher's, frozen)
- latent c = W_dkv · x, dims r (config), one latent per stored slot.
- k_nope[g] = W_uk[visit][g] · c   (192 dims, g = 4 KV groups) ; v[g] = W_uv[visit][g] · c (256 dims)
- rope key: rope_mode "group": k_rope[g] = W_kr[g] · x (4 × 64, stored); rope_mode "shared": one k_r = W_kr · x
  (64 dims, stored) and k_rope[g] = A[visit][g] ⊙ k_r where A is a learnable per-frequency complex scale
  (pairs of dims matching rotate_half's pairing: dims i and i+32), which commutes with RoPE.
- k = k_norm(concat(k_rope[g], k_nope[g]))  (dims 0..63 = rope part, 64..255 = nope part — verify the ordering against
  apply_rotary_pos_emb / partial rotary in the transformers code before training!), then RoPE as the teacher does.
- share_loop: for looped layers the latent (and rope key) is computed only at visit 0 and reused at visit 1 at the same
  positions, with visit-1's own W_uk/W_uv (and A). pair_share: for pairs (7,11) (15,19) (23,27) the second layer uses the
  first layer's stored latent/rope key with its own up-projections. (Stored slots = 8 with share_loop, 5 with both.)
- QAT: fake-quantize c to FP4 (E2M1 values, E4M3 scale per 16, straight-through estimator) after an optional fixed
  Hadamard rotation of c; rope keys fake-quantized to FP8 E4M3 with one scale per (token, group). Toggle by config.
- KV bytes/token reported = slots × (r/2 + r/16 [FP4] or 2r [bf16]) + rope bytes (64 or 256 dims × 1 byte + 2 per scale).

## Initialization (no random init)
From a calibration pass (~64 sequences × 2048 of fineweb-edu, disjoint from eval) collect, for each (layer, visit),
the covariance of the attention input x (4096×4096, fp64 accumulate). Teacher map M = [W_k_nope rows ; W_v] (1792×4096
after selecting nope dims per group). Activation-aware SVD: SVD of M · S where S = chol(Cov_x) (or Cov_x^{1/2});
W_uk|W_uv init = U_r Σ_r split back into groups, W_dkv = V_r^T S^{-1}. For a shared/reused latent (visit 1, pair
second layer) initialise its up-projections by least squares from the source latent (closed form from covariances,
like kv_analysis.py) — or simply copy and let training fix it; log which. W_kr init = teacher k_proj rope rows
(group mode) or the top principal direction set (shared mode) with A initialised by least squares. Report the
initial per-layer relative output error before training.

## Training (stage A, layer-local)
Teacher forward in fp16, no grad, with the residual stream ALWAYS driven by the teacher (student outputs only feed the
loss). For each visit: loss = ||y_S − y_T||² / ||y_T||² where y = o_proj output of that attention call. Sum over
visits/layers, one backward per sequence. Trainable: W_dkv, W_uk, W_uv, W_kr, A (fp32 master weights, AdamW,
lr 3e-4 cosine to 3e-5, warmup 50 steps, wd 0, grad clip 1.0). Data: fineweb-edu sample-10BT streaming, seq 2048,
documents disjoint from calibration and eval. Budget: SNKV_TRAIN_HOURS (default 3.0) wall-clock. Log loss per layer
every 20 steps. Throughput reference: teacher forward ≈ 313 tok/s on 2xT4 with the torch DeltaNet fallback; first try
`pip install flash-linear-attention causal-conv1d` and keep it only if a 1-sequence logits check vs the fallback matches
(max |Δlogit| < 0.05) — otherwise use the fallback.

## Evaluation (end-to-end: student attention replaces the teacher's everywhere, real cache semantics)
- Perplexity on wikitext-2 test and held-out fineweb-edu, 32 × 2048 each, teacher vs student (same sequences).
- Knowledge: cais/mmlu "all" test, fixed random 300 questions, 0-shot, score = argmax log-likelihood over " A"/" B"/
  " C"/" D" after a standard prompt; teacher vs student accuracy.
- 3 greedy generations (64 tokens) from fixed prompts, teacher vs student, saved as text.
- Write results.json: config, kv_bytes_per_token, per-layer init/final loss, ppl/mmlu teacher & student, timings.
  Save trained student params (only the new tensors) to /kaggle/working/student_<config>.pt.

## Configs (env SNKV_CONFIG)
- "A": r=512, share_loop, rope group, no pair, FP4 latent + FP8 rope (QAT on), Hadamard on.
- "B": r=256, share_loop, rope shared, pair_share, FP4 + FP8 QAT, Hadamard on.   (≈1 KB/token target)
- "C": r=512, no sharing, bf16 (no QAT) — the easy reference point.
