# SenseNova-Loop-9B-Preview: one-token decode spec (Gated DeltaNet + gated full attention)

Status: derived from source, not from a running model. Items marked UNVERIFIED are listed in section 10.

## 0. Sources and what they establish

| Item | Source | Notes |
|---|---|---|
| Config | `https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/raw/main/config.json` (fetched via curl, no gate) | `text_config`, `loop_settings`, `mhc` blocks |
| Tensor names/shapes | `.../raw/main/model.safetensors.index.json` (871 keys) | Confirms `model.language_model.layers.{i}.*` names, `mhc_attn`/`mhc_ffn` keys, no `attn` bias |
| Model code that actually runs | `scratchpad/src_dl/SenseNova-Loop/src/sensenova_loop/models/modeling_sensenova_loop.py` (local clone of the SenseNova-Loop repo, 457 lines) | Overrides the stock decoder layer and text forward. Loop plan and MHC live here |
| Mixer math (DeltaNet, attention, norms, RoPE) | `/opt/homebrew/lib/python3.14/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py` | Stock Qwen3.5 classes, reused unchanged by SenseNova |
| Cache layout | `/opt/homebrew/lib/python3.14/site-packages/transformers/cache_utils.py` (`LinearAttentionLayer`, ~line 1017) | Conv/recurrent state allocation |

The stock `modeling_qwen3_5.py` alone does NOT describe SenseNova-Loop-9B. The checkpoint is a shared-weight Loop model with MHC routing, see section 2.

## 1. Config values (text_config unless noted)

| Name | Value |
|---|---|
| hidden_size | 4096 |
| num_hidden_layers | 32 |
| layer_types | linear_attention x3, full_attention, repeated; full at physical 3,7,11,15,19,23,27,31 |
| linear_num_key_heads (Hk) | 16 |
| linear_num_value_heads (Hv) | 32 |
| linear_key_head_dim (Dk) | 128 |
| linear_value_head_dim (Dv) | 128 |
| linear_conv_kernel_dim | 4 |
| key_dim = Dk*Hk | 2048 |
| value_dim = Dv*Hv | 4096 |
| conv_dim = 2*key_dim+value_dim | 8192 |
| num_attention_heads (Hq) | 16 |
| num_key_value_heads (Hkv) | 4 |
| head_dim | 256 |
| attn_output_gate | true |
| attention_bias | false (all projections, all layers) |
| partial_rotary_factor | 0.25 -> rotary dims = 64 of 256 |
| rope_theta | 1e7 |
| rope_parameters.mrope_section | [11, 11, 10] (mrope_interleaved: true, see section 10) |
| rms_norm_eps | 1e-6 (also hard-coded 1e-6 in MHC router and in l2norm) |
| intermediate_size | 12288, hidden_act silu |
| vocab_size | 248320 |
| dtype | bfloat16 (inference default, `inference.py:40`) |
| mamba_ssm_dtype | float32 (config only; not read by the code paths below; the recurrent state is fp32 anyway, see 4.1) |
| loop_settings | start=4, end=28, counts=[2], eval_count=2 |
| mhc | start=4, end=28, streams=2, iterations=20, fixed_identity=false, pre_scale=1.0 |

## 2. Execution plan (applies to every token; this is not stock Qwen3.5)

`modeling_sensenova_loop.py:33-82` `execution_plan`:

    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))

With start=4, end=28, count=2, num_layers=32, the 56 execution slots ("visits") are:

    [0,1,2,3] + [4..27] + [4..27] + [28..31]

Consequences for a decode step:
- Physical weights are shared between the two loop visits of layers 4..27. Each visit has its own cache slot (`_VisitCache`, `modeling_sensenova_loop.py:181-212`). KV cache and DeltaNet state are NOT shared between visits.
- Visit counts: 42 linear-attention visits (3 + 2*18 + 3), 14 full-attention visits (1 + 2*6 + 1). Total 56.
- Full-attention physical layers inside the loop: 7, 11, 15, 19, 23, 27 (6 layers, each visited twice).

MHC (`MHCDecoderLayer`, `modeling_sensenova_loop.py:357-376`) applies to physical layers 4..27 (both visits). Layers 0-3 and 28-31 use the stock residual wiring.

Residual-stream shape across visits (`enter`/`leave`, lines 154-165):
- Visit of layer 4 (first pass): input is the 4096-d embedding; `enter` expands it to 2 streams: `hidden.unsqueeze(-2).expand(..., 2, 4096).contiguous()` (line 157).
- Layers 5..26: residual is 2 streams, stored flattened as 8192-d between layers; `enter` unflattens it (line 160).
- Layer 27 (physical == end-1 == 27): `leave` returns `residual.mean(-2)` (line 165), i.e. the 2 streams are averaged to 4096-d.
- Loop pass 2 starts at layer 4 again with a 4096-d input, so `enter` re-expands it by copying (both streams identical). The stream split is therefore reset at every loop boundary. This is what the code does; it is not an error in this spec.
- Layer 28 receives 4096-d.

Decode runs one token, so B=1, T=1. Batch dimension is kept below for clarity.

## 3. Parameter tensors

Names are the safetensors keys with prefix `model.language_model.` (confirmed in the index). Shapes are [out, in] as stored by `nn.Linear`.

### 3.1 Every layer (both types, MLP and norms)

| Tensor | Shape | Dtype in memory | Notes |
|---|---|---|---|
| `layers.{i}.input_layernorm.weight` | [4096] | bf16 | Qwen3_5RMSNorm, zero-centered (weight init 0, uses 1+w) |
| `layers.{i}.post_attention_layernorm.weight` | [4096] | bf16 | same norm class |
| `layers.{i}.mlp.gate_proj.weight` | [12288, 4096] | bf16 | no bias |
| `layers.{i}.mlp.up_proj.weight` | [12288, 4096] | bf16 | no bias |
| `layers.{i}.mlp.down_proj.weight` | [4096, 12288] | bf16 | no bias |

### 3.2 Gated DeltaNet layer (`linear_attn.*`, physical 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30; 24 physical layers)

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `linear_attn.in_proj_qkv.weight` | [8192, 4096] | bf16 | rows [0:2048]=q, [2048:4096]=k, [4096:8192]=v |
| `linear_attn.in_proj_z.weight` | [4096, 4096] | bf16 | z, 32 heads x 128 |
| `linear_attn.in_proj_b.weight` | [32, 4096] | bf16 | b (beta logits), one per v-head |
| `linear_attn.in_proj_a.weight` | [32, 4096] | bf16 | a (decay input), one per v-head |
| `linear_attn.conv1d.weight` | [8192, 1, 4] | bf16 | depthwise, kernel 4, no bias (`bias=False`, line 522-529) |
| `linear_attn.A_log` | [32] | fp32 | kept fp32 (`_keep_in_fp32_modules_strict`) |
| `linear_attn.dt_bias` | [32] | fp32 | kept fp32 |
| `linear_attn.norm.weight` | [128] | bf16 | Qwen3_5RMSNormGated, plain weight (init ones), NOT zero-centered |
| `linear_attn.out_proj.weight` | [4096, 4096] | bf16 | value_dim -> hidden |

### 3.3 Gated full-attention layer (`self_attn.*`, physical 3,7,11,15,19,23,27,31; 8 physical layers)

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `self_attn.q_proj.weight` | [8192, 4096] | bf16 | 16 heads x (256 query + 256 gate), INTERLEAVED per head (see A1) |
| `self_attn.k_proj.weight` | [1024, 4096] | bf16 | 4 kv heads x 256 |
| `self_attn.v_proj.weight` | [1024, 4096] | bf16 | 4 kv heads x 256 |
| `self_attn.q_norm.weight` | [256] | bf16 | per-head RMSNorm, zero-centered |
| `self_attn.k_norm.weight` | [256] | bf16 | per-head RMSNorm, zero-centered |
| `self_attn.o_proj.weight` | [4096, 4096] | bf16 | 16x256 -> hidden |

### 3.4 MHC routers (physical 4..27 only, 24 layers; both `mhc_attn` and `mhc_ffn`)

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `mhc_attn.weight` / `mhc_ffn.weight` | [8, 8192] | fp32 | output = 2*S + S*S = 2*2 + 4 = 8 logits; input = S*dim = 2*4096 |
| `mhc_attn.bias` / `mhc_ffn.bias` | [8] | fp32 | init: pre bias = logit(0.25, 0.75); post bias = logit(0.25, 0.75); mix bias = diag(log((1-1/16)*16)) (S=2) |

Line refs: `MHCRouter.__init__` lines 98-109, `_keep_in_fp32_modules_strict` line 402/414.

### 3.5 Top level (not per layer, for completeness)
- `model.language_model.embed_tokens.weight` [248320, 4096], `model.language_model.norm.weight` [4096] (zero-centered), `lm_head.weight` [248320, 4096] (untied). Not part of the layer spec.
- `mtp.*` keys (multi-token prediction head) are ignored by the plain decode path.

## 4. DeltaNet decode step (linear_attention visit, T=1)

Input: `h` bf16 [1,1,4096], the layer's own input after the MHC-or-plain wrapper (section 6 gives what `h` is for MHC layers). State in cache: conv state C bf16 [1, 8192, 4], recurrent state S fp32 [1, 32, 128, 128].

State layout details (from `cache_utils.py`): `LinearAttentionLayer.lazy_initialization` (line 1018-1049) allocates conv state with dtype of the incoming tensor (bf16) and shape `(*conv_states.shape[:-1], conv_kernel_size)`, i.e. last dimension 4. Recurrent state is `zeros_like(recurrent_states)`, where the incoming tensor is fp32 (see D8), so S is fp32.

Notation: `x` = bf16 tensors, `f` = fp32 tensors.

**D1. Input projections (bf16 GEMMs, output bf16).**
- `qkv = in_proj_qkv(h)` -> bf16 [1,1,8192]; transpose to [1,8192,1] (`mixed_qkv.transpose(1,2)`, line 566).
- `z = in_proj_z(h)` -> bf16 [1,1,4096], reshape [1,1,32,128].
- `b = in_proj_b(h)` -> bf16 [1,1,32].
- `a = in_proj_a(h)` -> bf16 [1,1,32].

**D2. Causal depthwise conv1d update, decode path (`causal_conv1d_update`, lines 250-267).**
- `xcat = cat([C, qkv], dim=-1).to(weight.dtype)` -> bf16 [1,8192,5]. Here weight = `conv1d.weight.squeeze(1)` bf16 [8192,4].
- `C <- xcat[:, :, -4:]` (copy). After this, C holds the last 4 raw (pre-conv, pre-activation) qkv columns: [c_{t-3}, c_{t-2}, c_{t-1}, x_t].
- `y = F.conv1d(xcat, weight.unsqueeze(1), bias=None, padding=0, groups=8192)` -> length 2; take `y[:, :, -1:]` (line 263). Equivalent: `y[c] = sum_j w[c,j] * window[j]`, window = [c_{t-3}, c_{t-2}, c_{t-1}, x_t], j=0 is the oldest. This is a cross-correlation, weight index 3 multiplies the current token.
- `qkv = SiLU(y)` (`ACT2FN["silu"]`, line 266), then `.to(bf16)`.
- Result bf16 [1,8192,1] -> transpose to [1,1,8192].

Note: the conv state carries 4 columns but only the last 3 past columns contribute. The 4th slot is written but its oldest value is never read in decode.

**D3. Split and reshape (lines 620-629).**
- `q, k, v = split(qkv, [2048, 2048, 4096], dim=-1)` (bf16).
- `q -> [1,1,16,128]`, `k -> [1,1,16,128]`, `v -> [1,1,32,128]`.

**D4. Gates, all computed from bf16 inputs (lines 617, 619).**
- `beta = b.sigmoid()`: computed in BF16 (b is bf16, the sigmoid output is bf16). Shape [1,1,32]. Converted to fp32 later at D6.
- `g = -A_log.float().exp() * softplus(a.float() + dt_bias)`:
  - `a.float()` -> fp32 (exact widening from bf16).
  - `dt_bias` fp32 [32], broadcast.
  - `softplus` is PyTorch default: beta=1, threshold=20 (for input > 20 returns input).
  - `A_log.float().exp()` fp32; multiply by the negative sign.
  - Result `g` fp32 [1,1,32], `g <= 0`.
  - Decay used later: `decay = exp(g)` in (0, 1].

**D5. GQA-style head expansion of q and k (lines 620-622).**
- `r = Hv/Hk = 32/16 = 2`.
- `q = q.repeat_interleave(2, dim=2)` -> [1,1,32,128]; same for `k`. Value head h uses q/k head `h // 2` (consecutive repeats, not tile).
- Still bf16 at this point. v is not repeated.

**D6. Entry to the recurrent kernel (`torch_recurrent_gated_delta_rule`, lines 438-501, decode path because seq_len==1 and state exists, line 627-641).**
- `initial_dtype = q.dtype` = bf16 (line 452).
- Cast q, k, v, beta, decay (= g) to fp32 and transpose to [1, H, T, D]: `x.transpose(1,2).to(torch.float32)` (lines 458-461). So q,k,v are fp32 with the bf16 values unchanged; beta fp32; decay fp32.
- `use_qk_l2norm_in_kernel=True` (passed on line 630/643), so:
  - `q = q * rsqrt(sum_d(q^2) + 1e-6)` (line 464), fp32, per head over Dk=128. Note: eps added to the SUM of squares, not the mean.
  - `k = k * rsqrt(sum_d(k^2) + 1e-6)` (line 465).
- `q = q / sqrt(128)` (line 468). `sqrt(128) = 11.313708498984761`. So the q scale is `1/sqrt(Dk)` applied AFTER l2norm. k is not scaled.

**D7. Recurrent state read (lines 470-475).**
- `S = initial_state.to(value)` -> fp32 [1, 32, 128, 128], index order S[h, k_dim, v_dim].

**D8. Per-token state update, exact order (lines 480-491, T=1, index h over 32 v-heads):**
1. Decay: `S <- S * exp(g_h)` (broadcast over k and v). Line 483-484: `decay_t = decay[:, :, i].exp()[..., None, None]`. Note this is `exp(g)` where `g` is the already-computed fp32 gate from D4, so the exp is applied here.
2. Read kv memory with the DECAYED state: `kv_mem[h, v] = sum_k S[h,k,v] * k[h,k]` (line 487). This is `S^T k`.
3. Delta: `delta[h, v] = (v[h,v] - kv_mem[h,v]) * beta[h]` (line 488). `beta` is the fp32 value of the bf16 sigmoid from D4.
4. Rank-1 write: `S[h,k,v] <- S[h,k,v] + k[h,k] * delta[h,v]` (line 489).
5. Output read with the UPDATED state: `o[h, v] = sum_k S[h,k,v] * q[h,k]` (line 491). q is the scaled, l2-normed q from D6.
- Final state `S` (fp32) is written back to the cache: `update_recurrent_state` does `recurrent_states[state_idx].copy_(...)` (cache_utils line 1104). Stored S is the post-update state for this token.

**D9. Cast output.** `core = o.transpose(1,2).to(bf16)` -> bf16 [1,1,32,128] (line 503-504).

**D10. Gated RMSNorm (`Qwen3_5RMSNormGated.forward`, lines 218-234, applied per v-head over Dv=128).**
Inputs: `core` bf16 [32,128] (flattened to rows), `z` bf16 [32,128]. Steps, exact dtypes:
1. `xf = core.float()`.
2. `xn = xf * rsqrt(mean_d(xf^2) + 1e-6)` (fp32). Normalize BEFORE the gate.
3. `y = weight * xn.to(bf16)` (weight is bf16 [128], result bf16). This is NOT zero-centered: weight init is ones and it multiplies directly.
4. `y = y * SiLU(z.float())` (bf16 * fp32 -> fp32 by type promotion; SiLU is computed in fp32 on z).
5. `out = y.to(bf16)` [32*128] -> reshape [1,1,4096].

**D11. Output projection.** `out = out_proj(out)` -> bf16 [1,1,4096]. No bias.

Return to the caller: bf16 [1,1,4096] which the MHC wrapper or plain residual consumes (section 6).

## 5. Gated full-attention decode step (full_attention visit, T=1)

Cache: K, V bf16 [1, 4, S, 256] per visit (`DynamicCache` stores post-RoPE K and post-norm V). `position` p = number of already-cached tokens (absolute, same for all three mRoPE axes in text-only decode, see section 10 for why this is safe).

**A1. Q/gate projection, per-head interleaving (lines 787-790).**
- `qg = q_proj(h)` -> bf16 [1,1,8192].
- `view(1,1,16,512)`, then `chunk(2, dim=-1)` -> `query` [1,1,16,256] = first 256 values of each head's 512; `gate` [1,1,16,256] = last 256 values of each head.
- So the row layout is per head: rows [512h, 512h+256) are query for head h, rows [512h+256, 512h+512) are gate for head h. Not a [all q | all gate] split.
- `gate = gate.reshape(1,1,4096)` (head-major, same order as attention output later).

**A2. Per-head RMSNorm on q (line 792).** `query = q_norm(query)` over the 256 dims, zero-centered RMSNorm (`Qwen3_5RMSNorm`, lines 841-858): `xn = x_fp32 * rsqrt(mean(x^2)+1e-6)`; `out = (xn * (1 + w_fp32)).to(bf16)`. Then transpose to [1,16,1,256].

**A3. K and V projections (lines 793-794).**
- `k = k_proj(h)` -> [1,1,1024] -> view [1,1,4,256] -> `k_norm` (same zero-centered RMSNorm) -> transpose [1,4,1,256].
- `v = v_proj(h)` -> [1,1,1024] -> view [1,1,4,256] -> transpose [1,4,1,256]. No norm on v.

**A4. Partial RoPE (lines 674-697, `apply_rotary_pos_emb`; cos/sin from `Qwen3_5TextRotaryEmbedding`, lines 143-215).**
- inv_freq (fp32, length 32) `= 1 / (1e7 ** (arange(0,64,2)/64))`, i.e. dim = int(256*0.25) = 64 rotary dims. Computed in fp32 (line 169-174).
- freqs = position * inv_freq in fp32; `cos = cos(freqs)`, `sin = sin(freqs)` in fp32 (line 207-209). `attention_scaling = 1.0`.
- Then cast `cos.to(x.dtype)`, `sin.to(x.dtype)` (line 213). So **cos and sin are rounded to bf16** before use.
- Recomposition (`recomposition_frequencies`, lines 210-215) gives a 64-wide vector `cat(freqs, freqs)`. For text-only decode, all three mRoPE axes carry the same position, so every frequency index resolves to the same value as plain 1D RoPE.
- `rotate_half(x) = cat(-x[32:], x[:32])` on the 64 rotary dims (non-interleaved, GPT-NeoX/half-split).
- Apply to query and key only, in bf16: `q_rot = q[..., :64]; q_embed = q_rot*cos + rotate_half(q_rot)*sin`; `q_embed = cat(q_embed, q[..., 64:])` (lines 682-692). Same for k with 4 kv heads.
- V is not rotated.

**A5. KV cache update.** `k_full = cat(K_cache, k)`, `v_full = cat(V_cache, v)` in bf16, length S+1 (`DynamicCache.update`). Here K is post-RoPE.

**A6. GQA expansion (lines 712-721, `repeat_kv`).** `n_rep = 16/4 = 4`. Query head h uses kv head `h // 4` (expand kv dim then reshape, so kv head j serves query heads 4j..4j+3).

**A7. Attention scores and softmax, scale = 256^-0.5 = 1/16.**
- Reference (eager, lines 724-746): `scores = matmul(q, k_full^T) * scaling` (bf16 matmul, result bf16, then multiply by 1/16 in bf16); `probs = softmax(scores, dim=-1, dtype=float32).to(bf16)`; `out = matmul(probs, v_full)` in bf16.
- Default runtime (`attn_implementation: "sdpa"` in config): fused SDPA with `scale=1/16`, internal fp32 accumulation, no intermediate bf16 rounding of scores. Numerics differ slightly from eager; the CUDA kernel should match SDPA semantics for the default path.
- Causal mask: none needed for T=1 with no padding (all cached keys are in the past).

**A8. Output gate (line 818).** `o = attn_out.reshape(1,1,4096)` (head-major, bf16); `o = o * sigmoid(gate)` where `gate` is bf16 from A1 and `sigmoid` is evaluated in bf16 (torch.sigmoid on bf16 returns bf16). Gate multiplication is applied BEFORE o_proj.

**A9. Output projection.** `out = o_proj(o)` -> bf16 [1,1,4096], no bias.

## 6. MHC wrapper and plain residual (per visit)

### 6.1 Plain layers (physical 0-3, 28-31; no MHC)
Stock `Qwen3_5DecoderLayer.forward` (modeling_qwen3_5.py lines 861-912):
```
r = h
h = input_layernorm(h)          # zero-centered RMSNorm, bf16 out
h = mixer(h)                    # DeltaNet (section 4) or full attention (section 5)
h = r + h                       # bf16 add
r = h
h = post_attention_layernorm(h)
h = mlp(h)                      # down(silu(gate(h)) * up(h)), all bf16 GEMMs and bf16 SiLU
h = r + h
```
MLP detail: `act_fn(gate_proj(x)) * up_proj(x)` with SiLU in bf16 (`ACT2FN[silu]` on bf16 tensor), then `down_proj`.

### 6.2 MHC layers (physical 4..27, all visits)
`MHCDecoderLayer.forward` (modeling_sensenova_loop.py lines 357-376). Let `R` be the residual, shape [1,1,2,4096] bf16 after `enter`. S=2 streams, dim=4096.

**M1. Enter.** `R = enter(h)` (section 2). First visit of layer 4: expand h to 2 identical streams. Layers 5-26 and second visit of 4: `h` is 8192-d flattened, unflatten to [.., 2, 4096] (stream-major). Note for loop pass 2 the input is 4096-d (after mean), so it is re-expanded to 2 identical streams.

**M2. Router (`MHCRouter.forward`, lines 117-143), attention side (`mhc_attn`), in fp32 under autocast disabled:**
1. `flat = R.flatten(-2).float()` -> fp32 [1,1,8192].
2. `n = flat * rsqrt(mean(flat^2) + 1e-6)` (line 129). RMS over all 8192 values (both streams jointly). No learned weight.
3. `logits = linear(n, W, b)` with W fp32 [8,8192], b fp32 [8] (line 130).
4. `pre = sigmoid(logits[0:2])` (line 132), fp32 [1,1,2].
5. `post = 2 * sigmoid(logits[2:4])` (line 133), fp32 [1,1,2].
6. `M = sinkhorn(logits[4:8].reshape(2,2), iterations=20)` (lines 85-91, line 139). Procedure in fp32, in log space: repeat 20 times: `L <- L - logsumexp(L, dim=-1)` (row normalize), then `L <- L - logsumexp(L, dim=-2)` (column normalize). Finally `M = exp(L)`. Result is a 2x2 matrix whose columns sum to 1 and rows approximately sum to 1. Index: `M[s, t]`, `s` = output stream, `t` = input stream.
7. `branch = matmul(bf16(pre).unsqueeze(-2), R).squeeze(-2)`: pre is cast to bf16 (line 140), then `branch[d] = sum_s bf16(pre_s) * R[s, d]` in bf16 matmul. Shape [1,1,4096]. (pre_scale = 1.0, so no scale.)

**M3. Attention branch.**
- `u = input_layernorm(branch)` (line 362), zero-centered RMSNorm, bf16.
- `a = mixer(u)`: DeltaNet (section 4) if the physical layer is linear_attention, else full attention (section 5). Output bf16 [1,1,4096].

**M4. Merge (`MHCRouter.merge`, lines 146-152, autocast disabled).**
- `mixed = matmul(bf16(M), R)`: `mixed[s, d] = sum_t bf16(M[s,t]) * R[t, d]` in bf16. Note M is cast to bf16 here.
- `upd = bf16(post).unsqueeze(-1) * a.unsqueeze(-2)`: `upd[s, d] = bf16(post_s) * a[d]`.
- `R' = mixed + upd` (bf16 add).

**M5. FFN-side router** on `R'` with `mhc_ffn` weights, exactly as M2 (fresh pre2, post2, M2).

**M6. FFN branch.** `b2 = bf16 matmul(bf16(pre2), R')` -> [1,1,4096]; `f = mlp(post_attention_layernorm(b2))` (line 374). Note: the norm and MLP are applied to the router branch `b2`, not to `R'` itself.

**M7. Merge.** `R'' = bf16(M2) @ R' + bf16(post2) * f` (same as M4).

**M8. Leave (lines 163-165).** If physical == 27 (end-1): `h_out = mean over the 2 streams of R''`, shape [1,1,4096] bf16 (`residual.mean(-2)`; torch mean on bf16 accumulates in fp32 internally and returns bf16). Otherwise `h_out = R''.flatten(-2)` -> [1,1,8192] bf16.

The residual `R` is NOT normalized anywhere except inside the router logits (step M2.2, which is a per-token RMS over the flattened streams). Stream values themselves are never RMS-normalized.

## 7. Visit schedule (decode, 56 visits)

| Visit # | Physical | Mixer | Wrapper | Residual form on entry |
|---|---|---|---|---|
| 1-3 | 0,1,2 | DeltaNet | plain | 4096 |
| 4 | 3 | full attn | plain | 4096 |
| 5 | 4 | DeltaNet | MHC | 4096, expanded to 2 streams |
| 6-28 | 5..27 | per layer_types | MHC | 8192 flattened; visit 28 (physical 27) emits 4096 (mean) |
| 29 | 4 (pass 2) | DeltaNet | MHC | 4096, re-expanded to 2 streams |
| 30-52 | 5..27 (pass 2) | per layer_types | MHC | 8192 flattened; visit 52 (physical 27) emits 4096 |
| 53-56 | 28,29,30,31 | DeltaNet, DeltaNet, DeltaNet, full | plain | 4096 |

Visit v in 5..28 is physical v-1; visit v in 29..52 is physical v-25. Visit index formula: physical p in the loop appears at visits p+1 and p+25.

Full-attention visits: physical 7, 11, 15, 19, 23, 27 appear at visits 8, 12, 16, 20, 24, 28 and 32, 36, 40, 44, 48, 52. Adding visit 4 (physical 3) and visit 56 (physical 31) gives 14 full-attention visits in total. 

Visit 1 is physical 0 and visit 56 is physical 31. The physical sequence is `[0,1,2,3] + list(range(4,28)) + list(range(4,28)) + [28,29,30,31]`, with each entry's mixer chosen by `config.layer_types[physical]`.

## 8. Per-token memory (decode, batch 1, bf16 KV, fp32 recurrent state)

Per DeltaNet visit:
- Recurrent state S: fp32 [1, 32, 128, 128] = 32*128*128*4 = **2,097,152 B (2.00 MiB)**. Constant, not per token.
- Conv state C: bf16 [1, 8192, 4] = 8192*4*2 = **65,536 B (64 KiB)**. Constant. (3 of the 4 columns are read; see D2.)
- Per visit total: 2,162,688 B.
- Linear-attention visits = 42 -> **90,832,896 B (86.6 MiB)** per sequence, independent of context length.
- For reference, a non-loop model with 24 DeltaNet layers would be 51,904,512 B (49.5 MiB).

Per full-attention visit, per cached token:
- K: 4 kv heads * 256 * 2 B = 2048 B; V: same. **4096 B (4 KiB) per token per visit.**
- Full-attention visits = 14 -> **57,344 B = 56 KiB per token**. Context S tokens: 56 KiB * S. Example: S = 32,768 -> 1.75 GiB.
- For reference, a non-loop model with 8 full layers would be 32 KiB per token.

MHC residual streams and router buffers are transient, not cached.

## 9. Quoted lines (exact, for checking)

Stock `modeling_qwen3_5.py`:
- L617 `beta = b.sigmoid()`
- L619 `g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`
- L621 `query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)`
- L464-465 `query = l2norm(query, dim=-1, eps=1e-6)` / `key = l2norm(key, dim=-1, eps=1e-6)`
- L296 `inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)`
- L468 `query = query / (query.shape[-1] ** 0.5)`
- L483 `decay_t = decay[:, :, i].exp()[..., None, None]`
- L487-491 `kv_mem = ...`, `delta = (v_t - kv_mem) * beta_t`, `last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)`, `core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)`
- L260-262 `hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)` / `conv_state.copy_(...[:, :, -state_len:])` / `out = F.conv1d(...)`
- L231-232 `hidden_states = self.weight * hidden_states.to(input_dtype)` / `hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))`
- L787-790 `query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)` / `gate = gate.reshape(*input_shape, -1)`
- L792-793 `query_states = self.q_norm(...)`, `key_states = self.k_norm(...)`
- L758 `self.scaling = self.head_dim**-0.5`
- L818 `attn_output = attn_output * torch.sigmoid(gate)`
- L854 `output = output * (1.0 + self.weight.float())`
- L191 `inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, ...)`
- L666-672 `rotate_half`

`modeling_sensenova_loop.py`:
- L82 `return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))`
- L89-90 `logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)` / `... dim=-2 ...`
- L128-133 `flat = residual.flatten(-2).float()` / `normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)` / `logits = ...linear(normalized, self.weight, self.bias)` / `pre = logits[..., :streams].sigmoid()` / `post = 2 * logits[..., streams:2 * streams].sigmoid()`
- L139-140 `mix = sinkhorn(...)` / `branch = torch.matmul(pre.to(residual.dtype).unsqueeze(-2), residual).squeeze(-2)`
- L149-150 `mixed_residual = torch.matmul(mix.to(residual.dtype), residual)` / `branch_update = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)`
- L157 / L160 / L165 `enter` / `leave` expressions
- L362 `branch = self.input_layernorm(branch)`; L372 `residual = MHCRouter.merge(...)`; L374 `branch = self.mlp(self.post_attention_layernorm(branch))`

## 10. Not confirmed / caveats

1. **Kernels.** Stock `causal_conv1d_update`, `causal_conv1d_fn`, and `torch_recurrent_gated_delta_rule` are wrapped with `use_kernel_func_from_hub_with_fallback` / `use_kernel_forward_from_hub` (`fla`, `causal_conv1d`). If those packages are present, the runtime uses fused kernels whose rounding may differ from the torch reference above. This spec describes the torch reference semantics. I did not check which backend a given run uses.
2. **Attention backend.** Config default is `sdpa`; the eager path rounds scores to bf16 before softmax (A7). Confirm the reference you test against.
3. **mRoPE `mrope_interleaved: true`.** The stock recomposition code (lines 210-215) does not read that key. For text-only decode, all three axes are identical, so the result is plain 1D half-split RoPE over 64 dims. This would differ for image/video tokens; out of scope.
4. **Position.** Decode position is `past_key_values.get_seq_length()` (`modeling_sensenova_loop.py:284-286`). I did not trace `get_seq_length` through the `_VisitCache` wrapper for the loop cache; assumed it returns the cached token count.
5. **Stock vs SenseNova norm of `beta`.** `beta` is computed in bf16 (D4), not fp32. Kept as the reference behavior.
6. **Not covered:** MTP head (`mtp.*`, `generate_mtp`), vision tower, final norm and lm_head (only named in 3.5), sampling.
7. **Loop cache semantics.** The DynamicCache is created with `num_hidden_layers = 56` slots (`_loop_cache_config`, line 255-260). I inferred this from the code and did not run it.
8. **Provenance.** The local SenseNova-Loop source is an unversioned clone in the scratchpad (git HEAD not recorded in this spec). The config and index were fetched live from the raw URLs; the model card and GGUF repo were not consulted because the raw files were ungated.
9. **Not run.** No forward pass was executed. All numbers are from reading code and config.
