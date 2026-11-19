# engine/ — how the decode engine is organised

Goal of the engine: run SenseNova-Loop-9B one token at a time on one GPU, from the project's own kernels, and
check it against the HF fp16 teacher. Design rule for v1: correct and simple first. Attention is plain SDPA-style
over a dense fp16 KV cache (no compressed-latent attention yet).

## Layout

| Path | What it is |
|---|---|
| `runtime/snkv_engine.py` | The engine. One self-contained file (torch + stdlib only). `kaggle/render.py` embeds it into the Kaggle job. |
| `kernels/w4_gemv.cu` | Fused 4-bit GEMV: optional RMSNorm prologue `(1+w)`, optional residual epilogue. Used for every linear. |
| `kernels/mhc_fused.cu` | Fused mHC step: optional stream merge, router (RMS, 8 logits, sigmoids, 20-step Sinkhorn), branch RMSNorm. |
| `kernels/deltanet_fused.cu` | Fused Gated DeltaNet decode step: conv update + SiLU, L2 norm, gates, delta rule, gated RMSNorm. |
| `kernels/kv4_attn.cu` | Latent/compressed attention kernel. Not used by the v1 engine (kept for the later step). |
| `ref/*.py` | PyTorch references for the kernels (`w4_ref`, `mhc_ref`, `deltanet_ref`), used by the CPU backend and tests. |
| `convert/gguf_to_hf.py` | GGUF -> HF safetensors converter (release weights). |
| `../tests/test_engine_cpu.py` | CPU tests: packer, plan, whole-model check against HF on a tiny random model. |
| `../kaggle/engine_e2e/engine_e2e.py` | Kaggle validation and benchmark job (template; rendered by `kaggle/render.py engine_e2e`). |

## Weights

Format (same bytes as `ref/w4_ref.py`): `W[n,k] = q[n,k] * scale[n,g] + minv[n,g]`, group `g = k // 64`, q in 0..15.
`qw` uint8 `[N, K/2]`, `meta` fp16 `[N, K/64, 2]`. 0.5625 bytes per weight.

- `export_host(model)` packs every linear on the device where its HF weight lives and returns CPU tensors. The
  packed model is about 6.5 GB, which is why it can be built while HF still holds both GPUs.
- Quantized: all linears, including `embed_tokens` (rows are dequantized on lookup) and `lm_head`.
- Kept exact in fp32: RMSNorm weights, mHC routers (weight + bias), `A_log`, `dt_bias`, conv taps, q/k norms.
- `to_device(host, device, policy)` moves everything to the device with the dtypes the kernels need.

## Execution plan

`build_plan` returns the 56 visits: `[0..3] + [4..27] + [4..27] + [28..31]` (physical layer ids). Visit `v` owns
cache slot `v`, the same slot convention as the SenseNova Loop `_VisitCache`.

- Plain layers (physical 0-3, 28-31): `h += mixer(norm(h))`, then `h += mlp(norm(h))`. The norm is the GEMV prologue,
  and the residual add is the GEMV epilogue.
- mHC layers (physical 4-27, both passes): two sub-blocks per visit.
  - `mhc_step(R, pending_ffn_branch, post, mix, attn router, input_layernorm)` merges the previous sub-block and routes.
    It returns `u` (the normalized attention input).
  - mixer(u) gives `a`.
  - `mhc_step(R, a, post1, mix1, ffn router, post_attention_layernorm)` returns `b2`.
  - mlp(b2) gives `f`. The merge `mix2 @ R + post2 * f` is deferred into the next call.
  - Enter at physical 4 (expand `h` to 2 streams). Leave at physical 27 (merge, then mean over the 2 streams).

Dtype policy (`Policy`): the GPU policy is `act=fp16` (GEMV, attention, MLP, lm_head, the teacher's dtype) and
`mhc=bf16` (mHC, DeltaNet, conv state). The kernels fix these contracts. Boundary casts are bf16 <-> fp16. Every
HF weight is an exact bf16 value, so these casts add no weight error. The residual stream is bf16 inside mHC
layers and fp16 elsewhere, so the teacher's rounding differs from the engine's.

## Per-mixer state

- Full attention: K (post-RoPE) and V of shape `[1, Hkv=4, max_len, 256]`, fp16. Query head `h` uses kv head
  `h // 4`. Output gate: `o * sigmoid(gate)` before o_proj. RoPE: partial, 64 of 256 dims, `cos/sin` rounded to the
  activation dtype as HF does.
- DeltaNet: conv state `[1, 8192, 4]` bf16 and recurrent state `[1, 32, 128, 128]` fp32. The kernel updates both
  in place.

Memory: 57,344 bytes of KV per token (14 full-attention visits x K+V x 4 heads x 256 x fp16). The DeltaNet state is
constant, about 90.8 MB. Projected KV is `57344 * ctx` bytes (32K ≈ 1.75 GiB, 128K ≈ 7.0 GiB); the v1 engine
allocates `max_len` up front.

## Decode step and CUDA graph

`Engine._run()` reads the token and position from device buffers (`tok_buf`, `pos_buf`). It computes RoPE from
device tables with `index_select`, writes KV with `index_copy_`, and uses a mask `arange <= pos` over the full
`max_len`. The step therefore has no host syncs and is the same for every position, so `capture()` records it once.

`capture()` warms up at the last slot (`max_len - 1`) and restores the DeltaNet states afterwards. Keep
`max_len >= context + generated + 2`. The engine is built per context length (`max_len` is part of the graph).

`load_prefill(snapshot)` copies the HF prefill result in: `snapshot_cache` reads the HF `DynamicCache` after prefill
(slot `v` = `cache.layers[v]`, and the conv/recurrent getters). Positions continue from `S`.

## Kernel binding contracts (checked against the sources)

- `w4_gemv(x fp16[T,K], qw, meta, norm_w fp16[K] | None, resid fp16[T,N] | None, eps, rpw, slab)`. Launch config
  `rpw=2, slab=2048`. Requires `N % 8 == 0` and `K % 2048 == 0`. Every linear in the model satisfies this.
- `mhc_step(residual bf16[T,2,4096] (written in place), branch_prev bf16 | None, post_prev fp32[T,2] | None,
  mix_prev fp32[T,2,2] | None, W fp32[8,8192], bias fp32[8], norm_w bf16[4096], iters, pre_scale)` returns
  `[residual, out bf16[T,4096], post fp32[T,2], mix fp32[T,2,2]]`.
- `deltanet_decode(qkv, z, b, a, conv_state, rec_state, A_log, dt_bias, conv_w, norm_w)` (bf16 except rec fp32 and
  A_log/dt_bias fp32). Returns bf16 `[T, 4096]`.

## Tests and checks

- CPU (run with a Python that has torch and transformers): `python3 tests/test_engine_cpu.py`.
  - Packer is byte-identical to `w4_ref.pack`.
  - Plan: 42 linear + 14 full visits, 48 mHC visits.
  - Tiny random SenseNova model (real layer count and plan, small dims). HF runs the same dequantized weights.
    The fp32 plumbing check gives worst relative logit error 4.2e-6 over 8 steps, with 8/8 top-1 agreement.
  - The bf16 side is a sanity bound only. HF itself in bf16 deviates from HF fp32 by 4-20% on this model.
- Kaggle (`sachin0496/sensenovakv-engine-e2e`): 3 prompts x 64 teacher-forced steps. Per step: top-1 agreement,
  max |dlogit|, KL(HF || engine). Then tokens/s at 1K and 8K for HF `generate` and for the engine graph, and peak
  `max_memory_allocated` on cuda:0 at 8K. The job writes `results.json` after every stage.

## Known gaps (v1)

- The CUDA path has not run yet: the GPU job is the first run. The CPU tests cover the Python plumbing and the
  reference kernels, not the compiled kernels.
- Attention reads the whole `max_len` buffer each step (masked), so long-context decode cost is O(max_len), not O(ctx).
- No MTP, no batching, no compressed-latent (FP4 / MLA) cache. `kv4_attn.cu` is not wired in.
- Prefill is done by HF; the engine only decodes.
- KL and top-1 are measured against the HF fp16 teacher on 3 short prompts. This is a smoke-level quality check,
  not a benchmark score.
