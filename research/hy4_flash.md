# Tencent Hy4-Flash (HF repo `tencent/Hy4-flash`): research notes

Retrieved 2026-11-18. Sources are listed per claim. "NOT FOUND" means not confirmed in any fetched source. Config values come from a WebFetch summary of the raw `config.json`, so the exact keys are as reported by that fetch. Two fetches agree on the numbers below. I did not download the file myself.

## Naming and status

- The model is not called "HY4 Flash" on Hugging Face. The repos are `tencent/Hy4-flash` (220B, image-text-to-text), `tencent/Hy4-flash-FP8`, and `tencent/Hy4-flash-base` (219B). Source: https://huggingface.co/tencent (org page, 9 of 180 repos visible).
- GitHub: https://github.com/Tencent-Hunyuan/Hy4-flash (listed on the org page: https://github.com/Tencent-Hunyuan).
- Release date: 2026-11-16. Source: https://www.eesel.ai/blog/tencent-hy4 and https://ai-tldr.dev/releases/tencent-hy4-flash/index.md
- Conflict: ai-tldr.dev says open weights are "planned, no date, no licence". The HF repo itself has 148 safetensors shards and a Tencent Hy License. The HF repo is the primary source, so I treat the weights as public. Source: https://huggingface.co/tencent/Hy4-flash/tree/main

## 1. Architecture

| Item | Value | Source |
|---|---|---|
| Total params | 220B including 64B Engram (the spec table gives 155.5B excluding Engram, ViT and DFly2) | https://huggingface.co/tencent/Hy4-flash , https://github.com/Tencent-Hunyuan/Hy4-flash |
| Active params | 12B per token (11.3B text + 0.43B vision) | same |
| Dense vs MoE | MoE. Layer 1 is dense (FFN intermediate 13,824); the other 41 are MoE | README raw |
| Routed experts | 512 routed + 1 shared per MoE layer; 16 routed active per token; MoE latent 1,536 and expert intermediate 1,536 | config + README |
| Layers | 42 backbone (1 dense + 41 MoE). Plus 27-layer ViT (hidden 1,152) | config, README |
| Hidden size | 3,072 (vocab 120,832) | config, README |
| Attention | Hybrid: 12 sparse layers ("IndexPool" / `deepseek_sparse_attention`) + 30 sliding-window (SWA, 512 window), 1:3 ratio. Pattern: 2 sparse, then 10 × (3 SWA + 1 sparse). Both use MLA | README raw, HF model card |
| MLA dims | q_lora_rank 1,024; kv_lora_rank 512; qk_nope 192; qk_rope 64; v_head_dim 256 | config |
| Heads | 32 (sparse layers), 64 (SWA layers). Config top-level num_attention_heads 32, head_dim 64 | README, config |
| KV heads | Config lists num_key_value_heads = 32, but MLA caches a shared latent, so this is not a meaningful KV-cache head count. True KV head count: NOT FOUND | config |
| Sparse selection | Top-2048 tokens (index_topk 2048), index_n_heads 16, index_head_dim 128 | config |
| Context | 1,048,576 tokens | config, README, https://openrouter.ai/tencent/hy4-flash |
| Looping / recurrent depth / weight sharing | NOT FOUND. Config defines 42 distinct layer types and per-layer MoE types | config |
| Multi-token prediction | Not standard MTP. Speculative-decoding drafter "DFly2": 5 layers, 0.85B params, reuses the target's KV cache and compressed-indexer selections from sparse layers 25, 29, 33, 37, 41. num_nextn_predict_layers = 0 | README, config |
| Engram | Hashed bigram/trigram embedding tables, applied at layers 1, 11, 21, 31 (0-indexed; config ids 2,12,22,32). 8 hash heads per n-gram, 10.3M vocab per head, 768 dims per order. Table 63.4B params | README, config |
| Residuals | Identity Hyper-Connections (iHC), 4 streams | README raw |
| Quantisation | FP8 variant `Hy4-flash-FP8`. Engram embeddings stored in FP8 (`engram_fp8_embedding: true`). Serving copy mentions 4-bit; details NOT FOUND | config, README, https://ai-tldr.dev/releases/tencent-hy4-flash/index.md |
| Vendor KV claim | KV cache "1/24 of Hy3 and 1/6 of Hy4". Absolute bytes: NOT FOUND | README raw |

Config checks: model_type `hy_v4_flash`, architectures `HYV4FlashForConditionalGeneration` (text_config `HYV4FlashForCausalLM`), dtype bfloat16, rope_theta 10,000,000 (sparse) and 10,000 (SWA).

## 2. KV cache bytes per token (my arithmetic from config)

Assumptions: bf16 = 2 B, FP8 = 1 B. The cache holds the MLA latent (kv_lora_rank 512 + qk_rope 64 = 576 values per token per layer). I did not verify the cache layout in vLLM/SGLang.

- Sparse layers (12), per token, bf16: 12 × 576 × 2 = **13,824 B (13.5 KiB)**. FP8: 6,912 B.
- At 1,048,576 tokens: bf16 **14.5 GB (13.5 GiB)**; FP8 7.2 GB.
- SWA layers (30): fixed, not growing with context.
  - If latent-only: 30 × 512 × 1,152 B = 17.7 MB.
  - If full heads (64 × (256 K + 256 V) × 2 B = 65,536 B per token per layer): 30 × 512 × 65,536 = **1.0 GB fixed**.
  - Which one applies: NOT FOUND.
- Indexer keys (12 layers, 128-dim): NOT CONFIRMED whether cached separately. If one 128-dim bf16 key per token per layer: 3 KB/token extra.
- Illustrative contrast (not Tencent's number): a 42-layer, 32-head, 256-dim K+V bf16 cache = 1,376,256 B per token, about 100 times the sparse-layer MLA figure.
- Engram table: 63.4B params = 126.8 GB bf16, 63.4 GB FP8. It is a static lookup table, not KV cache.

## 3. Ideas relevant to a ~9B looped model under 8 GB

Only the points below have a source. My judgement is marked as such.

1. **MLA latent KV** (Hy4-flash uses it). Per-layer cache is 576 values instead of full K/V. Source: config.
2. **Sparse global layers plus SWA.** Only 12 of 42 layers grow with context. The SWA layers hold a 512-token window. Source: README raw.
3. **Top-k token selection** (2,048 tokens). Bounds attention compute, not weights. Source: config.
4. **Engram-style offload.** The 63.4B lookup table can sit off-GPU, and the indexes are computed from token n-grams. This decouples parameter memory from the compute path. Source: README. Judgement: a looped model could keep its hash tables in CPU RAM or on disk and page in rows, but the table size would need scaling down for 9B.
5. **FP8 / quantised weights.** Source: config and README. Judgement: needed to fit 9B params in 8 GB.
6. **CLA (cross-layer KV sharing).** Hunyuan-Large pairs GQA (8 KV groups) with CLA, sharing KV every 2 layers. The paper's table says GQA+CLA saves about 95% of KV versus MHA. Vendor docs say CLA halves the KV portion. Sources: https://arxiv.org/abs/2411.02265 (search result, not fetched in full), https://huggingface.co/tencent/Tencent-Hunyuan-Large (search result).
7. **Hybrid Mamba-Transformer.** Hunyuan-TurboS (arXiv 2505.15431) has 128 layers: 57 Mamba, 7 attention, 64 FFN, per the v3 abstract. Source: https://arxiv.org/abs/2505.15431 (search excerpt; the 560B total figure is from a third party).
8. **Drafter sharing KV.** DFly2 reuses the target's KV cache, so speculative decoding adds no KV memory. Source: README.

Not found in Hy4-flash: parameter-shared or looped layers; a sub-linear attention state beyond the sparse/SWA split.

## 4. Inference support and licence

- **vLLM:** Supported. Docker image `vllm/vllm-openai:hy4-flash`, model `tencent/Hy4-flash-FP8`, `--tensor-parallel-size 8`, `--attention-backend FLASHMLA_SPARSE`, speculative method `dfly` with 5 tokens. Source: README raw.
- **SGLang:** Supported. Image `lmsysorg/sglang:hy4-flash`, `--speculative-algorithm DFLY`, `--speculative-dfly-block-size 6`, `--tp 8`. Source: README raw.
- **Transformers:** Not yet in main. Install from PR 50903. Source: README raw.
- **llama.cpp / GGUF:** A community/AngelSlim GGUF is linked, described as a compressed build for testing. Source: HF model card. Supported status in mainline llama.cpp: NOT FOUND.
- **Other:** Docker Model Runner, Ollama, LM Studio (via GGUF). Source: HF model card.
- **Hosted APIs:** Tencent Cloud TokenHub (`hy4-flash`), Vercel (`tencent/hy4-flash`), OpenRouter (`tencent/hy4-flash`, 1 provider). Sources: https://vercel.com/ai-gateway/models/hy4-flash.md , https://openrouter.ai/tencent/hy4-flash
- **Licence:** Tencent Hy License, LICENSE file in the repo (13.9 kB). Terms not read. Source: README raw, https://huggingface.co/tencent/Hy4-flash/tree/main

## Source caveats

- Architecture numbers come from the HF config and README. They are company-published. No arXiv tech report was found for Hy4-flash.
- Third-party pages (ai-tldr, eesel, Yahoo/PR coverage) repeat Tencent's figures. Performance numbers are internal.
- Config fetch summaries were produced by a small model. Key values are consistent across two fetches. I did not check the raw JSON byte-for-byte.
- Not fetched: the modeling `.py` files (not visible in the repo root; `finetune/` and `assets/` not expanded). Transformers PR 50903 was not opened.

## Unanswered

- Whether the SWA layers store MLA latent or full per-head K/V.
- Whether the indexer keeps its own KV cache.
- The true KV-head count.
- Absolute per-token KV for Hy3 or Hy4 (so the 1/24 and 1/6 claims cannot be checked).
- Any looping or weight sharing (none found).
