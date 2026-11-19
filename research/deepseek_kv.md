# DeepSeek V4 / V4.1 KV cache: findings (as of 2026-11-18)

Confidence key: [P] primary source read (arXiv HTML / HF model card / config fetch via WebFetch, which is an LLM-summarized fetch; verify raw files before quoting). [S] secondary write-up only. [C] my arithmetic.

## Bottom line
- The "840 bytes/token" figure is NOT FOUND in any source. The published number is **890 bytes/token** for DeepSeek-V4.1-Flash global (HBM-resident) KV cache [P: arXiv 2609.19969 abstract; HF model card].
- No accounting from the documented facts gives exactly 840. The closest natural reconstruction gives ~900 (see section 2). The 890 derivation was NOT found in the arXiv HTML text read (chars 0-200k of ~212k).

## 1. Attention / KV design (V4.1-Flash)
- Paper: "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression", arXiv 2609.19969, submitted 2026-09-17 [P] https://arxiv.org/abs/2609.19969 ; HTML https://arxiv.org/html/2609.19969v1
- Architecture: Causal Encoder-Decoder (CED), 40 layers = 20 encoder + 20 decoder [P: arXiv §2.1, §4.2.1; HF card]
- Attention: CSA2 (Compressed Sparse Attention 2). Encoder: layers 1-2 SWA-only, layers 3-20 CSA2 with m=2. Decoder: layers 21-40 CSA2 with m=1 [P: arXiv §4.2.1]
- Query heads 64, head dim 512, query compression dim 1280; 1 KV head (MQA) [P: config; arXiv §4.2.1]
- KV latent: 512 channels [P: arXiv §2.4.4]. Not named kv_lora_rank in the paper.
- RoPE dim: 64 (qk_rope_head_dim) [P: config]. Paper does not state it.
- Sliding window n_win = 128 [P: arXiv §4.2.1; config sliding_window=128]
- Indexer: 32 heads, head dim 128, top-K = 512 [P: arXiv §4.2.1; config index_topk=512]
- Storage: main KV FP4 (E2M1, one E4M3 scale per 16 channels); SWA KV FP8; indexer Q/K FP4 (QAT); Engram embeddings FP8 [P: arXiv §2.4.4, §2.4.2]
- Cross-layer sharing: groups of Full / Reindex / Reuse. Encoder: 3 groups of (1 Full + 5 Reuse). Decoder: group 1 = Full + 3 Reuse; groups 2-5 = Reindex + 3 Reuse. Reindex/Reuse share main KV and indexer K from the most recent Full layer; Reuse also shares top-K indices [P: arXiv §4.2.1, §2.3.1, Fig 4]
- Decoder global KV is projected from the encoder's final hidden state H_{L/2} (Eq. 1) [P: arXiv §2.2]
- Token compression: CSA2 compresses 2x (encoder) or 1x (decoder) along sequence; HCA (V4-Pro, 128x) not in V4.1-Flash config [P]
- Persistent (SSD/host) KV: ~1/8 of V4-Flash; SWA KV moved to host-DRAM pool with TTL; "SWA Bounded Replay" rebuilds SWA state [P: arXiv §3.2.1]
- Conflict: config.json lists 40 hidden layers but compress_ratios has 43 entries (last 3 = 0, likely MTP). Flagged, not resolved.

Other V4 family (non-1) [P: HF blog https://huggingface.co/blog/deepseekv4 ; configs]
- V4-Pro: 61 layers; V4-Flash: 43 layers; both head_dim 512, 1 KV head, qk_rope 64, sliding_window 128, KV dtype FP8 with BF16 RoPE dims, indexer FP4.
- HCA (128x compression, dense MQA over compressed blocks) + CSA (4x, top-k). Layers alternate CSA/HCA.
- Config sources: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json , https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json , https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json
- Note: V4.1-Flash model card (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) describes 552B backbone; Safetensors metadata 763B.
- kv_lora_rank: NOT FOUND in V4/V4.1 configs. V4 uses unified head_dim 512 (latent-style) instead [P config; S community repro].

DeepSeek V3.2 (for baseline) [P: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json]: 61 layers, kv_lora_rank 512, qk_rope_head_dim 64, qk_nope 128, v_head 128, index_n_heads 64, index_head_dim 128, index_topk 2048, FP8 quant.

## 2. Per-token KV bytes (arithmetic)
Element sizes from stated formats [C]:
- FP4 main KV entry (512 ch): 512*0.5 + 32 scales*1 B = 288 B
- FP4 indexer K entry (128 ch): 128*0.5 + 8 scales = 72 B
Entries per token (only Full layers own storage; Reuse/Reindex share) [C, assumption]:
- Encoder: 3 Full layers at m=2 -> 1.5 entries/token
- Decoder: decoder global KV from H_{L/2} -> 1 entry/token (or 1 Full decoder layer at m=1 if read per-layer)
Totals [C]:
- Main KV only: 2.5 x 288 = **720 B/token**
- Indexer K only: 2.5 x 72 = 180 B
- Main + indexer: **900 B/token** (closest to 890; difference 10 B unexplained)
- Main + indexer for encoder only: 828 B
- No scales (256 B FP4 + 64 B indexer): 800 B
- Per layer average (900/40): ~22.5 B/layer
- Nothing gives exactly 840 with documented sizes.
- Excluded from 890 per secondary source: SWA state, allocator overhead, activations, duplication [S].
- V4-Flash 3,514 B/token [S: aiweekly / secondary; not seen in paper text] -> 890/3514 = 0.25, consistent with "1/4" claim.

Baselines [C from configs]:
- GQA8 bf16, head_dim 128: 8*128*2(K,V)*2 B = 4096 B/layer; x40 layers = 163,840 B/token
- DeepSeek V3 / V3.2 MLA bf16: (512+64)*2 = 1152 B/layer; x61 = 70,272 B/token
- Ratio V4.1 (890) vs V3-style MLA bf16 (70,272): ~79x [C]. Paper cites 437x vs V1 (Fig 1b caption) [P].

Other published ratios [P: HF blog]: V4-Pro at 1M ctx: 27% FLOPs and 10% KV vs V3.2; V4-Flash: 10% FLOPs, 7% KV vs V3.2; V4 KV ~2% of GQA8 bf16 (baseline unclear per variant).

## 3. Kernels
FlashMLA [P: https://github.com/deepseek-ai/FlashMLA]
- Current: NVIDIA SM100 / SM103 (Blackwell) only; Huawei Ascend 950. Hopper (SM90) dropped in 2026.09.30 release; use commit ba89a34 for Hopper.
- Kernels: flash_mla_sparse_fwd (prefill); flash_mla_with_kvcache with indices (decode, FP8/FP4 KV; bf16 KV decode unsupported); fused norm+RoPE+attn+cast; dense MHA prefill/backward (CUDA only).
- Requirements: CUDA 13.1+ (13.2 recommended), PyTorch 2.10+.
- V3.2 DSA kernels removed from current release; V4.0 unsupported; V4.1 supported (added 2026.09.10).
- No AMD ROCm or CPU support in the repo. Vendor forks listed separately (MetaX, Moore Threads, Hygon, Iluvatar, Intellifusion).
Portable/AMD leads:
- AMD GEAK: DeepSeek V4 sparse MLA ported from FlashMLA's PyTorch reference to Triton; passes FlashMLA sparse prefill/decode tests (304 + 4,748 cases); used in SGLang. https://rocm.blogs.amd.com/software-tools-optimization/geak-mla-optimization/README.html [S-to-P: blog]
- vLLM ROCm DeepSeek-V4-Flash sparse MLA Triton PR on a fork, status unverified. vLLM V4 attention modules: vllm/model_executor/layers/deepseek_v4_attention.py (per search snippet). https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v4/attention/
- llama.cpp: a release note mentions deepseek4.cpp; CSA/indexer support NOT confirmed. Check ggml-org/llama.cpp source directly.
- Triton/PyTorch reference: FlashMLA tests contain a PyTorch reference (per GEAK). No standalone official Triton reference from DeepSeek found.

## 4. Retrofitting MLA onto GQA/MHA
- TransMLA (arXiv 2502.07864, NeurIPS 2025): converts GQA to MLA by bridge transform; pretrained weights carried over; minimal fine-tuning. LLaMA-2-7B: 68.75% KV compression with 1.65% drop over 6 benchmarks training-free; 93% compression recovered with ~6B training tokens (paper numbers as reported in search summary; verify tables). Code: https://github.com/fxmeng/TransMLA . https://arxiv.org/html/2502.07864v4
- MHA2MLA (Ji et al., arXiv 2502.14837, ACL 2025): partial RoPE removal + joint SVD low-rank of K/V weights; fine-tune on small data fraction (ACL abstract: 0.6%-1%; v1: 3-6 per mille; sources disagree, check paper). Llama2-7B: KV cache reduced 92.19% with ~1% LongBench drop (v2). Code: https://github.com/JT-Ushio/MHA2MLA
- X-EcoMLA: structured SVD + cross-layer distillation; data fraction NOT FOUND (only secondhand via CARE, ICLR 2026).
- Caveat: these are GQA/MHA->MLA conversions for dense models, not the V4 CSA2 design; V4 is trained natively.

## Open items
- 840 B: NOT FOUND. Ask the user for the source of that number.
- 890 derivation: not in arXiv text read (0-200k of 212k chars). Read the PDF DeepSeek_V41_Tech_Report.pdf (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) with `hf download` or curl; WebFetch could not parse it.
- Config values came through an LLM-summarized WebFetch; download raw config.json and verify before use.
