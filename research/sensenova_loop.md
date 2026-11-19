# SenseNova-Loop-9B research notes

Sources are cited inline. "WebFetch" results are LLM-extracted summaries of pages, not raw bytes. Where a raw file could not be fetched, that is stated.

Access notes:
- Raw `config.json` for the parent repo `sensenova/SenseNova-Loop-9B` returned HTTP 401 (repo is gated: "You must agree to share your Hugging Face information"). Source: https://huggingface.co/sensenova/SenseNova-Loop-9B/raw/main/config.json and /blob/main/config.json
- Raw `config.json` for `sensenova/SenseNova-Loop-9B-GGUF` returned HTTP 404. Source: https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF/raw/main/config.json
- The Preview repo's `config.json` was fetched (raw, summarized by WebFetch) and is the closest config available. It is the **Preview**, not the release. Source: https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/raw/main/config.json
- No `modeling_*.py` is in the GitHub repo root listing. Model code lives in `src/sensenova_loop/`. Listing: `models/` (dir), `__init__.py`, `__main__.py`, `batch.py`, `cli.py`, `compat.py`, `generation.py`, `inference.py`, `loading.py`, `mtp.py`, `profiling.py`, `request.py`, `speculative_cache.py`. Source: https://github.com/OpenSenseNova/SenseNova-Loop/tree/main/src/sensenova_loop (listing is truncated, "View all files"). The `models/` contents were NOT fetched. `modeling_sensenova_loop` is referenced by the Preview config `auto_map` (per WebFetch summary), so the `.py` may live under `models/` or be pulled from the Hub repo. NOT FOUND in fetched material.

## 1. Architecture

| Item | Value | Source |
|---|---|---|
| Distinct decoder layers | 32 | https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF ; https://huggingface.co/sensenova/SenseNova-Loop-9B ; https://github.com/OpenSenseNova/SenseNova-Loop |
| Layer executions per forward | 56 | same as above |
| Loop block | layers [4, 28) = 24 layers (indices 4..27) | same |
| Loop passes | 2, shared weights; flow `B(B(h))` | same |
| Pre-loop / post-loop | layers 0–3 run once; layers 28–31 + head run once | same |
| Loop count | Fixed at 2 by default. Experiments with 1–4 loops. "Dynamic Loop architectures that adapt the number of loops to each input" listed as future work | https://huggingface.co/sensenova/SenseNova-Loop-9B |
| Early exit / adaptive loops | NOT FOUND (not implemented per model card) | https://huggingface.co/sensenova/SenseNova-Loop-9B |
| Per-loop embeddings / per-loop gates | NOT FOUND (not mentioned) | all pages above |
| Loop scaling | LiveCodeBench V6: 2→4 loops = +1.04 pts. AIDABench data viz: 2→4 = +1.35. AIDABench-QA and Workbuddy-office peak at 3 loops | https://huggingface.co/sensenova/SenseNova-Loop-9B |
| Residual structure | Two-stream manifold-constrained hyper-connections (mHC), 2 streams, Sinkhorn normalization, 20 iterations, FP32 routing params. Applied over layers 4–28 | https://github.com/OpenSenseNova/SenseNova-Loop ; Preview config (WebFetch summary) |
| Hidden size | 4,096 | https://github.com/OpenSenseNova/SenseNova-Loop ; Preview config |
| Intermediate (FFN) size | 12,288 | same |
| Vocab size | 248,320 | same |
| Attention heads (full attn) | 16 | same |
| KV heads | 4 (GQA, 4:1) | same |
| head_dim | 256 (from Preview config only; not confirmed on release) | Preview config (WebFetch summary) |
| Attention type | Hybrid: "three linear-attention layers followed by one full-attention layer" in each group of four. Full-attention layers = 8 of 32 | https://huggingface.co/sensenova/SenseNova-Loop-9B ; Preview config says "full_attention layer every 4th layer" |
| Exact indices of full-attn layers | NOT FOUND (assumed 3,7,...,31 below; unverified) | - |
| Linear-attention type (name, state size) | NOT FOUND | - |
| RoPE | partial rotary factor 0.25, rope theta 10,000,000 (Preview config) | Preview config (WebFetch summary) |
| Max context | 512K tokens (model card); Preview config max positions 262,144 | https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF ; Preview config |
| MoE or dense | Dense (no expert count reported). MoE: NOT FOUND | - |
| Precision | BF16 backbone; FP32 mHC routing | https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF |
| Parameters | ~9B stored; ~14.6B "inference-equivalent" (because the loop reuses weights) | https://huggingface.co/sensenova/SenseNova-Loop-9B |
| Vision encoder | 27 layers, hidden 1152, 16 heads, patch 16, merge 2 (Preview config) | Preview config (WebFetch summary) |
| Architecture string | `sensenova_loop` (config `model_type`; GGUF arch string per model card) | https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF ; Preview config |
| Model class | `SenseNovaLoopForConditionalGeneration` | Preview config (WebFetch summary) |

## 2. KV cache

Formula: `bytes/token = 2 (K,V) * N_kv_layers * kv_heads * head_dim * bytes_per_elem`

Inputs: kv_heads = 4, head_dim = 256 (Preview config), bytes = 2 for bf16 and fp16.

Full-attention layer positions are not confirmed. Assumption used below: every 4th layer, indices 3, 7, 11, 15, 19, 23, 27, 31 (8 layers). Of these, 7, 11, 15, 19, 23, 27 fall inside the loop [4, 28), and 3 and 31 are outside it. Linear-attention layers keep a recurrent state, not per-token KV, so they are excluded. Their state size is NOT FOUND.

**Case A: one KV entry per physical layer (loop passes share a cache).** Physical full-attn layers = 8.
- bf16: 2 * 8 * 4 * 256 * 2 = 32,768 B = 32 KiB per token
- fp16: 2 * 8 * 4 * 256 * 2 = 32,768 B = 32 KiB per token
- At 512K tokens: 524,288 * 32,768 = 16 GiB

**Case B: each loop pass keeps its own KV (no sharing).** Effective full-attn executions = 2 (layers 3, 31) + 6 * 2 (loop layers run twice) = 14.
- bf16: 2 * 14 * 4 * 256 * 2 = 57,344 B = 56 KiB per token
- fp16: 2 * 14 * 4 * 256 * 2 = 57,344 B = 56 KiB per token
- At 512K tokens: 524,288 * 57,344 = 28 GiB

Whether the implementation uses Case A or B is NOT FOUND. The model card does not say. Check `src/sensenova_loop/` (cache/inference code) to decide. Non-loop comparison: a plain 32-layer model with 8 full-attn layers would be the same as Case A.

## 3. GGUF files

Source: https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF (WebFetch summary)

| Quant | Size |
|---|---|
| BF16 | 18.4 GB |
| Q8_0 | 9.8 GB |
| Q5_K_M | 6.67 GB |
| Q4_K_M | 5.82 GB |

- The model card also says 4-bit weights use "approximately 4.9 GB" (likely the AWQ or the working footprint; the source is ambiguous).
- The Hub page says there are 22 quantized variants across llama.cpp, Ollama, and LM Studio (source: https://huggingface.co/sensenova/SenseNova-Loop-9B). Only the four above were named.
- Mainstream llama.cpp support: **No.** The model card says it "requires the OpenSenseNova/llama.cpp build" (source: https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF). Upstream llama.cpp support: NOT FOUND in fetched material.
- GGUF metadata arch string: `sensenova_loop` (per the page; I did not inspect the binary).
- Example: `llama serve -hf sensenova/SenseNova-Loop-9B-GGUF:Q4_K_M -ngl 99 -c 4096 --port 8000` (source: GGUF model card). Also: Ollama `hf.co/sensenova/SenseNova-Loop-9B-GGUF:Q4_K_M`, Docker Model Runner.
- Note: the `llama serve` command as printed is not standard llama.cpp syntax (llama.cpp uses `llama-server`). Flag as unverified.

## 4. Benchmarks and license

Benchmarks (vendor-reported, no independent evaluation found):
- Versus Qwen3.8-27B (as named on the page): NL2Repo 41.36 vs 44.60; AIDABench 65.31 vs 65.01; AutomationBench 36.67 (matched). Source: https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF
- Preview to release: AutomationBench 14.33 → 36.67; Terminal-Bench 2.1 40.45 → 59.55; GAIA 37.86 → 69.90; BIRD 71.38 → 72.34. Source: https://huggingface.co/sensenova/SenseNova-Loop-9B
- Terminal-Bench 2.1 for Preview 40.45 vs 29.20 for Qwen3.5-9B. Source: https://www.mindstudio.ai/blog/sensenova-loop-9b-benchmarks (secondary, from a search snippet).
- Known limitations (vendor-listed): weak on complex instruction following with many tool definitions, inconsistent on long-horizon coding and agent tasks, 3D/animation-heavy websites need 2–3 refinement rounds. Source: https://huggingface.co/sensenova/SenseNova-Loop-9B

License: **Apache-2.0** for the model card, the GitHub repo, and the Preview. Sources: https://huggingface.co/sensenova/SenseNova-Loop-9B ; https://github.com/OpenSenseNova/SenseNova-Loop

Gating: the parent repo is gated (agree to share contact info). Source: https://huggingface.co/sensenova/SenseNova-Loop-9B

## 5. Inference code and related work

- Transformers: `trust_remote_code=True`, `transformers==5.18.0`, PyTorch 2.11.0, CUDA 13.0 (the Preview env). `AutoModelForCausalLM`, `dtype=torch.bfloat16`. Source: https://huggingface.co/sensenova/SenseNova-Loop-9B
- SGLang: PR #47281 required (not mainstream). https://github.com/sgl-project/sglang/pull/47281
- vLLM: PR #64832 required (not mainstream). https://github.com/vllm-project/vllm/pull/64832 ; `--served-model-name sensenova_loop`
- LightLLM: `ghcr.io/modeltc/lightllm:main`, `--tp 4`. Source: https://huggingface.co/sensenova/SenseNova-Loop-9B
- llama.cpp: fork `OpenSenseNova/llama.cpp` required. Source: https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF
- MTP speculative decoding supported; draft training in https://github.com/OpenSenseNova/SenseSpec
- Fine-tuning: https://github.com/OpenSenseNova/torchtitan
- GitHub: https://github.com/OpenSenseNova/SenseNova-Loop (Apache-2.0, 209 stars, 11 forks, 46 commits on main per WebFetch)

Paper / tech report: **NOT FOUND.** The model card, GitHub README, and search results link none. The nearest research basis is the Ouro looped-LM paper, which I did not verify as the basis for SenseNova:
- arXiv:2510.25741, "Scaling Latent Reasoning via Looped Language Models" (Zhu et al., ByteDance / UC Santa Cruz). https://arxiv.org/abs/2510.25741 . Project page http://ouro-llm.github.io . A German article links SenseNova to this work (secondary; https://www.drweb.de/sensenova-loop-9b-sensetime-ki-modell/ , not fetched).

Coverage gaps: raw parent `config.json` (401), GGUF `config.json` (404), the release (not Preview) config, `src/sensenova_loop/models/` source, and the llama.cpp fork diff were not retrieved.
