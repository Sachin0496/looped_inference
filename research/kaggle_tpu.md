# Kaggle TPU script kernels: research notes

Scope: read-only research. Nothing was pushed or created on Kaggle.
Confidence labels: **[src]** = verified in installed CLI/SDK source; **[web]** = third-party or search-snippet source, not confirmed; **[unknown]** = not found.

## 1. Requesting a TPU in kernel-metadata.json

**[src] CLI version:** kaggle 2.2.4, `/opt/homebrew/lib/python3.14/site-packages/kaggle`.

- `kaggle/api/kaggle_api_extended.py:6290-6298`: `kernels_initialize` template fields: `enable_gpu` "false", `enable_tpu` "false", `enable_internet` "true", `machine_shape` "", `kernel_sources` [], `model_sources` [].
- `kaggle/api/kaggle_api_extended.py:6437-6439`: `kernels_push` maps `enable_gpu`, `enable_tpu` (bool, default False) and `enable_internet` (bool, **default True**) into the save request.
- `kaggle/api/kaggle_api_extended.py:6450`: `request.machine_shape = acc if acc else meta["machine_shape"]`. The CLI does not validate the string; it is passed through to the server.
- `kaggle/cli.py:1315`: `kaggle kernels push --accelerator ACC` overrides the metadata value. Help text (`cli.py:2833`): "Specify the type of accelerator to use for the kernel run".
- `kagglesdk/kernels/types/kernels_api_service.py:191-195` (ApiSaveKernelRequest docstring): `machine_shape` options "currently supported": `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`. The same list appears at lines 281-283 and 2030-2032 and 2395-2397.

**Conclusion [src]:** A TPU is requested by setting `"enable_tpu": true` plus a `machine_shape`. The only TPU value in the installed SDK is `Tpu1VmV38`, which is the TPU v3-8 shape. No v5e or v6e value is listed in the installed source.

**[web] Conflicting value:** A Kaggle docs page for kernels metadata (github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md, fetched via a summarizer) lists `"TpuV5E8"` (TPU v5e-8) as a machine_shape example, along with `NvidiaTeslaT4` and `NvidiaL4`. That page was fetched through a summarizer, so treat the exact strings as unconfirmed. A HuggingFace deploy script (huggingface.co/spaces/cuilabs/bee, scripts/push_kaggle_tpu_kernel.py) reportedly uses `"TpuV6E8"`. The fetch returned HTTP 401, so this is from search snippets only. Snippets also said a JAX AI Stack PR pointed users to v5e-8 as the Kaggle TPU.

**[unknown] Recommended test:** Run `kaggle kernels push` on a throwaway kernel with `enable_tpu: true` and each candidate `machine_shape` (`TpuV5E8`, `Tpu1VmV38`, `TpuV6E8`). Then check the server response or the kernel's `machine_shape` via `kaggle kernels status` / `kaggle kernels list`. The API may reject unknown strings. Do this only with user approval, since it creates a kernel.

`kaggle kernels push --help` shows only `-p`, `-t/--timeout`, and `--accelerator ACC`. It lists no accepted values.

## 2. TPU VM environment (v5e-8)

| Fact | Status | Source |
|---|---|---|
| Weekly TPU quota ~20 h | [web] most 2026 aggregators agree; Kaggle page not reached | search snippets (gpuperhour.com, agentdeals.dev, resourify.com) |
| Max session runtime: 9 h for TPU, 12 h for CPU/GPU | [web] one source claims it read Kaggle's docs; not reached directly | search snippet (apispine.com / aggregator) |
| TPU quota is queryable | [src] `kaggle quota` prints GPU and TPU rows (`kaggle_api_extended.py:6169-6215`); SDK fields `time_used`, `total_time_allowed`, `quota_refresh_time` (`kagglesdk/kernels/types/kernels_api_service.py:6-30, 521-527`) | installed source |
| v5e-8 = 8 chips, single host | [web] third-party TPU tutorial (v5litepod-8 = 8 cores, single host); Google docs cover the v5e type | search snippets |
| HBM per chip | [unknown] Google's v5e spec is 16 GB HBM per chip (about 128 GB total for 8 chips). Not verified against Kaggle | general knowledge, not fetched |
| Host RAM, /kaggle/working and /tmp disk | [unknown] not found | none |
| Preinstalled JAX version | [unknown] not found; no Kaggle TPU docs text was reachable | none |
| torch / torch_xla preinstalled | [unknown] one search snippet mentions a PyTorch/XLA 2.0 on Kaggle notebooks, not verified | Google docs snippet |
| Kaggle docs page (kaggle.com/docs/tpu) | [unknown] fetch returned only the page title, no body | fetch |
| kaggle-docker repo TPU dir | [web] `tpu/` directory exists in github.com/Kaggle/docker-python; contents not readable through the fetch; pins not verified | github fetch |

Check on a live session before relying on any of these: `cat /etc/os-release`, `pip list | grep -iE "jax|libtpu|torch"`, `df -h /kaggle/working /tmp`, `free -g`, `python -c "import jax; print(jax.devices())"`.

## 3. Internet access on script kernels

- **[src]** The CLI default for `enable_internet` is True (`kaggle_api_extended.py:6439`). The web summary of the metadata docs said the default is false; the code wins for CLI pushes.
- **[unknown]** Whether internet access actually works on TPU sessions specifically. Kaggle has historically restricted internet on some accelerator sessions. Test with `pip download` or `curl` to a HuggingFace URL inside a short TPU kernel, and check the kernel's internet toggle.

## 4. kernel_sources (attaching another kernel's output)

- **[src]** `kernel_sources` is a list of `"username/kernel-slug"` strings (`kaggle_api_extended.py:6399-6401`, validated by `validate_kernel_string` at line 8955, which requires 2 or 3 slash-separated parts). It is sent as `kernel_data_sources` (line 6442).
- **[src]** On pull, `kernel_sources` is read back from `server_metadata.kernel_data_sources` (line 6619).
- **[unknown] Mount path.** The CLI does not set it. A search snippet suggests competition sources mount at `/kaggle/input/competitions/<slug>/`, not `/kaggle/input/<slug>/`. For kernel outputs, the path is not confirmed. Verify at runtime with `ls /kaggle/input/` and `find /kaggle/input -maxdepth 3`.
- **[unknown] TPU compatibility.** Nothing found. The attach is a server-side field, and nothing in the CLI blocks it for TPU kernels. Test it.

## 5. JAX on Kaggle TPU v5e: gotchas

- **[web] Install command (Google docs):** `pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`. Google's v5e docs describe this for Cloud TPU VMs. Not verified on Kaggle.
- **[web] Version floor:** JAX 0.4.6+ supports TPU v5e (Vertex AI docs snippet).
- **[web] libtpu mismatch:** Google's runtime table pairs each framework version with a libtpu version. Installing `jax[tpu]` alongside a preinstalled jax/libtpu can conflict. Check `jax.__version__` and `libtpu` before and after any install.
- **[web, general] "TPU in use" / device busy:** Usually another process still holds the TPU (a previous cell, a background process, or a second Python in the same session). Restart the kernel or kill stale processes before retrying.
- **[unknown] jax.devices() count on Kaggle v5e-8.** Expect 8 devices if the host exposes all chips. Confirm on a live session.
- **[general] Use one process per host.** Multiple JAX processes cannot share the TPU on the same host.

## Next steps (suggested, not done)

1. Test `machine_shape` values on a throwaway kernel (needs user approval, since it creates a kernel).
2. Run a probe script that records `pip list`, `df -h`, `free -g`, `jax.devices()`, and `/kaggle/input` contents, then use `kaggle kernels output` to retrieve it.
3. Confirm quota and session limits from the live account (`kaggle quota`).

## Sources

- Installed CLI/SDK (line refs above): `/opt/homebrew/lib/python3.14/site-packages/kaggle/api/kaggle_api_extended.py`, `/opt/homebrew/lib/python3.14/site-packages/kaggle/cli.py`, `/opt/homebrew/lib/python3.14/site-packages/kagglesdk/kernels/types/kernels_api_service.py`
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md (summarized, not verbatim)
- https://www.kaggle.com/docs/tpu (title only, no body)
- https://huggingface.co/spaces/cuilabs/bee/blob/main/scripts/push_kaggle_tpu_kernel.py (401, snippet only)
- https://docs.cloud.google.com/tpu/docs/v5e-training (install command, via search snippet)
- https://gpuperhour.com/blog/free-cloud-gpus-and-credits, https://agentdeals.dev/vendor/kaggle, https://apispine.com/kaggle/pricing (quota and session snippets, third-party)
