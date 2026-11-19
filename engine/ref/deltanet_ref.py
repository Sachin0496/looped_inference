"""Reference for one Gated DeltaNet decode step (one new token per sequence, T sequences).

Code is copied from the transformers Qwen3.5 implementation at
  /opt/homebrew/lib/python3.14/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py
with the kernel-hub decorators removed so the pure-PyTorch fallback runs (the `fla` and
`causal_conv1d` packages are not installed on Kaggle). Source line numbers are given per function.
Pure torch: runs on CPU and CUDA.

Decode step (Qwen3_5GatedDeltaNet.forward, seq_len == 1, cache present), inputs are the in_proj outputs:
  qkv bf16 [T, 8192]  (rows [0:2048]=q, [2048:4096]=k, [4096:8192]=v; the in_proj_qkv output)
  z   bf16 [T, 4096]  (in_proj_z output, viewed as 32 heads x 128)
  b   bf16 [T, 32]    (in_proj_b output)
  a   bf16 [T, 32]    (in_proj_a output)
  conv_state bf16 [T, 8192, 4]   updated in place (causal_conv1d_update)
  rec_state  fp32 [T, 32, 128, 128] updated in place (update_recurrent_state)
  A_log fp32 [32], dt_bias fp32 [32], conv_w bf16 [8192, 4] (conv1d.weight.squeeze(1)), norm_w bf16 [128]
Output: bf16 [T, 4096] (input of out_proj).
"""
import torch
import torch.nn.functional as F

# Config values of SenseNova-Loop-9B text_config (see research/qwen35_decode_spec.md section 1)
NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4
KEY_DIM = HEAD_K_DIM * NUM_K_HEADS      # 2048
VALUE_DIM = HEAD_V_DIM * NUM_V_HEADS    # 4096
CONV_DIM = 2 * KEY_DIM + VALUE_DIM      # 8192
RMS_EPS = 1e-6


# Qwen3_5GatedDeltaNet.forward, lines 617-619 (beta, g), decode branch.
def compute_gates(a, b, A_log, dt_bias):
    beta = b.sigmoid()                                                   # L617, bf16 result
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)           # L619, fp32
    return beta, g


# modeling_qwen3_5.py lines 249-266 (causal_conv1d_update, fallback body, decorator removed)
def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


# modeling_qwen3_5.py lines 294-297
def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


# modeling_qwen3_5.py lines 437-498 (torch_recurrent_gated_delta_rule, decorator and **kwargs removed)
def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None,
                                     output_final_state=False, use_qk_l2norm_in_kernel=False):
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query = query / (query.shape[-1] ** 0.5)

    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# modeling_qwen3_5.py lines 225-234 (Qwen3_5RMSNormGated.forward; weight is NOT zero-centred)
def rmsnorm_gated(hidden_states, gate, weight, eps=RMS_EPS):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = weight * hidden_states.to(input_dtype)
    hidden_states = hidden_states * F.silu(gate.to(torch.float32))
    return hidden_states.to(input_dtype)


# Qwen3_5GatedDeltaNet.forward, decode branch (lines 550-665), starting from the in_proj outputs
# and ending before out_proj.
def deltanet_decode_step(qkv, z, b, a, conv_state, rec_state, A_log, dt_bias, conv_w, norm_w):
    T = qkv.shape[0]
    mixed_qkv = qkv.reshape(T, CONV_DIM, 1)                             # transpose(1,2) of [T,1,8192]
    mixed_qkv = causal_conv1d_update(mixed_qkv, conv_state, conv_w, None, "silu")
    mixed_qkv = mixed_qkv.transpose(1, 2)                               # [T, 1, 8192]
    query, key, value = torch.split(mixed_qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    query = query.reshape(T, 1, -1, HEAD_K_DIM)
    key = key.reshape(T, 1, -1, HEAD_K_DIM)
    value = value.reshape(T, 1, -1, HEAD_V_DIM)

    beta, g = compute_gates(a.reshape(T, 1, NUM_V_HEADS), b.reshape(T, 1, NUM_V_HEADS), A_log, dt_bias)
    if NUM_V_HEADS // NUM_K_HEADS > 1:                                  # L621-622, GQA repeat
        query = query.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)
        key = key.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2)

    core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=rec_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    rec_state.copy_(last_recurrent_state)                               # update_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, HEAD_V_DIM)
    zz = z.reshape(-1, HEAD_V_DIM)
    core_attn_out = rmsnorm_gated(core_attn_out, zz, norm_w)
    return core_attn_out.reshape(T, VALUE_DIM)


def make_weights(seed=0, device="cpu"):
    """Per-layer constants with realistic scales (conv taps, A_log = log U(0.01,16), dt_bias, norm weight)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    conv_w = (torch.randn(CONV_DIM, CONV_KERNEL, generator=g) * 0.3).to(torch.bfloat16)
    A = torch.empty(NUM_V_HEADS).uniform_(0.01, 16, generator=g)
    A_log = torch.log(A)
    dt_bias = torch.empty(NUM_V_HEADS).uniform_(-2.0, 0.5, generator=g)
    norm_w = (1.0 + 0.2 * torch.randn(HEAD_V_DIM, generator=g)).to(torch.bfloat16)
    return {k: v.to(device) for k, v in dict(conv_w=conv_w, A_log=A_log, dt_bias=dt_bias, norm_w=norm_w).items()}


def make_step_inputs(T, gen, device="cpu", scale_qkv=1.0):
    """One decode step of random in_proj outputs. Scales: qkv ~ N(0,1), z ~ N(0,1), b ~ N(0,2), a ~ N(0,1)."""
    qkv = (torch.randn(T, CONV_DIM, generator=gen) * scale_qkv).to(torch.bfloat16)
    z = torch.randn(T, VALUE_DIM, generator=gen).to(torch.bfloat16)
    b = (torch.randn(T, NUM_V_HEADS, generator=gen) * 2.0).to(torch.bfloat16)
    a = torch.randn(T, NUM_V_HEADS, generator=gen).to(torch.bfloat16)
    return {k: v.to(device) for k, v in dict(qkv=qkv, z=z, b=b, a=a).items()}


def make_state(T, gen, device="cpu"):
    """Initial conv state (bf16, unit-scale history) and recurrent state (fp32, small)."""
    conv_state = torch.randn(T, CONV_DIM, CONV_KERNEL, generator=gen).to(torch.bfloat16)
    rec_state = (torch.randn(T, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, generator=gen) * 0.05).float()
    return conv_state.to(device), rec_state.to(device)


if __name__ == "__main__":
    # CPU smoke run: a few steps, checks shapes, finiteness and that the state moves.
    gen = torch.Generator(device="cpu").manual_seed(0)
    w = make_weights(0)
    conv_state, rec_state = make_state(2, gen)
    before = rec_state.clone()
    for _ in range(4):
        inp = make_step_inputs(2, gen)
        out = deltanet_decode_step(inp["qkv"], inp["z"], inp["b"], inp["a"], conv_state, rec_state,
                                   w["A_log"], w["dt_bias"], w["conv_w"], w["norm_w"])
    assert out.shape == (2, VALUE_DIM) and out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    assert rec_state.dtype == torch.float32 and not torch.equal(before, rec_state)
    print("cpu smoke ok: out", tuple(out.shape), "max|out|", float(out.float().abs().max()),
          "max|S|", float(rec_state.abs().max()))
