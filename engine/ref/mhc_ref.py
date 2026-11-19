"""Reference mHC step, copied in meaning from SenseNova-Loop's modeling_sensenova_loop.py.

One "step" is what the fused kernel replaces:
    residual = merge(residual, branch, post_prev, mix_prev)   # optional, skipped at the first mHC layer
    branch, post, mix = router(residual)                      # MHCRouter.forward
    branch = rmsnorm(branch)                                  # Qwen3_5RMSNorm, (1 + w) convention
Shapes: residual [T, S, D] bf16, branch [T, D] bf16, router weight [2S + S*S, S*D] fp32.
"""
import torch


def sinkhorn(logits, iterations):
    logits = logits.float()
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()


def router(residual, weight, bias, iterations=20, pre_scale=1.0):
    S = residual.shape[-2]
    with torch.autocast(device_type=residual.device.type, enabled=False):
        flat = residual.flatten(-2).float()
        normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        logits = torch.nn.functional.linear(normalized, weight, bias)
        pre = logits[..., :S].sigmoid()
        post = 2 * logits[..., S:2 * S].sigmoid()
        mix = sinkhorn(logits[..., 2 * S:].unflatten(-1, (S, S)), iterations)
        branch = torch.matmul(pre.to(residual.dtype).unsqueeze(-2), residual).squeeze(-2)
        if pre_scale != 1.0:
            branch = branch * pre_scale
    return branch, post, mix


def merge(residual, branch, post, mix):
    with torch.autocast(device_type=residual.device.type, enabled=False):
        mixed_residual = torch.matmul(mix.to(residual.dtype), residual)
        branch_update = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
        return mixed_residual + branch_update


def rmsnorm(x, weight, eps=1e-6):
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.type_as(x)


def mhc_step(residual, weight, bias, norm_w, branch_prev=None, post_prev=None, mix_prev=None,
             iterations=20, pre_scale=1.0):
    """Returns (new_residual, normed_branch, post, mix)."""
    if branch_prev is not None:
        residual = merge(residual, branch_prev, post_prev, mix_prev)
    branch, post, mix = router(residual, weight, bias, iterations, pre_scale)
    return residual, rmsnorm(branch, norm_w), post, mix


def make_inputs(T, S=2, D=4096, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = 2 * S + S * S
    residual = torch.randn(T, S, D, generator=g).to(torch.bfloat16)
    branch = torch.randn(T, D, generator=g).to(torch.bfloat16)
    weight = torch.randn(out, S * D, generator=g) * 0.02
    bias = torch.randn(out, generator=g)
    norm_w = torch.randn(D, generator=g) * 0.1
    post = 2 * torch.rand(T, S, generator=g)
    mix = sinkhorn(torch.randn(T, S, S, generator=g), 20)
    return {k: v.to(device) for k, v in dict(residual=residual, branch_prev=branch, weight=weight,
                                             bias=bias, norm_w=norm_w, post_prev=post,
                                             mix_prev=mix).items()}
