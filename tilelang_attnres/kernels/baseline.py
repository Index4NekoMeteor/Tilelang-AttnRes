import torch


# from https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py
def apply_attn_res(prefix_sum, block_residual, proj, norm):
    """
    prefix_sum:     (num_tokens, hidden_size)
    block_residual: (num_tokens, num_blocks, hidden_size)
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)
