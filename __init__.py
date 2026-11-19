import torch
from torch import nn
from torch.autograd import Function
from kernels import block_residual_fwd, block_residual_bwd


class AttnResFunction(Function):
    @staticmethod
    def forward(
        ctx,
        prefix_sum,
        block_residual,
        proj_weight,
        norm_weight,
        eps: float = 1e-6,
    ):
        output, rsqrt_output, max_output, exp_output, div_output, softmax_output = block_residual_fwd(
            prefix_sum, block_residual, proj_weight, norm_weight, eps=eps
        )
        ctx.save_for_backward(
            prefix_sum,
            block_residual,
            proj_weight,
            norm_weight,
            rsqrt_output,
            max_output,
            exp_output,
            div_output,
            softmax_output,
        )
        return output

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        (
            block_residual, 
            prefix_sum, 
            proj_weight, 
            norm_weight, 
            rsqrt_output, 
            max_output, 
            exp_output, 
            div_output, 
            softmax_output
        ) = ctx.saved_tensors

        dprefix_sum, dblock_residual, dproj_weight, dnorm_weight = block_residual_bwd(
            grad,
            block_residual,
            prefix_sum,
            proj_weight,
            norm_weight,
            rsqrt_output,
            max_output,
            exp_output,
            div_output,
            softmax_output,
        )
        return dprefix_sum, dblock_residual, dproj_weight, dnorm_weight, None


def attn_res_tilelang(prefix_sum, block_residual, proj_weight, norm_weight, eps=1e-6):
    return AttnResFunction.apply(prefix_sum, block_residual, proj_weight, norm_weight, eps=eps)


class AttnRes(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.proj_weight = nn.Parameter(torch.ones(hidden_size))
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, prefix_sum, block_residual):
        return AttnResFunction.apply(prefix_sum, block_residual, self.proj_weight, self.norm_weight, self.eps)