import os
import sys
import torch
import tilelang
from itertools import product
from tilelang import language as T
from tilelang.autotuner.capture import set_autotune_inputs
from tilelang.layout.swizzle import make_linear_layout, make_swizzled_layout


# NOTE: disable autotuner
# def block_residual_bwd_configs(**kwargs):
#     dtype = kwargs.get("dtype")
#     keys = ("threads", "threads_2", "block_size", "block_size_2")

#     if dtype != "float32":
#         values = product(
#             [256, 512],
#             [32, 64, 128],
#             [256, 512, 1024],
#             [64, 128],
#         )
#     else:
#         values = product(
#             [256, 512],
#             [32, 64, 128],
#             [256, 512, 1024],
#             [64],
#         )
    
#     return [dict(zip(keys, value)) for value in values]


# @tilelang.autotune(
#     configs=block_residual_bwd_configs,
#     do_not_specialize=["threads", "threads_2", "block_size", "block_size_2"],
#     warmup=3,
#     rep=5,
# )
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def block_residual_bwd_kernel(
    num_blocks,
    hidden_size,
    dtype,
    weight_dtype,
    threads=128,
    threads_2=64,
    token_size=32,
    block_size=256,
    block_size_2=64,
    num_stages: int = 1,
    accum_dtype=T.float32,
):
    num_token = T.dynamic("num_token")
    total_block = num_blocks + 1

    @T.prim_func
    def bwd_kernel(
        grad: T.Tensor((num_token, hidden_size), dtype),
        prefix_sum: T.Tensor((num_token, hidden_size), dtype),
        block_residual: T.Tensor((num_token, num_blocks, hidden_size), dtype),
        norm_weight: T.Tensor((hidden_size,), weight_dtype),
        proj_weight: T.Tensor((hidden_size,), weight_dtype),
        # fwd cache values
        rsqrt_output: T.Tensor((num_token, total_block), accum_dtype),
        max_output: T.Tensor((num_token, 1), T.int32),
        exp_output: T.Tensor((num_token, total_block), accum_dtype),
        div_output: T.Tensor((num_token, 1), accum_dtype),
        softmax_output: T.Tensor((num_token, total_block), accum_dtype),
        # output value
        doutput: T.Tensor((num_token, total_block, hidden_size), dtype),
        dproj_weight: T.Tensor((hidden_size), accum_dtype),
        dnorm_weight: T.Tensor((hidden_size), accum_dtype),
        add_workspace: T.Tensor((num_token, total_block), accum_dtype),
    ):
        with T.Kernel(num_token, threads=threads) as token:
            T.annotate_min_blocks_per_sm(2)
            T.assume(num_token > 0)  # ty: ignore
            T.assume(num_blocks > 0)
            T.assume(total_block > 0)
            T.assume(hidden_size > 0)

            T.pdl_sync()
            # ======= stage 1 =======
            idx_local = T.alloc_fragment((1), T.int32)
            divisor_local = T.alloc_fragment((1), accum_dtype)
            grad_share_ptx = T.alloc_shared((block_size), dtype)
            norm_share_ptx = T.alloc_shared((block_size), accum_dtype)
            proj_share_ptx = T.alloc_shared((block_size), accum_dtype)
            hidden_share_ptx = T.alloc_shared((total_block, block_size), dtype)

            weight_local_ptx = T.alloc_fragment((block_size), accum_dtype)
            exp_output_local_ptx = T.alloc_fragment((total_block), accum_dtype)
            rsqrt_output_local_ptx = T.alloc_fragment((total_block), accum_dtype)
            softmax_output_local_ptx = T.alloc_fragment((total_block), accum_dtype)
            add_workspace_local_ptx = T.alloc_fragment((total_block), accum_dtype)

            div_3 = T.alloc_fragment((total_block), accum_dtype)
            mul_12 = T.alloc_fragment((total_block), accum_dtype)
            bmm_2 = T.alloc_fragment((total_block), accum_dtype)
            bmm_weight = T.alloc_fragment((total_block), accum_dtype)

            mul_3_sum = T.alloc_fragment((1), accum_dtype)
            div_3_sum = T.alloc_fragment((1), accum_dtype)
            tile_sum_1 = T.alloc_fragment((total_block, block_size), accum_dtype)
            tile_sum_2 = T.alloc_fragment((total_block, block_size), accum_dtype)

            T.copy(max_output[token, :], idx_local)
            T.copy(div_output[token, :], divisor_local)
            T.copy(exp_output[token, :], exp_output_local_ptx)
            T.copy(rsqrt_output[token, :], rsqrt_output_local_ptx)
            T.copy(softmax_output[token, :], softmax_output_local_ptx)

            T.clear(mul_3_sum)
            T.clear(tile_sum_1)
            T.clear(tile_sum_2)

            inv_1 = T.float32(1.0) / divisor_local[0]  # ty: ignore
            inv_2 = T.float32(1.0) / hidden_size  # ty: ignore

            for block in T.Pipelined(
                T.ceildiv(hidden_size, block_size),
                num_stages=num_stages,
            ):
                ptx_start = block * block_size
                ptx_end = ptx_start + block_size

                T.copy(grad[token, ptx_start:ptx_end], grad_share_ptx)
                T.copy(norm_weight[ptx_start:ptx_end], norm_share_ptx)
                T.copy(proj_weight[ptx_start:ptx_end], proj_share_ptx)
                T.copy(block_residual[token, :, ptx_start:ptx_end], hidden_share_ptx[0:num_blocks, :])
                T.copy(prefix_sum[token, ptx_start:ptx_end], hidden_share_ptx[num_blocks:total_block, :])

                for j in T.Parallel(block_size):
                    weight_local_ptx[j] = norm_share_ptx[j] * proj_share_ptx[j]

                for i, j in T.Parallel(total_block, block_size):
                    grad_fp32 = T.cast(grad_share_ptx[j], accum_dtype)
                    hidden_fp32 = T.cast(hidden_share_ptx[i, j], accum_dtype)
                    tile_sum_1[i, j] += hidden_fp32 * grad_fp32
                    tile_sum_2[i, j] += hidden_fp32 * weight_local_ptx[j]

            T.reduce_sum(tile_sum_1, bmm_2, batch=total_block)
            T.reduce_sum(tile_sum_2, bmm_weight, batch=total_block)

            for i in T.unroll(total_block):
                div_3[i] = bmm_2[i] * inv_1

            for i in T.unroll(total_block):
                mul_3_sum[0] += -div_3[i] * softmax_output_local_ptx[i]

            for i in T.unroll(total_block):
                div_3[i] = (div_3[i] + mul_3_sum[0]) * exp_output_local_ptx[i]

            T.reduce_sum(div_3, div_3_sum, dim=-1)

            for i in T.Parallel(total_block):
                div_3[i] = div_3[i] - T.if_then_else(i == idx_local[0], div_3_sum[0], T.float32(0.0))  # ty:ignore
                add_workspace_local_ptx[i] = div_3[i] * rsqrt_output_local_ptx[i]

            T.copy(add_workspace_local_ptx, add_workspace[token, :total_block])

            for i in T.unroll(total_block):
                pow_2 = rsqrt_output_local_ptx[i] * rsqrt_output_local_ptx[i] * rsqrt_output_local_ptx[i]
                mul_12[i] = (div_3[i] * bmm_weight[i] * -0.5) * pow_2 * inv_2

            # ======= stage 2 =======
            for block in T.Pipelined(
                T.ceildiv(hidden_size, block_size),
                num_stages=num_stages,
            ):
                ptx_start = block * block_size
                ptx_end = ptx_start + block_size

                T.copy(grad[token, ptx_start:ptx_end], grad_share_ptx)
                T.copy(norm_weight[ptx_start:ptx_end], norm_share_ptx)
                T.copy(proj_weight[ptx_start:ptx_end], proj_share_ptx)
                T.copy(block_residual[token, :, ptx_start:ptx_end], hidden_share_ptx[0:num_blocks, :])
                T.copy(prefix_sum[token, ptx_start:ptx_end], hidden_share_ptx[num_blocks:total_block, :])

                for j in T.Parallel(block_size):
                    weight_local_ptx[j] = norm_share_ptx[j] * proj_share_ptx[j]

                for j in T.Parallel(block_size):
                    for i in T.unroll(total_block):
                        grad_fp32 = T.cast(grad_share_ptx[j], accum_dtype)
                        hidden_fp32 = T.cast(hidden_share_ptx[i, j], accum_dtype)

                        bmm_1 = grad_fp32 * softmax_output_local_ptx[i]
                        mul_6 = div_3[i] * weight_local_ptx[j] * rsqrt_output_local_ptx[i]

                        doutput[token, i, ptx_start + j] = T.cast(
                            mul_6 + bmm_1 + mul_12[i] * (hidden_fp32 * 2.0), weight_dtype
                        )

        with T.Kernel(T.ceildiv(hidden_size, block_size_2), threads=threads_2) as hidden_block:
            ptx_start = hidden_block * block_size_2
            ptx_end = ptx_start + block_size_2

            mul_5 = T.alloc_fragment((token_size, block_size_2), accum_dtype)
            mul_5_sum = T.alloc_fragment((block_size_2), accum_dtype)
            add_local_ptx = T.alloc_shared((token_size, total_block), accum_dtype)

            proj_weight_ptx = T.alloc_fragment((block_size_2), weight_dtype)
            norm_weight_ptx = T.alloc_fragment((block_size_2), weight_dtype)
            hidden_share_ptx = T.alloc_shared((token_size, total_block, block_size_2), dtype)

            dproj_weight_local_ptx = T.alloc_fragment((block_size_2), weight_dtype)
            dnorm_weight_local_ptx = T.alloc_fragment((block_size_2), weight_dtype)

            T.copy(proj_weight[ptx_start:ptx_end], proj_weight_ptx)
            T.copy(norm_weight[ptx_start:ptx_end], norm_weight_ptx)

            T.clear(mul_5_sum)

            T.pdl_sync()
            for token in T.Pipelined(T.ceildiv(num_token, token_size), num_stages=num_stages):
                token_ptx_start = token * token_size
                token_ptx_end = token_ptx_start + token_size

                T.copy(add_workspace[token_ptx_start:token_ptx_end, :], add_local_ptx)
                T.copy(
                    block_residual[token_ptx_start:token_ptx_end, :, ptx_start:ptx_end],
                    hidden_share_ptx[:, 0:num_blocks, :],
                )
                T.copy(
                    prefix_sum[token_ptx_start:token_ptx_end, ptx_start:ptx_end],
                    hidden_share_ptx[:, num_blocks:total_block, :],
                )

                for t, j in T.Parallel(token_size, block_size_2):
                    tile_acc = T.alloc_local((1), accum_dtype)
                    tile_acc[0] = 0
                    for i in T.unroll(total_block):
                        hidden_fp32 = T.cast(hidden_share_ptx[t, i, j], accum_dtype)
                        tile_acc[0] += add_local_ptx[t, i] * hidden_fp32
                    mul_5[t, j] = tile_acc[0]

                T.reduce_sum(mul_5, mul_5_sum, dim=0, clear=False)

            for j in T.Parallel(block_size_2):
                dproj_weight_local_ptx[j] = T.cast(mul_5_sum[j] * norm_weight_ptx[j], weight_dtype)
                dnorm_weight_local_ptx[j] = T.cast(mul_5_sum[j] * proj_weight_ptx[j], weight_dtype)

            T.copy(dproj_weight_local_ptx, dproj_weight[ptx_start:ptx_end])
            T.copy(dnorm_weight_local_ptx, dnorm_weight[ptx_start:ptx_end])

    return bwd_kernel


def block_residual_bwd(
    grad,
    prefix_sum,
    block_residual,
    proj_weight,
    norm_weight,
    # fwd cache values
    rsqrt_output,
    max_output,
    exp_output,
    div_output,
    softmax_output,
):
    device = block_residual.device
    num_token, num_blocks, hidden_size = block_residual.shape
    total_block = num_blocks + 1

    dtype = str(block_residual.dtype).replace("torch.", "")
    weight_dtype = str(norm_weight.dtype).replace("torch.", "")

    doutput = torch.empty((num_token, total_block, hidden_size), device=device, dtype=block_residual.dtype)
    dproj_weight = torch.empty((hidden_size), device=device, dtype=norm_weight.dtype)
    dnorm_weight = torch.empty((hidden_size), device=device, dtype=norm_weight.dtype)
    add_workspace = torch.empty((num_token, total_block), device=device, dtype=torch.float32)


    kernel = block_residual_bwd_kernel(
        block_residual.shape[1], 
        block_residual.shape[2], 
        dtype, weight_dtype
    )
        
    kernel(
        grad,
        prefix_sum,
        block_residual,
        norm_weight,
        proj_weight,
        rsqrt_output,
        max_output,
        exp_output,
        div_output,
        softmax_output,
        doutput,
        dproj_weight,
        dnorm_weight,
        add_workspace,
    )

    return (
        doutput[:, num_blocks, :].view(num_token, hidden_size),
        doutput[:, 0:num_blocks, :],
        dproj_weight,
        dnorm_weight,
    )



