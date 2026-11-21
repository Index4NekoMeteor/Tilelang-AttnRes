import torch
import tilelang
from tilelang import language as T


# @tilelang.autotune(
#     configs=[
#         {"block_size": bs, "threads": th, "num_stages": ns}
#         for bs in [256, 512, 1024]
#         for th in [64]
#         for ns in [1, 2, 3]
#     ],
#     warmup=3,
#     rep=5,
#     do_not_specialize=["block_size", "threads", "num_stages"],
# )
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def block_residual_fwd_kernel(
    num_block,
    hidden_size,
    dtype,
    weight_dtype,
    eps=1e-6,
    block_size=1024,
    threads=64,
    num_stages: int = 2,
    accum_dtype=T.float32,
):
    num_token = T.dynamic("num_token")
    total_block = num_block + 1

    @T.prim_func
    def fwd_kernel(
        prefix_sum: T.Tensor((num_token, hidden_size), dtype),
        block_residual: T.Tensor((num_token, num_block, hidden_size), dtype),
        norm_weight: T.Tensor((hidden_size,), weight_dtype),
        proj_weight: T.Tensor((hidden_size,), weight_dtype),
        # output value
        output: T.Tensor((num_token, hidden_size), dtype),
        rsqrt_output: T.Tensor((num_token, total_block), accum_dtype),
        max_output: T.Tensor((num_token, 1), T.int32),
        exp_output: T.Tensor((num_token, total_block), accum_dtype),
        div_output: T.Tensor((num_token, 1), accum_dtype),
        softmax_output: T.Tensor((num_token, total_block), accum_dtype),
    ):
        with T.Kernel(num_token, threads=threads) as token:
            inv = T.float32(1.0) / hidden_size  # ty: ignore

            rsqrt = T.alloc_shared((total_block,), accum_dtype)

            hidden_weight_max = T.alloc_fragment((1,), accum_dtype)
            hidden_dot_sum_ptx = T.alloc_fragment((total_block,), accum_dtype)
            hidden_square_sum_ptx = T.alloc_fragment((total_block,), accum_dtype)

            divisor = T.alloc_fragment((1,), accum_dtype)
            result_ptx = T.alloc_fragment((block_size,), accum_dtype)

            T.clear(hidden_dot_sum_ptx)
            T.clear(hidden_square_sum_ptx)

            hidden_share_ptx = T.alloc_shared((total_block, block_size), dtype)

            for block in T.Pipelined(T.ceildiv(hidden_size, block_size), num_stages=num_stages):
                ptx = block * block_size

                tile_sum = T.alloc_fragment((total_block,), accum_dtype)
                tile_dot = T.alloc_fragment((total_block,), accum_dtype)
                
                weight = T.alloc_fragment((block_size,), accum_dtype)
                hidden_dot_ptx = T.alloc_fragment((total_block, block_size), accum_dtype)
                hidden_square_ptx = T.alloc_fragment((total_block, block_size), accum_dtype)

                norm_weight_share_ptx = T.alloc_fragment((block_size,), accum_dtype)
                proj_weight_share_ptx = T.alloc_fragment((block_size,), accum_dtype)

                T.copy(block_residual[token, :, ptx : ptx + block_size], hidden_share_ptx[0:num_block, :])
                T.copy(prefix_sum[token, ptx : ptx + block_size], hidden_share_ptx[num_block:total_block, :])
                T.copy(norm_weight[ptx : ptx + block_size], norm_weight_share_ptx)
                T.copy(proj_weight[ptx : ptx + block_size], proj_weight_share_ptx)

                for j in T.Parallel(block_size):
                    weight[j] = norm_weight_share_ptx[j] * proj_weight_share_ptx[j]

                for i, j in T.Parallel(total_block, block_size):
                    hidden_fp32 = T.cast(hidden_share_ptx[i, j], accum_dtype)
                    hidden_dot_ptx[i, j] = hidden_fp32 * weight[j]
                    hidden_square_ptx[i, j] = hidden_fp32 * hidden_fp32

                T.reduce_sum(hidden_dot_ptx, tile_dot, dim=-1)
                T.reduce_sum(hidden_square_ptx, tile_sum, dim=-1)
                for i in T.Parallel(total_block):
                    hidden_dot_sum_ptx[i] += tile_dot[i]
                    hidden_square_sum_ptx[i] += tile_sum[i]

            for i in T.Parallel(total_block):
                rsqrt[i] = T.rsqrt(hidden_square_sum_ptx[i] * inv + eps)

            for i in T.Parallel(total_block):
                hidden_dot_sum_ptx[i] = hidden_dot_sum_ptx[i] * rsqrt[i]

            T.copy(rsqrt, rsqrt_output[token, :])

            T.reduce_max(hidden_dot_sum_ptx, hidden_weight_max, dim=-1)

            idx_ptx = T.alloc_fragment((total_block,), T.int32)
            idx_sum = T.alloc_fragment((1,), T.int32)

            for i in T.Parallel(total_block):
                idx_ptx[i] = T.if_then_else(
                    hidden_dot_sum_ptx[i] == hidden_weight_max[0],
                    T.cast(i, T.int32),
                    T.int32(total_block),  # ty: ignore
                )
            T.reduce_min(idx_ptx, idx_sum, dim=-1)
            T.copy(idx_sum, max_output[token, :])

            for i in T.Parallel(total_block):
                hidden_dot_sum_ptx[i] = T.exp(hidden_dot_sum_ptx[i] - hidden_weight_max[0])

            T.copy(hidden_dot_sum_ptx, exp_output[token, :])

            T.reduce_sum(hidden_dot_sum_ptx, divisor, dim=-1)

            inv_2 = T.float32(1.0) / divisor[0]  # ty: ignore
            T.copy(divisor, div_output[token, :])

            for i in T.Parallel(total_block):
                hidden_dot_sum_ptx[i] = hidden_dot_sum_ptx[i] * inv_2

            T.copy(hidden_dot_sum_ptx, softmax_output[token, :])

            for block in T.Pipelined(T.ceildiv(hidden_size, block_size), num_stages=num_stages):
                T.clear(result_ptx)

                ptx = block * block_size
                T.copy(block_residual[token, :, ptx : ptx + block_size], hidden_share_ptx[0:num_block, :])
                T.copy(prefix_sum[token, ptx : ptx + block_size], hidden_share_ptx[num_block:total_block, :])

                for k, j in T.Parallel(total_block, block_size):
                    result_ptx[j] += hidden_dot_sum_ptx[k] * T.cast(hidden_share_ptx[k, j], accum_dtype)

                T.copy(result_ptx, output[token, ptx : ptx + block_size])

    return fwd_kernel


def block_residual_fwd(prefix_sum, block_residual, proj_weight, norm_weight, eps=1e-6):
    num_token, num_block, hidden_size = block_residual.shape
    total_block = num_block + 1

    device = block_residual.device
    max_output = torch.empty(num_token, 1, dtype=torch.int32, device=device)
    div_output = torch.empty(num_token, 1, dtype=torch.float32, device=device)
    output = torch.empty(num_token, hidden_size, dtype=block_residual.dtype, device=device)
    exp_output = torch.empty(num_token, total_block, dtype=torch.float32, device=device)
    rsqrt_output = torch.empty(num_token, total_block, dtype=torch.float32, device=device)
    softmax_output = torch.empty(num_token, total_block, dtype=torch.float32, device=device)

    weight_dtype = str(norm_weight.dtype).replace("torch.", "")
    dtype = str(block_residual.dtype).replace("torch.", "")


    kernel = block_residual_fwd_kernel(
        block_residual.shape[1],
        block_residual.shape[2],
        dtype, weight_dtype, eps=eps
    )
    
    kernel(
        prefix_sum,
        block_residual,
        proj_weight,
        norm_weight,
        output,
        rsqrt_output,
        max_output,
        exp_output,
        div_output,
        softmax_output,
    )

    return output, rsqrt_output, max_output, exp_output, div_output, softmax_output