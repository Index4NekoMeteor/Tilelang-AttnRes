import torch
import tilelang
import itertools
from tilelang import language as T
from tilelang.autotuner import set_autotune_inputs


@tilelang.autotune(
    configs=[
        {"block_size": bs, "threads": th, "num_stages": ns}
        for bs in [256, 512, 1024]
        for th in [64]
        for ns in [1, 2, 3]
    ],
    warmup=4,
    rep=512,
    do_not_specialize=["block_size", "threads", "num_stages"],
)
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
        block_residual: T.Tensor((num_token, num_block, hidden_size), dtype),
        prefix_sum: T.Tensor((num_token, hidden_size), dtype),
        norm_weight: T.Tensor((hidden_size,), weight_dtype),
        proj_weight: T.Tensor((hidden_size,), weight_dtype),
        # output value
        output: T.Tensor((num_token, hidden_size), dtype),
        rsqrt_output: T.Tensor((num_token, total_block), accum_dtype),
        max_output: T.Tensor((num_token, 1), T.int64),
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

            idx_ptx = T.alloc_fragment((total_block,), T.int64)
            idx_sum = T.alloc_fragment((1,), T.int64)

            for i in T.Parallel(total_block):
                idx_ptx[i] = T.if_then_else(
                    hidden_dot_sum_ptx[i] == hidden_weight_max[0],
                    T.cast(i, T.int64),
                    T.int64(total_block),  # ty: ignore
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




def block_residual_fwd(block_residual, prefix_sum, proj_w, norm_w, eps=1e-6):
    num_token, num_block, hidden_size = block_residual.shape
    total_block = num_block + 1

    device = block_residual.device
    max_output = torch.empty(num_token, 1, dtype=torch.int64, device=device)
    div_output = torch.empty(num_token, 1, dtype=torch.float32, device=device)
    output = torch.empty(num_token, hidden_size, dtype=block_residual.dtype, device=device)
    exp_output = torch.empty(num_token, total_block, dtype=torch.float32, device=device)
    rsqrt_output = torch.empty(num_token, total_block, dtype=torch.float32, device=device)
    softmax_output = torch.empty(num_token, total_block, dtype=torch.float32, device=device)

    weight_dtype = str(norm_w.dtype).replace("torch.", "")
    dtype = str(block_residual.dtype).replace("torch.", "")

    with set_autotune_inputs(
        block_residual,
        prefix_sum,
        proj_w,
        norm_w,
        output,
        rsqrt_output,
        max_output,
        exp_output,
        div_output,
        softmax_output,
    ):
        kernel = block_residual_fwd_kernel(
            block_residual.shape[1],
            block_residual.shape[2],
            dtype, weight_dtype, eps=eps
        )
    
    kernel(
        block_residual,
        prefix_sum,
        proj_w,
        norm_w,
        output,
        rsqrt_output,
        max_output,
        exp_output,
        div_output,
        softmax_output,
    )

    return output, rsqrt_output, max_output, exp_output, div_output, softmax_output


if __name__ == "__main__":
    from tilelang.profiler.bench import do_bench
    from torch.profiler import profile, ProfilerActivity

    # from https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py
    def _apply_attn_res(prefix_sum, block_residual, proj, norm):
        """
        prefix_sum:     (num_tokens, hidden_size)
        block_residual: (num_tokens, num_blocks, hidden_size)
        """
        v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
        v_float = v.float()
        variance = v_float.pow(2).mean(-1, keepdim=True)
        k = v_float * torch.rsqrt(variance + 1e-6)
        score_weight = norm.float() * proj.squeeze(0).float()
        scores = (k * score_weight).sum(-1)
        probs = scores.softmax(-1).unsqueeze(1)
        hidden_states = torch.matmul(probs, v_float).squeeze(1)
        return hidden_states.to(v.dtype)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    SHAPES = [
        (1280, 8, 7168),
        (7680, 8, 768),
        (768, 1, 4096),
    ]
    DTYPES = [torch.bfloat16, torch.float32]

    for dtype in DTYPES:
        for shape in SHAPES:
            torch.manual_seed(0)
            bs, N, H = shape
            prefix_sum = torch.randn(bs, H, dtype=dtype, device="cuda")
            block_residual = torch.randn(bs, N, H, dtype=dtype, device="cuda")
            w = torch.randn(H, dtype=torch.float32, device="cuda")
            w2 = torch.randn(H, dtype=torch.float32, device="cuda")

            def Tilelang():
                return block_residual_fwd(block_residual, prefix_sum, w, w2)[0]

            compile_kernel = torch.compile(_apply_attn_res)

            def Torch_compile():
                return compile_kernel(prefix_sum, block_residual, w, w2)

            def Torch():
                return _apply_attn_res(prefix_sum, block_residual, w, w2)

            print(f"\n=== dtype={dtype}, shape={list(shape)} ===")
            torch.cuda.empty_cache()
            t_tile = do_bench(Tilelang, warmup=100)
            torch.cuda.empty_cache()
            t_torch = do_bench(Torch, warmup=100)
            torch.cuda.empty_cache()
            t_comp = do_bench(Torch_compile, warmup=100)

            print(f"Tilelang: {t_tile:.3f} ms | Torch: {t_torch:.3f} ms | Compile: {t_comp:.3f} ms")
            print(f"Tilelang speedup: {t_torch / t_tile:.2f}x | Compile speedup: {t_torch / t_comp:.2f}x")

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            r_tile = Tilelang()
            r_torch = Torch()
            r_comp = Torch_compile()
            atol = 0.01 if dtype == torch.float32 else 0.05
            p1 = torch.allclose(r_tile, r_torch, rtol=0.01, atol=atol)
            p2 = torch.allclose(r_tile, r_comp, rtol=0.01, atol=atol)
            print(f"rel_err   Tile-vs-Torch={rel(r_tile, r_torch):.3e}  Tile-vs-Compile={rel(r_tile, r_comp):.3e}")
            print(f"TileLang vs Torch: {'PASS' if p1 else 'FAIL'} | vs Compile: {'PASS' if p2 else 'FAIL'}")
            