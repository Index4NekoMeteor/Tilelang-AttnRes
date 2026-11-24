import torch
import warnings
from tilelang.profiler.bench import do_bench
import torch._inductor.config as inductor_config
from tilelang_attnres.kernels import block_residual_fwd, block_residual_bwd, apply_attn_res

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")
torch._dynamo.config.recompile_limit = 16
if hasattr(inductor_config.triton, "coalesce_tiling_analysis"):
    inductor_config.triton.coalesce_tiling_analysis = False


SHAPES = [
    (1280, 8, 7168),
    (2560, 4, 7168),
    (5120, 2, 7168),
    (10240, 1, 7168),
    (7680, 8, 768),
    (768, 1, 4096),
]
DTYPES = [torch.bfloat16, torch.float32]

compile = torch.compile(apply_attn_res)

for dtype in DTYPES:
    for shape in SHAPES:
        torch.manual_seed(0)
        bs, N, H = shape
        prefix_sum = torch.randn(bs, H, dtype=dtype, device="cuda", requires_grad=True)
        block_residual = torch.randn(bs, N, H, dtype=dtype, device="cuda", requires_grad=True)
        proj_weight = torch.ones(H, dtype=torch.float32, device="cuda", requires_grad=True)
        norm_weight = torch.ones(H, dtype=torch.float32, device="cuda", requires_grad=True) + 1
        dout = torch.randn(bs, H, dtype=dtype, device="cuda", requires_grad=True)

        output, rsqrt_output, max_output, exp_output, div_output, softmax_output = block_residual_fwd(
            prefix_sum, block_residual, proj_weight, norm_weight
        )

        def Tilelang():
            return block_residual_bwd(
                dout,
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

        def Torch():
            return torch.autograd.grad(
                apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight),
                [prefix_sum, block_residual, proj_weight, norm_weight],
                dout,
                retain_graph=False,
            )

        def Torch_compile():
            return torch.autograd.grad(
                compile(prefix_sum, block_residual, proj_weight, norm_weight),
                [prefix_sum, block_residual, proj_weight, norm_weight],
                dout,
                retain_graph=False,
            )

        print(f"\n=== dtype={dtype}, shape={list(shape)} ===")
        torch.cuda.empty_cache()
        t_tile: float = do_bench(Tilelang, warmup=100, _n_repeat=50)
        torch.cuda.empty_cache()
        t_torch: float = do_bench(Torch, warmup=100, _n_repeat=50)
        torch.cuda.empty_cache()
        t_comp: float = do_bench(Torch_compile, warmup=100, _n_repeat=50)

        print(f"Tilelang: {t_tile:.3f} ms | Torch: {t_torch:.3f} ms | Compile: {t_comp:.3f} ms")
        print(f"Tilelang speedup: {t_torch / t_tile:.2f}x | Compile speedup: {t_torch / t_comp:.2f}x")