import torch
import warnings
from utils import check, rel
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
        proj_weight = torch.nn.Linear(H, 1, dtype=torch.float32, device="cuda")
        norm_weight = torch.nn.RMSNorm(H, dtype=torch.float32, device="cuda", eps=1e-6)
        norm_weight.variance_epsilon = norm_weight.eps
        dout = torch.randn(bs, H, dtype=dtype, device="cuda", requires_grad=True)

        output, rsqrt_output, max_output, exp_output, div_output, softmax_output = block_residual_fwd(
            prefix_sum, block_residual, proj_weight.weight.view(-1), norm_weight.weight.view(-1)
        )

        def Tilelang():
            return block_residual_bwd(
                dout,
                prefix_sum,
                block_residual,
                proj_weight.weight.view(-1),
                norm_weight.weight.view(-1),
                rsqrt_output,
                max_output,
                exp_output,
                div_output,
                softmax_output,
            )

        def Torch():
            return torch.autograd.grad(
                apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight),
                [prefix_sum, block_residual, proj_weight.weight, norm_weight.weight],
                dout,
                retain_graph=False,
            )

        def Torch_compile():
            return torch.autograd.grad(
                compile(prefix_sum, block_residual, proj_weight, norm_weight),
                [prefix_sum, block_residual, proj_weight.weight, norm_weight.weight],
                dout,
                retain_graph=False,
            )

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        r_tile = Tilelang()
        r_torch = Torch()
        r_comp = Torch_compile()

        print(f"\n=== dtype={dtype}, shape={list(shape)} ===")
        p1 = check("Tile-vs-Torch", r_tile, r_torch)
        p2 = check("Tile-vs-Compile", r_tile, r_comp)
        print(f"TileLang vs Torch: {'PASS' if p1 else 'FAIL'} | vs Compile: {'PASS' if p2 else 'FAIL'}")
