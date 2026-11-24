import torch
import warnings
from utils import rel
from tilelang.profiler.bench import do_bench
import torch._inductor.config as inductor_config
from tilelang_attnres.kernels import apply_attn_res
from tilelang_attnres.kernels.attn_res_fwd_copy import block_residual_fwd


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

compile_kernel = torch.compile(apply_attn_res)

for dtype in DTYPES:
    for shape in SHAPES:
        torch.manual_seed(0)
        bs, N, H = shape
        prefix_sum = torch.randn(bs, H, dtype=dtype, device="cuda")
        block_residual = torch.randn(bs, N, H, dtype=dtype, device="cuda")
        proj_weight = torch.nn.Linear(H, 1, dtype=torch.float32, device="cuda")
        norm_weight = torch.nn.RMSNorm(H, dtype=torch.float32, device="cuda", eps=1e-6)
        norm_weight.variance_epsilon = norm_weight.eps

        def Tilelang():
            return block_residual_fwd(prefix_sum, block_residual, proj_weight.weight.view(-1), norm_weight.weight.view(-1))

        def Torch_compile():
            return compile_kernel(prefix_sum, block_residual, proj_weight, norm_weight)

        def Torch():
            return apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight)

        print(f"\n=== dtype={dtype}, shape={list(shape)} ===")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        r_tile = Tilelang()[0]
        r_torch = Torch()
        r_comp = Torch_compile()
        atol = 0.01 if dtype == torch.float32 else 0.05
        p1 = torch.allclose(r_tile, r_torch, rtol=0.01, atol=atol)
        p2 = torch.allclose(r_tile, r_comp, rtol=0.01, atol=atol)
        print(f"rel_err   Tile-vs-Torch={rel(r_tile, r_torch):.3e}  Tile-vs-Compile={rel(r_tile, r_comp):.3e}")
        print(f"TileLang vs Torch: {'PASS' if p1 else 'FAIL'} | vs Compile: {'PASS' if p2 else 'FAIL'}")
