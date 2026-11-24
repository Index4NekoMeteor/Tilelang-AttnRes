import torch
import warnings
from tilelang.profiler.bench import do_bench
from benchmark.utils import measure_peak_memory
import torch._inductor.config as inductor_config
from tilelang_attnres import attn_res_tilelang, apply_attn_res


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
        prefix_sum = torch.randn(bs, H, dtype=dtype, device="cuda", requires_grad=True)
        block_residual = torch.randn(bs, N, H, dtype=dtype, device="cuda", requires_grad=True)
        proj_weight = torch.nn.Linear(H, 1, dtype=torch.float32, device="cuda")
        norm_weight = torch.nn.RMSNorm(H, dtype=torch.float32, device="cuda", eps=1e-6)
        norm_weight.variance_epsilon = norm_weight.eps
        dout = torch.randn(bs, H, dtype=dtype, device="cuda", requires_grad=True)

        def Tilelang():
            return attn_res_tilelang(prefix_sum, block_residual, proj_weight, norm_weight)

        def Torch_compile():
            return compile_kernel(prefix_sum, block_residual, proj_weight, norm_weight)

        def Torch():
            return apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight)

        print(f"\n=== dtype={dtype}, shape={list(shape)} ===")
        torch.cuda.empty_cache()
        t_tile: float = do_bench(Tilelang, warmup=100, _n_repeat=100, backend="cupti")
        torch.cuda.empty_cache()
        t_torch: float = do_bench(Torch, warmup=100, _n_repeat=100, backend="cupti")
        torch.cuda.empty_cache()
        t_comp: float = do_bench(Torch_compile, warmup=100, _n_repeat=100, backend="cupti")

        tilelang_peak_memory = measure_peak_memory(Tilelang)
        torch_peak_memory = measure_peak_memory(Torch)
        comp_peak_memory = measure_peak_memory(Torch_compile)

        print(f"Time -> Tilelang: {t_tile:.3f} ms | Torch: {t_torch:.3f} ms | Compile: {t_comp:.3f} ms")
        print(f"Speedup -> Tilelang: {t_torch / t_tile:.2f}x | Compile: {t_torch / t_comp:.2f}x")
        print(
            f"Mem  -> Tilelang: {tilelang_peak_memory:.2f} MB | Torch: {torch_peak_memory:.2f} MB | Compile: {comp_peak_memory:.2f} MB"
        )
