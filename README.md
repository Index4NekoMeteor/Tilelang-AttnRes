# TileLang-AttnRes

This repository implements the **Attention Residual** architecture from **Kimi K3** using [TileLang](https://github.com/tile-ai/tilelang), including both **forward** and **backward** kernels. The operators are benchmarked against native PyTorch and `torch.compile`.

## Repository Structure

```text
Tilelang-AttnRes/
├── assets/                      # Images / resources
├── benchmark/                   # Benchmark scripts
│   ├── benchmark_autograd_wrapper_bwd.py
│   ├── benchmark_autograd_wrapper_fwd.py
|   ├── benchmark_bwd.py
|   ├── benchmark_fwd.py
|   └── utils.py
├── testing/                     # Benchmark scripts
│   ├── test_autograd_wrapper_bwd.py
│   ├── test_autograd_wrapper_fwd.py
|   ├── test_bwd_correctness.py
|   ├── test_fwd_correctness.py
|   └── utils.py
├── tilelang_attnres/            # Python package
│   ├── __init__.py              # Autograd & nn.Module wrappers
│   └── kernels/
│       ├── attn_res_fwd.py      # Forward kernel
│       ├── attn_res_bwd.py      # Backward kernel
│       └── baseline.py
├── requirements.txt
└── README.md
```

## Usage

### Install

```bash
pip install tilelang-attnres
# or 
pip install git+https://github.com/Index4NekoMeteor/tilelang-attnres.git
```

### Replace the original function(training/inference)

In modeling_kimi_k3.py
```python
from tilelang_attnres import attn_res_tilelang


class KimiDecoderLayer(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        ...

    def _forward_attn_residual(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        block_residual: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states

        # training
        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = attn_res_tilelang(  # replace _apply_attn_res
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)
        ...
        
        # inference
        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = block_residual_inference(  # replace _apply_attn_res
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)
        ...
```


## Benchmark

### Testing environment

- OS: Windows 11
- CPU: AMD Ryzen 7 9800X3D
- GPU: NVIDIA RTX 5080 (16 GB VRAM)
- Dependencies: see [requirements.txt](requirements.txt)

### Forward

**bfloat16:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | Mem TileLang | Mem Torch | Mem Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168  | 0.334 | 5.336 | 1.281 | **15.99×** | 4.16× | 17.64 MB | 1105.66 MB | 368.59 MB |
| 2560×4×7168  | 0.246 | 5.596 | 1.437 | **22.74×** | 3.90× | 35.17 MB | 1226.17 MB | 456.10 MB |
| 5120×2×7168  | 0.326 | 7.665 | 2.303 | **23.49×** | 3.33× | 70.21 MB | 1470.20 MB | 630.12 MB |
| 10240×1×7168 | 0.505 | 10.096| 3.503 | **19.98×** | 2.88× | 140.31 MB| 1960.26 MB | 980.16 MB |
| 7680×8×768   | 0.292 | 3.022 | 0.765 | **10.34×** | 3.95× | 12.10 MB | 709.54 MB  | 236.78 MB |
| 768×1×4096   | 0.019 | 0.192 | 0.077 | **9.98×**  | 2.48× | 6.02 MB  | 84.03 MB   | 42.01 MB  |

**float32:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | Mem TileLang | Mem Torch | Mem Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168  | 0.425 | 4.355 | 1.250 | **10.24×** | 3.48× | 35.14 MB  | 948.16 MB  | 351.13 MB |
| 2560×4×7168  | 0.492 | 5.003 | 1.522 | **10.17×** | 3.29× | 70.17 MB  | 1050.17 MB | 420.15 MB |
| 5120×2×7168  | 0.678 | 6.532 | 2.304 | **9.63×**  | 2.83× | 140.21 MB | 1260.20 MB | 560.18 MB |
| 10240×1×7168 | 1.040 | 9.151 | 3.332 | **8.80×**  | 2.75× | 280.31 MB | 1680.26 MB | 840.23 MB |
| 7680×8×768   | 0.333 | 2.852 | 0.874 | **8.56×**  | 3.26× | 23.35 MB  | 608.29 MB  | 225.79 MB |
| 768×1×4096   | 0.033 | 0.181 | 0.089 | **5.51×**  | 2.02× | 12.02 MB  | 72.03 MB   | 36.02 MB  |

### Backward

**`rel_err` is the maximum global L2 relative error across all four gradients**

**bfloat16:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | Mem TileLang | Mem Torch | Mem Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168  | 0.891 | 15.864 | 4.025  | **17.80×** | 3.94× | 175.22 MB | 1913.55 MB | 1123.14 MB |
| 2560×4×7168  | 0.806 | 18.291 | 5.300  | **22.69×** | 3.45× | 210.25 MB | 2135.05 MB | 1260.16 MB |
| 5120×2×7168  | 1.026 | 23.424 | 8.113  | **22.83×** | 2.89× | 280.30 MB | 2590.05 MB | 1566.50 MB |
| 10240×1×7168 | 1.501 | 33.302 | 11.801 | **22.18×** | 2.82× | 420.39 MB | 3500.05 MB | 1820.23 MB |
| 7680×8×768   | 0.636 | 10.189 | 2.604  | **16.03×** | 3.91× | 113.36 MB | 1226.26 MB | 730.36 MB  |
| 768×1×4096   | 0.075 | 0.847  | 0.330  | **11.25×** | 2.57× | 18.07 MB  | 174.04 MB  | 78.02 MB   |

**float32:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | Mem TileLang | Mem Torch | Mem Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168  | 1.681 | 14.912 | 4.243  | **8.87×**  | 3.51× | 351.22 MB | 1931.05 MB | 983.14 MB  |
| 2560×4×7168  | 1.976 | 17.129 | 5.214  | **8.67×**  | 3.29× | 420.25 MB | 2170.05 MB | 1120.16 MB |
| 5120×2×7168  | 2.519 | 21.783 | 7.458  | **8.65×**  | 2.92× | 560.30 MB | 2660.05 MB | 1426.50 MB |
| 10240×1×7168 | 3.716 | 30.747 | 10.068 | **8.27×**  | 3.05× | 840.39 MB | 3640.05 MB | 1498.21 MB |
| 7680×8×768   | 0.854 | 9.596  | 2.780  | **11.24×** | 3.45× | 225.86 MB | 1237.51 MB | 640.36 MB  |
| 768×1×4096   | 0.105 | 0.860  | 0.330  | **8.19×**  | 2.61× | 36.07 MB  | 180.04 MB  | 60.20 MB   |

> **Note**
> - `rel_err = ‖TileLang − baseline‖₂ / ‖baseline‖₂`, computed in `float32` to avoid bf16 accumulation artifacts in the metric itself.
> - Timing uses `tilelang.profiler.bench.do_bench` with `warmup=100`.
> - bf16 backward results are also verified; expected `rel_err` is around `1e-4`.

## Citation

```bibtex
@misc{chen2026attnres,
  title         = {Attention Residuals},
  author        = {Kimi Team and Chen, Guangyu and Zhang, Yu and Su, Jianlin and Xu, Weixin and Pan, Siyuan and Wang, Yaoyu and Wang, Yucheng and Chen, Guanduo and Yin, Bohong and Chen, Yutian and Yan, Junjie and Wei, Ming and Zhang, Y. and Meng, Fanqing and Hong, Chao and Xie, Xiaotong and Liu, Shaowei and Lu, Enzhe and Tai, Yunpeng and Chen, Yanru and Men, Xin and Guo, Haiqing and Charles, Y. and Lu, Haoyu and Sui, Lin and Zhu, Jinguo and Zhou, Zaida and He, Weiran and Huang, Weixiao and Xu, Xinran and Wang, Yuzhi and Lai, Guokun and Du, Yulun and Wu, Yuxin and Yang, Zhilin and Zhou, Xinyu},
  year          = {2026},
  archiveprefix = {arXiv},
  eprint        = {2603.15031},
  primaryclass  = {cs.CL}
}
```

## License

This project is licensed under the terms in the [LICENSE](LICENSE) file.