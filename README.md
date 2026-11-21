# TileLang-AttnRes

This repository implements the **Attention Residual** architecture from **Kimi K3** using [TileLang](https://github.com/tile-ai/tilelang), including both **forward** and **backward** kernels. The operators are benchmarked against native PyTorch and `torch.compile`.

## Repository Structure

```
Tilelang-AttnRes/
├── assets/                      # Images / resources
├── benchmark/                   # Benchmark and correctness scripts
├── kernels/
│   ├── attn_res_fwd.py          # Forward kernel
│   └── block_residual_bwd.py    # Backward kernel
├── requirements.txt
└── README.md
```


## Benchmark

### Testing environment

- OS: Windows 11
- CPU: AMD Ryzen 7 9800X3D
- GPU: NVIDIA RTX 5080 (16 GB VRAM)
- Dependencies: see [requirements.txt](requirements.txt)

### Forward

**bfloat16:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | rel_err TileLang/Torch | rel_err TileLang/Compile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168  | 0.224 | 4.633 | 1.105 | **20.69×** | 4.19× | 2.34e-05 | 1.90e-05 |
| 2560×4×7168  | 0.241 | 5.365 | 1.397 | **22.24×** | 3.84× | 2.37e-05 | 2.02e-05 |
| 5120×2×7168  | 0.398 | 7.037 | 2.252 | **17.70×** | 3.12× | 1.94e-05 | 1.69e-05 |
| 10240×1×7168 | 0.566 | 9.957 | 3.409 | **17.59×** | 2.92× | 1.68e-05 | 1.35e-05 |
| 7680×8×768   | 0.156 | 3.014 | 1.094 | **19.35×** | 2.75× | 2.55e-05 | 2.68e-05 |
| 768×1×4096   | 0.031 | 0.221 | 0.095 | **7.19×**  | 2.33× | 1.22e-05 | 1.43e-05 |

**float32:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | rel_err TileLang/Torch | rel_err TileLang/Compile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168  | 0.424 | 4.347 | 1.751 | **10.25×** | 2.48× | 1.04e-06 | 1.05e-06 |
| 2560×4×7168  | 0.527 | 4.959 | 2.071 | **9.40×**  | 2.40× | 1.13e-06 | 8.31e-07 |
| 5120×2×7168  | 0.769 | 6.449 | 2.943 | **8.39×**  | 2.19× | 9.85e-07 | 8.12e-07 |
| 10240×1×7168 | 1.279 | 9.000 | 3.204 | **7.04×**  | 2.81× | 8.55e-07 | 6.57e-07 |
| 7680×8×768   | 0.277 | 2.833 | 1.387 | **10.24×** | 2.04× | 5.67e-07 | 5.31e-07 |
| 768×1×4096   | 0.046 | 0.212 | 0.100 | **4.56×**  | 2.11× | 6.97e-07 | 4.93e-07 |

### Backward

**`rel_err` is the maximum global L2 relative error across all four gradients**

**bfloat16:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | rel_err TileLang/Torch | rel_err TileLang/Compile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168 | 0.785 | 15.725 | 3.980 | **20.03×** | 3.95× | 2.33e-04 | 2.67e-04 |
| 2560×4×7168 | 0.849 | 18.064 | 5.179 | **21.27×** | 3.49× | 1.94e-04 | 2.31e-04 |
| 5120×2×7168 | 1.017 | 23.004 | 7.880 | **22.62×** | 2.92× | 1.99e-04 | 1.69e-04 |
| 10240×1×7168 | 1.386 | 32.607 | 11.515 | **23.52×** | 2.83× | 1.94e-04 | 1.62e-04 |
| 7680×8×768 | 0.601 | 10.142 | 4.212 | **16.86×** | 2.41× | 8.36e-05 | 8.73e-05 |
| 768×1×4096 | 0.084 | 0.966 | 0.396 | **11.54×** | 2.44× | 1.51e-04 | 1.05e-04 |

**float32:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | rel_err TileLang/Torch | rel_err TileLang/Compile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168 | 1.625 | 14.788 | 5.512 | **9.10×** | 2.68× | 1.01e-05 | 1.30e-05 |
| 2560×4×7168 | 1.871 | 18.311 | 6.622 | **9.79×** | 2.77× | 1.31e-05 | 1.05e-05 |
| 5120×2×7168 | 2.322 | 21.396 | 10.483 | **9.22×** | 2.04× | 1.09e-05 | 8.40e-06 |
| 10240×1×7168 | 3.281 | 30.027 | 10.138 | **9.15×** | 2.96× | 1.03e-05 | 8.46e-06 |
| 7680×8×768 | 0.936 | 9.553 | 4.486 | **10.20×** | 2.13× | 3.00e-06 | 2.76e-06 |
| 768×1×4096 | 0.100 | 0.911 | 0.379 | **9.11×** | 2.41× | 8.72e-06 | 6.23e-06 |

> **Note**
> - `rel_err = ‖TileLang − baseline‖₂ / ‖baseline‖₂`, computed in `float32` to avoid bf16 accumulation artifacts in the metric itself.
> - Timing uses `tilelang.profiler.bench.do_bench` with `warmup=100`.
> - bf16 backward results are also verified; expected `rel_err` is around `1e-4`.

## Future

- [x] Implement backward

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