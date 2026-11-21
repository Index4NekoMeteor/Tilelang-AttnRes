# TileLang-AttnRes

This repository implements the **Attention Residual** architecture from **Kimi K3** using [TileLang](https://github.com/tile-ai/tilelang), including both **forward** and **backward** kernels. The operators are benchmarked against native PyTorch and `torch.compile`.

## Repository Structure

```text
Tilelang-AttnRes/
├── assets/                      # Images / resources
├── benchmark/                   # Benchmark scripts
│   ├── benchmark_fwd.py
│   └── benchmark_bwd.py
├── tilelang_attnres/            # Python package
│   ├── __init__.py              # Autograd & nn.Module wrappers
│   └── kernels/
│       ├── attn_res_fwd.py      # Forward kernel
│       └── attn_res_bwd.py      # Backward kernel
├── test.py                      # Correctness verification script
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
| 1280×8×7168  | 0.258 | 5.699 | 1.104 | **22.12×** | 5.16× | 2.34e-05 | 1.90e-05 |
| 2560×4×7168  | 0.243 | 5.711 | 1.854 | **23.53×** | 3.08× | 2.37e-05 | 2.02e-05 |
| 5120×2×7168  | 0.398 | 8.079 | 2.666 | **20.29×** | 3.03× | 1.94e-05 | 1.69e-05 |
| 10240×1×7168 | 0.570 | 11.232| 3.447 | **19.70×** | 3.26× | 1.68e-05 | 1.35e-05 |
| 7680×8×768   | 0.142 | 3.163 | 1.231 | **22.26×** | 2.57× | 2.55e-05 | 2.38e-05 |
| 768×1×4096   | 0.032 | 0.223 | 0.101 | **7.06×**  | 2.21× | 1.22e-05 | 1.43e-05 |

**float32:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | rel_err TileLang/Torch | rel_err TileLang/Compile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168  | 0.432 | 5.121 | 2.152 | **11.84×** | 2.38× | 1.04e-06 | 8.80e-07 |
| 2560×4×7168  | 0.540 | 5.555 | 2.301 | **10.28×** | 2.41× | 1.13e-06 | 8.18e-07 |
| 5120×2×7168  | 0.761 | 8.021 | 2.806 | **10.54×** | 2.86× | 9.85e-07 | 7.60e-07 |
| 10240×1×7168 | 1.298 | 10.252| 3.258 | **7.90×**  | 3.15× | 8.55e-07 | 6.57e-07 |
| 7680×8×768   | 0.280 | 2.966 | 1.680 | **10.61×** | 1.77× | 5.67e-07 | 5.37e-07 |
| 768×1×4096   | 0.048 | 0.211 | 0.104 | **4.38×**  | 2.03× | 6.97e-07 | 4.93e-07 |

### Backward

**`rel_err` is the maximum global L2 relative error across all four gradients**

**bfloat16:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | rel_err TileLang/Torch | rel_err TileLang/Compile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168 | 0.916 | 16.155 | 4.040 | **17.64×** | 4.00× | 2.41e-04 | 2.22e-04 |
| 2560×4×7168 | 0.861 | 18.376 | 5.302 | **21.33×** | 3.47× | 3.21e-04 | 2.43e-04 |
| 5120×2×7168 | 1.021 | 26.337 | 7.984 | **25.80×** | 3.30× | 2.49e-04 | 2.18e-04 |
| 10240×1×7168| 1.406 | 32.962 | 11.492| **23.45×** | 2.87× | 3.07e-04 | 2.31e-04 |
| 7680×8×768  | 0.611 | 10.257 | 5.539 | **16.80×** | 1.85× | 1.15e-04 | 1.12e-04 |
| 768×1×4096  | 0.083 | 0.921 | 0.409 | **11.16×** | 2.25× | 2.90e-04 | 1.52e-04 |

**float32:**

| Shape | TileLang (ms) | Torch (ms) | Torch Compile (ms) | Speedup (TileLang) | Speedup (Compile) | rel_err TileLang/Torch | rel_err TileLang/Compile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1280×8×7168 | 1.658 | 14.967 | 5.566 | **9.03×** | 2.69× | 3.15e-05 | 1.57e-05 |
| 2560×4×7168 | 2.427 | 17.111 | 6.642 | **7.05×** | 2.58× | 2.13e-05 | 1.68e-05 |
| 5120×2×7168 | 2.347 | 21.645 | 10.614| **9.22×** | 2.04× | 2.13e-05 | 1.48e-05 |
| 10240×1×7168| 3.316 | 33.348 | 10.281| **10.06×**| 3.24× | 1.94e-05 | 1.86e-05 |
| 7680×8×768  | 0.943 | 11.038 | 4.543 | **11.71×**| 2.43× | 6.04e-06 | 5.77e-06 |
| 768×1×4096  | 0.112 | 0.901 | 0.375 | **8.07×** | 2.40× | 1.59e-05 | 1.41e-05 |

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