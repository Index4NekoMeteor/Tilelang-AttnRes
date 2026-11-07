# Tilelang-AttnRes(forward only)

This repository implements the [Attention Residual](https://arxiv.org/pdf/2603.15031) architecture from Kimi K3 using [TileLang](https://github.com/tile-ai/tilelang).

![Attention Residuals](assets\image.png)

### 1. Benchmark

#### 1.1 Testing environment
OS: Windows 11  
CPU: 9800x3d  
GPU: RTX 5080(16G vram)  
Requirements: [requirements.txt](./requirements.txt)


#### 1.2 Tested with input shape `[1280, 8, 7168] | [7680, 8, 768] | [768, 1, 4096]` and `bfloat16` precision.

| Shape | Tilelang (ms) | Torch (ms) | Torch Compile (ms) | Speedup(Tilelang) | Speedup(Torch Compile) | rel_err Tilelang/Torch | rel_err Tilelang/Torch Compile |
|---|---|---|---|---|---|---|---|
| 1280×8×7168 | 0.246 | 4.698 | 1.120 | **19.14×** | 4.20× | 2.24e-05 | 1.98e-05 |
| 7680×8×768  | 0.165 | 3.030 | 1.163 | **18.41×** | 2.60× | 2.48e-05 | 2.46e-05 |
| 768×1×4096  | 0.035 | 0.231 | 0.090 | **6.57×**  | 2.56× | 1.38e-05 | 1.52e-05 |

#### 1.3 Tested with input shape `[1280, 8, 7168] | [7680, 8, 768] | [768, 1, 4096]` and `float32` precision.

| Shape | Tilelang (ms) | Torch (ms) | Torch Compile (ms) | Speedup(Tilelang) | Speedup(Torch Compile) | rel_err Tilelang/Torch | rel_err Tilelang/Torch Compile |
|---|---|---|---|---|---|---|---|
| 1280×8×7168 | 0.454 | 4.479 | 2.230 | **9.87×** | 2.01× | 1.03e-06 | 9.95e-07 |
| 7680×8×768  | 0.299 | 2.848 | 1.259 | **9.54×** | 2.26× | 5.31e-07 | 5.26e-07 |
| 768×1×4096  | 0.054 | 0.225 | 0.112 | **4.14×** | 2.01× | 7.47e-07 | 4.86e-07 |

> **Notes**
> - `rel_err = ‖Tilelang − baseline‖₂ / ‖baseline‖₂`
> - Timing uses `tilelang.profiler.bench.do_bench` with `warmup=100`


### 2. Usage

```python
import torch
from attn_res_fwd import block_residual_fwd

# block_residual: (num_tokens, num_blocks, hidden_size)
# prefix_sum:     (num_tokens, hidden_size)
# proj_w, norm_w: (hidden_size,)
num_tokens, num_blocks, hidden_size = 1280, 8, 7168
dtype = torch.bfloat16
device = "cuda"

prefix_sum = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
block_residual = torch.randn(num_tokens, num_blocks, hidden_size, dtype=dtype, device=device)

proj_w = torch.randn(hidden_size, dtype=torch.float32, device=device)
norm_w = torch.randn(hidden_size, dtype=torch.float32, device=device)

# forward
output, rsqrt, max_idx, exp_out, div_out, softmax_out = block_residual_fwd(
    block_residual, prefix_sum, proj_w, norm_w, eps=1e-6
)

# output:      (num_tokens, hidden_size)          bf16 --- result ---

# rsqrt:       (num_tokens, num_blocks + 1)       fp32 --- saved for backward ---
# max_idx:     (num_tokens, 1)                    int64
# exp_out:     (num_tokens, num_blocks + 1)       fp32
# div_out:     (num_tokens, 1)                    fp32
# softmax_out: (num_tokens, num_blocks + 1)       fp32
```
> **Note**: Current version is forward-only. The intermediate tensors are returned for analysis/debugging purposes. You can discard them in production to save memory.

### 3. Future
Implement backward

### 4. Citation
```bib
@misc{chen2026attnres,
  title         = {Attention Residuals},
  author        = {Kimi Team  and Chen, Guangyu  and Zhang, Yu  and Su, Jianlin  and Xu, Weixin  and Pan, Siyuan  and Wang, Yaoyu  and Wang, Yucheng  and Chen, Guanduo  and Yin, Bohong  and Chen, Yutian  and Yan, Junjie  and Wei, Ming  and Zhang, Y.  and Meng, Fanqing  and Hong, Chao  and Xie, Xiaotong  and Liu, Shaowei  and Lu, Enzhe  and Tai, Yunpeng  and Chen, Yanru  and Men, Xin  and Guo, Haiqing  and Charles, Y.  and Lu, Haoyu  and Sui, Lin  and Zhu, Jinguo  and Zhou, Zaida  and He, Weiran  and Huang, Weixiao  and Xu, Xinran  and Wang, Yuzhi  and Lai, Guokun  and Du, Yulun  and Wu, Yuxin  and Yang, Zhilin  and Zhou, Xinyu},
  year          = {2026},
  archiveprefix = {arXiv},
  eprint        = {2603.15031},
  primaryclass  = {cs.CL}
}
```