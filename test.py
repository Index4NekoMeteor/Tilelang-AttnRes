import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tilelang_attnres import block_residual_fwd, block_residual_bwd, attn_res_tilelang, AttnRes
except ImportError:
    print("⚠️ 尝试从 tilelang_attnres 导入失败，退回到本地当前目录导入...")
    try:
        from tilelang_attnres.kernels import block_residual_fwd, block_residual_bwd
        from tilelang_attnres import attn_res_tilelang, AttnRes
    except ImportError:
        print("❌ 导入失败，请确保路径正确。")
        sys.exit(1)


def get_inputs(device="cuda", dtype=torch.float32):
    bs, N, H = 64, 4, 512
    # 固定随机种子，保证每次跑的数据一样
    torch.manual_seed(42)
    prefix_sum = torch.randn(bs, H, dtype=dtype, device=device, requires_grad=True)
    block_residual = torch.randn(bs, N, H, dtype=dtype, device=device, requires_grad=True)
    proj_weight = torch.randn(H, dtype=torch.float32, device=device, requires_grad=True)
    norm_weight = torch.randn(H, dtype=torch.float32, device=device, requires_grad=True)
    dout = torch.randn(bs, H, dtype=dtype, device=device)
    return bs, N, H, prefix_sum, block_residual, proj_weight, norm_weight, dout


def rel(a, b):
    """全局 L2 相对误差"""
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def apply_attn_res_ref(prefix_sum, block_residual, proj_weight, norm_weight, eps=1e-6):
    """纯 PyTorch 的 Reference 实现，用于提供 ground truth"""
    # 注意：这里根据你 attn_res_fwd 里的数学逻辑手写
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)  # [bs, N+1, H]
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)

    score_weight = norm_weight.float() * proj_weight.float()
    scores = (k * score_weight).sum(-1)  # [bs, N+1]

    probs = scores.softmax(-1).unsqueeze(1)  # [bs, 1, N+1]
    hidden_states = torch.matmul(probs, v_float).squeeze(1)  # [bs, H]

    return hidden_states.to(v.dtype)


def test_correctness():
    print("\n" + "=" * 50)
    print("============ 正确性数值验证 (vs PyTorch) ============")
    print("=" * 50)

    bs, N, H, prefix_sum, block_residual, proj_w, norm_w, dout = get_inputs()
    eps = 1e-6

    # ---------------------------------------------------------
    # 1. Reference (PyTorch 自动求导)
    # ---------------------------------------------------------
    print("-> 正在计算 PyTorch Reference...")
    ref_out = apply_attn_res_ref(prefix_sum, block_residual, proj_w, norm_w, eps)
    ref_dprefix, ref_dblock, ref_dproj, ref_dnorm = torch.autograd.grad(
        outputs=ref_out, inputs=[prefix_sum, block_residual, proj_w, norm_w], grad_outputs=dout, retain_graph=True
    )

    # ---------------------------------------------------------
    # 2. TileLang (使用 autograd.Function 封装)
    # ---------------------------------------------------------
    print("-> 正在计算 TileLang...")
    tl_out = attn_res_tilelang(prefix_sum, block_residual, proj_w, norm_w, eps)
    tl_dprefix, tl_dblock, tl_dproj, tl_dnorm = torch.autograd.grad(
        outputs=tl_out, inputs=[prefix_sum, block_residual, proj_w, norm_w], grad_outputs=dout, retain_graph=True
    )

    # ---------------------------------------------------------
    # 3. 对比数值
    # ---------------------------------------------------------
    print("\n--- 前向对比 ---")
    err_out = rel(tl_out, ref_out)
    print(f"  output:         rel_err = {err_out:.3e}")
    assert err_out < 1e-4, "❌ 前向输出误差过大！"

    print("\n--- 反向梯度对比 ---")
    err_prefix = rel(tl_dprefix, ref_dprefix)
    err_block = rel(tl_dblock, ref_dblock)
    err_proj = rel(tl_dproj, ref_dproj)
    err_norm = rel(tl_dnorm, ref_dnorm)

    print(f"  dprefix_sum:    rel_err = {err_prefix:.3e}")
    print(f"  dblock_residual:rel_err = {err_block:.3e}")
    print(f"  dproj_weight:   rel_err = {err_proj:.3e}")
    print(f"  dnorm_weight:   rel_err = {err_norm:.3e}")

    # 设定阈值 (float32 下正常应该在 1e-5 以下)
    threshold = 1e-4
    assert err_prefix < threshold, "❌ dprefix_sum 误差过大！"
    assert err_block < threshold, "❌ dblock_residual 误差过大！"
    assert err_proj < threshold, "❌ dproj_weight 误差过大！"
    assert err_norm < threshold, "❌ dnorm_weight 误差过大！"

    print("\n✅ 前向和反向的数值验证全部通过！(rel_err 均小于 1e-4)")


def test_nn_module_math():
    """验证 nn.Module 内部计算是否正常"""
    print("\n=== 测试 nn.Module 封装 ===")
    bs, N, H, prefix_sum, block_residual, _, _, dout = get_inputs()

    # 初始化 Module
    module = AttnRes(hidden_size=H).to("cuda")
    # 为了对比，把 module 内的 weight 初始化成刚才的数据
    with torch.no_grad():
        module.proj_weight.copy_(torch.randn_like(module.proj_weight))
        module.norm_weight.copy_(torch.randn_like(module.norm_weight))

    tl_out = module(prefix_sum, block_residual)

    # 用 reference 算一次
    ref_out = apply_attn_res_ref(prefix_sum, block_residual, module.proj_weight, module.norm_weight)

    err = rel(tl_out, ref_out)
    print(f"Module Forward vs Reference: rel_err = {err:.3e}")
    assert err < 1e-4, "❌ Module 前向误差过大！"

    print("✅ Module 封装验证通过！")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 CUDA 才能运行测试。")
        exit(0)

    try:
        test_correctness()
        test_nn_module_math()
        print("\n🎉 所有测试通过，且与 PyTorch Reference 误差极小！完美！")
    except AssertionError as e:
        print(f"\n{e}")
    except Exception as e:
        import traceback

        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
