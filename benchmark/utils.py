import torch


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()

def check(name, a, b):
    assert len(a) == len(b), f"{name}: len {len(a)} vs {len(b)}"
    all_ok = True
    for i, (x, y) in enumerate(zip(a, b)):
        is_fp32 = y.dtype == torch.float32
        x = x.float()
        y = y.float()

        err = rel(x, y)

        threshold = 1e-4 if is_fp32 else 2e-3
        ok = err < threshold

        print(f"  [{name}] grad[{i}]: {'PASS' if ok else 'FAIL'}  rel={err:.3e}")
        all_ok &= ok
    return all_ok