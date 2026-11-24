import torch

def measure_peak_memory(func, *args, **kwargs):

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    base_mem = torch.cuda.memory_allocated()

    _ = func(*args, **kwargs)

    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated()
    
    mem_increment_mb = (peak_mem - base_mem) / 1024 / 1024
    return mem_increment_mb
