"""Forward-pass timing of the async-copy vs cooperative-copy kernels.

    MFA_ASYNC_COPY=1 python scripts/bench_async_copy.py
    MFA_ASYNC_COPY=0 python scripts/bench_async_copy.py
"""
import os, sys, time, gc, json
import torch
import torch.nn.functional as F
from mps_flash_attn import flash_attention

mode = os.environ.get("MFA_ASYNC_COPY", "auto")
configs = [(1, 8, 1024, 64), (1, 8, 4096, 64), (2, 8, 4096, 64), (1, 8, 4096, 128), (1, 8, 8192, 64), (1, 4, 16384, 64)]
rows = []
for B, H, N, D in configs:
    gc.collect(); torch.mps.empty_cache()
    q, k, v = [torch.randn(B, H, N, D, device="mps", dtype=torch.float16) for _ in range(3)]
    for _ in range(3):
        flash_attention(q, k, v); torch.mps.synchronize()
    runs = 10
    torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(runs):
        flash_attention(q, k, v)
    torch.mps.synchronize(); mfa = (time.perf_counter() - t) / runs * 1000
    for _ in range(2):
        F.scaled_dot_product_attention(q, k, v); torch.mps.synchronize()
    torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(runs):
        F.scaled_dot_product_attention(q, k, v)
    torch.mps.synchronize(); sdpa = (time.perf_counter() - t) / runs * 1000
    flops = 4 * B * H * N * N * D
    rows.append((B, H, N, D, mfa, flops / mfa / 1e9, sdpa))
    print(f"mode={mode} B={B} H={H} N={N:5d} D={D:3d}  mfa {mfa:8.2f} ms  {flops/mfa/1e9:7.0f} GFLOPS   sdpa {sdpa:8.2f} ms", flush=True)
json.dump(rows, open(f"bench_async_{mode}.json", "w"))
