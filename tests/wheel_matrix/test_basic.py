from mps_flash_attn import flash_attention
import torch

q = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)

out = flash_attention(q, k, v)
print("OK", out.shape, out.dtype, out.device)
