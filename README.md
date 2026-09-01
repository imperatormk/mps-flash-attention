# MPS Flash Attention

Flash Attention for PyTorch on Apple Silicon (M1/M2/M3/M4/M5).

**O(N) memory** instead of O(N²), enabling 100K+ sequence lengths on unified memory.

## Performance

Benchmarked on Apple Silicon (M1/M2/M3/M4/M5):

| Seq Length | vs PyTorch SDPA | Notes |
|------------|-----------------|-------|
| 1024 | 1.1-2.0x faster | Crossover point |
| 2048 | 1.7-3.7x faster | Sweet spot |
| 4096 | 2.0-3.9x faster | Peak performance |
| 8192+ | 3-4x faster | SDPA often OOMs |

Average speedup: **1.8x** across all configurations.

## Installation

```bash
pip install mps-flash-attn
```

### Build from source

```bash
git clone --recursive https://github.com/mpsops/mps-flash-attention.git
cd mps-flash-attention

# Build Swift bridge
cd swift-bridge && swift build -c release && cd ..

# Install (uses the torch already in your environment)
pip install -e . --no-build-isolation
```

To build the full PyPI matrix (5 Python versions, dual torch ABIs), use
`scripts/build_all_wheels.sh`. The wheels land in `dist/`. Run
`bash tests/wheel_matrix/test_wheel_matrix.sh` to verify each wheel against
the torch versions it claims to support.

## Usage

### Basic Attention

```python
from mps_flash_attn import flash_attention

# (B, H, N, D) format
q = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)

out = flash_attention(q, k, v)
```

### Causal Masking

```python
out = flash_attention(q, k, v, is_causal=True)
```

### Sliding Window (Mistral/Llama 3.2)

```python
# Only attend to last 4096 tokens
out = flash_attention(q, k, v, is_causal=True, window_size=4096)
```

### Quantized KV Cache (2-4x memory savings)

```python
from mps_flash_attn import flash_attention_fp8, quantize_kv_fp8

# Quantize K/V to FP8
k_quant, k_scale = quantize_kv_fp8(k)
v_quant, v_scale = quantize_kv_fp8(v)

# Run attention with quantized KV
out = flash_attention_fp8(q, k_quant, v_quant, k_scale, v_scale)
```

### 100K+ Long Sequences

```python
from mps_flash_attn import flash_attention_chunked

# Process 100K tokens without OOM
q = torch.randn(1, 8, 100000, 64, device='mps', dtype=torch.float16)
k = torch.randn(1, 8, 100000, 64, device='mps', dtype=torch.float16)
v = torch.randn(1, 8, 100000, 64, device='mps', dtype=torch.float16)

out = flash_attention_chunked(q, k, v, chunk_size=8192)
```

### Drop-in SDPA Replacement

```python
from mps_flash_attn import replace_sdpa

replace_sdpa()  # Patches F.scaled_dot_product_attention

# Now all PyTorch attention uses Flash Attention on MPS
```

### torch.compile() Support

```python
from mps_flash_attn import register_custom_op

register_custom_op()

@torch.compile
def my_attention(q, k, v):
    return torch.ops.mfa.flash_attention(q, k, v, False, None, None)
```

### Training with BF16 Backward

```python
out = flash_attention(q, k, v, bf16_backward=True)  # 2x faster backward
loss = out.sum()
loss.backward()
```

### Benchmarking

```bash
# Quick benchmark
python -m mps_flash_attn.benchmark --suite quick

# Full suite with report
python -m mps_flash_attn.benchmark --suite full --output report.html
```

```python
from mps_flash_attn.benchmark import run_suite, compare_vs_sdpa

results = run_suite(seq_lengths=[1024, 2048, 4096])
compare_vs_sdpa()
```

## Features

| Feature | Supported | Notes |
|---------|:---------:|-------|
| Forward pass | yes | FP16/BF16/FP32 |
| Backward pass | yes | Full gradient support |
| Causal masking | yes | Native kernel support |
| Attention masks | yes | Boolean masks |
| Sliding window | yes | For local attention models |
| GQA/MQA | yes | Grouped-query attention |
| Quantized KV | yes | FP8, INT8, NF4 |
| Chunked attention | yes | 100K+ tokens |
| torch.compile() | yes | Custom op backend |
| Dropout | no | Not supported |

## Architecture

```
Python API (mps_flash_attn)
         |
    C++ Extension (_C_legacy.so or _C_modern.so, picked at import)
         | dlopen
    Swift Bridge (MFABridge.swift)
         |
    Metal Flash Attention (kernel generation)
         |
    Metal GPU Shaders
```

## Requirements

- macOS 14+ (Sonoma, Sequoia, or Tahoe)
- Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- PyTorch 2.5 through 2.13

### Tested torch + Python combinations

|         | torch 2.5 | 2.6 | 2.7 | 2.8 | 2.9 | 2.10 | 2.11 | 2.12 | 2.13 |
|---------|:---------:|:---:|:---:|:---:|:---:|:----:|:----:|:----:|:----:|
| py 3.10 | OK        | OK  | OK  | OK  | OK  | OK   | OK   |      |      |
| py 3.11 | OK        | OK  | OK  | OK  | OK  | OK   | OK   |      |      |
| py 3.12 | OK        | OK  | OK  | OK  | OK  | OK   | OK   |      | OK   |
| py 3.13 |           | OK  | OK  | OK  | OK  | OK   | OK   |      |      |
| py 3.14 |           |     |     |     | OK  | OK   | OK   |      |      |

Empty cells are combinations PyTorch itself does not ship wheels for, or
that have not been run through `tests/wheel_matrix` yet.

The `dependencies` field in the wheel pins `torch>=2.5,<2.14`, so `pip install
mps-flash-attn` will refuse to install on a torch outside this range rather
than installing and crashing at runtime. When a new torch minor is released we
cut a release that bumps the upper bound.

## Credits

- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) by Philip Turner
- [Flash Attention](https://arxiv.org/abs/2205.14135) paper by Tri Dao et al.

## License

MIT
