# NVIB Cloaking Library

High-performance C-based implementation of Nonparametric Variational Information Bottleneck (NVIB) cloaking for privacy-preserving embedding transformations.

## Structure

```
nvib/
├── __init__.py           # Package initialization
├── nvib_cloak.c          # Core C implementation with SIMD optimizations
├── nvib_cloak.h          # C header file
├── nvib_wrapper.py       # Python ctypes wrapper
└── README.md             # This file
```

## Building

From the repository root:

```bash
make
```

Or build everything:

```bash
make all
```

## Usage

```python
from infemeral.nvib import NVIBCloaker
import torch

# Initialize cloaker
cloaker = NVIBCloaker(dim=4096, beta=1.0)

# Apply cloaking to embedding
embedding = torch.randn(1, 4096, dtype=torch.float32)
noised = cloaker.cloak(embedding)
```

## Testing

```bash
# Run NVIB integration tests
pytest tests/test_nvib_integration.py -v
```

## Features

- **SIMD Optimized**: AVX-512/AVX2/SSE4.1 support for sub-millisecond latency
- **Fast PRNG**: Xorshift128+ with Box-Muller transform for Gaussian sampling
- **Privacy-Preserving**: Configurable privacy budget (β parameter)
- **Python Integration**: Seamless PyTorch tensor support via ctypes

## Configuration

NVIB settings can be configured via environment variables:

- `INFEMERAL_NVIB_BETA`: Privacy budget (default: 1.0)
- `INFEMERAL_NVIB_DIM`: Embedding dimension (default: 4096)
- `INFEMERAL_NVIB_PRNG_SEED`: PRNG seed (default: random)
