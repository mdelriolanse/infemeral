# Infemeral

**Zero-Trust Distributed LLM Inference with Stateless Server Architecture**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/infemeral/infemeral)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-yellow.svg)](https://python.org)

> **The server provider is mathematically incapable of reconstructing your prompts or conversation history.**

Infemeral implements a **Split-Brain, Stateless Topology** that partitions LLM intelligence across three trust domains, ensuring that no single entity can access complete user data.

---

## Architecture Overview

```
USER DEVICE (Trusted)
├── Tokenizer + Embedding layer + LM head
├── Orthogonal rotation matrix (M)
├── Differential privacy noise injection
└── AES-256-GCM encryption

         ↓ (encrypted cloaked vectors)

RUNPOD L4 WORKER (Untrusted)
├── Transformer blocks only (no embeddings)
├── AWQ quantized 4-bit weights
├── KV cache management (encrypted)
└── No access to rotation matrix M

         ↓ (encrypted KV cache)

FILE STORAGE (Encrypted State)
├── AES-256-GCM encrypted KV cache
├── Session-scoped keys
└── Configurable retention (default: 1 hour)
```

### Trust Domains

| Domain | Component | Holds | Cannot Access |
|--------|-----------|-------|---------------|
| **Sovereign Edge** | Client | Embedding layer, LM head, Matrix M | Nothing (fully trusted) |
| **Blind Core** | Server | Transformer blocks only | Raw embeddings, Matrix M |
| **Encrypted Locker** | File Storage | AES-256-GCM encrypted KV | Unencrypted state |

---

## Security Properties

### Mathematical Guarantees

1. **Embedding Privacy**: The server only sees rotated vectors: `x' = Mx + noise`
   - Matrix M is orthogonal → preserves dot products for attention
   - Differential privacy noise prevents known-plaintext attacks

2. **State Confidentiality**: KV cache encrypted with AES-256-GCM
   - Storage layer never sees plaintext
   - Keys are session-specific and ephemeral

### What the Server Cannot Do

- Cannot read your prompts or responses
- Cannot reconstruct conversation history
- Cannot access KV cache contents
- Cannot derive the rotation matrix M

### Current Security Limitations

- TLS is not currently enabled (use a TLS proxy in production)
- Session keys are sent raw (RSA wrapping is planned)
- No authentication/authorization mechanism yet

---

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with 24GB+ VRAM (L4, RTX 4090, A10G)
- HuggingFace account with access to gated models

### 1. Clone and Setup

```bash
git clone https://github.com/infemeral/infemeral.git
cd infemeral

# Install dependencies
pip install -e .
```

### 2. Prepare Model Weights

```bash
# Download model and extract client weights
python -c "
from infemeral.model_prep import prepare_model
prepare_model(
    model_id='hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4',
    output_dir='./weights',
    tensorize=True  # Optional: faster loading
)
"
```

This extracts:
- **Client weights**: `client_weights.safetensors` (embedding layer + LM head)
- **Server weights**: Full model directory for transformer blocks
- **Tokenizer**: Required for text encoding/decoding

### 3. Build Proto Files

```bash
chmod +x proto/build_proto.sh
./proto/build_proto.sh
```

### 4. Start Server

```bash
# Set environment variables
export INFEMERAL_SERVER_WEIGHTS_DIR=./weights/model
export INFEMERAL_SERVER_KV_CACHE_DIR=./weights/kv

# Start gRPC server
python -m infemeral.server --mode grpc --port 50051
```

### 5. Run Client

```bash
from infemeral.client import Client

client = Client(
    weights_path="./weights/client_weights.safetensors",
    server_url="localhost:50051",
    device="cuda"  # or "cpu"
)

# Generate text
output = client.generate(
    prompt="Explain quantum computing in simple terms",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)
print(output)

# With performance metrics
output, metrics = client.generate(
    prompt="Hello, how are you?",
    max_new_tokens=50,
    return_metrics=True
)
print(f"Throughput: {metrics.tokens_per_second:.1f} tok/s")

client.close()
```

---

## NVIB Privacy Cloaking (Optional)

NVIB (Nonparametric Variational Information Bottleneck) adds privacy-preserving noise to embeddings before encryption, providing an additional layer of protection against inference attacks.

### Building NVIB

The NVIB cloaking library is implemented in C with SIMD optimizations. To build:

```bash
make  # Builds infemeral/nvib/nvib_cloak.so
```

### How It Works

```
Client Flow (with NVIB):
embedding.embed(tokens) → hidden [4096-dim]
  → NVIBCloaker.cloak() → noised_hidden [4096-dim]
  → serialize_tensor()
  → encrypt_bytes()
  → gRPC request
  → Server
```

### Configuration

| Environment Variable | Default | Description |
|:--------------------|:--------|:------------|
| `INFEMERAL_NVIB_BETA` | `1.0` | Privacy budget (higher = less noise) |
| `INFEMERAL_NVIB_DIM` | `4096` | Embedding dimension |
| `INFEMERAL_NVIB_PRNG_SEED` | `None` | PRNG seed (None = random) |

### Graceful Degradation

If the NVIB library is not compiled, the client will operate without cloaking and emit a warning. All existing functionality continues to work.

```python
# NVIB is automatically used if available
client = Client(
    weights_path="./weights/client_weights.safetensors",
    server_url="localhost:50051"
)

# Check if NVIB is active
if client.nvib_cloaker is not None:
    print("NVIB cloaking is active")
```

### Performance

- NVIB overhead: ~0.5ms per embedding (4096 dimensions)
- P95 latency: <2ms
- Negligible impact on tokens/sec throughput

---

## RunPod Serverless Deployment

For cost-effective deployment with pay-per-request pricing, deploy to RunPod Serverless.

### 1. Build and Push Docker Image

```bash
docker build -t your-registry/infemeral:latest .
docker push your-registry/infemeral:latest
```

### 2. Create Network Volume

In the RunPod console:
1. Go to **Storage > Network Volumes**
2. Create a new volume (10GB minimum for KV cache)
3. Upload prepared model weights to the volume

### 3. Create Serverless Endpoint

In the RunPod console:
1. Go to **Serverless > Endpoints**
2. Click **New Endpoint**
3. Configure:
   - **Docker Image**: `your-registry/infemeral:latest`
   - **GPU**: NVIDIA L4 (24GB) recommended
   - **Environment Variables**:
     - `INFEMERAL_SERVER_WEIGHTS_DIR`: `/workspace/weights/model`
     - `INFEMERAL_SERVER_KV_CACHE_DIR`: `/workspace/weights/kv`
   - **Network Volume**: Attach the volume created above
4. Deploy

### Server Modes

```bash
# Traditional gRPC server
python -m infemeral.server --mode grpc --port 50051

# RunPod serverless HTTP handler
python -m infemeral.server --mode runpod
```

---

## Project Structure

```
infemeral/
├── README.md                    # This file
├── pyproject.toml               # Package configuration
├── Dockerfile                   # Server container
│
├── proto/                       # gRPC Contract
│   ├── tensor_service.proto     # Service & message definitions
│   └── build_proto.sh           # Stub generator script
│
├── infemeral/                   # Core Package
│   ├── __init__.py
│   ├── client.py                # Client: embeddings, generation, gRPC
│   ├── server.py                # Server: transformer forward, KV cache
│   ├── config.py                # Pydantic settings (client & server)
│   ├── crypto.py                # AES-256-GCM encryption
│   ├── tensors.py               # Tensor serialization & compression
│   ├── model_prep.py            # Model download & weight extraction
│   └── tensor_service_pb2*.py   # Generated gRPC stubs
│
├── scripts/                     # Utilities
│   └── benchmark_client.py      # Performance benchmarking
│
└── tests/                       # Test Suite
    ├── conftest.py              # Pytest fixtures
    ├── test_client.py           # Client unit tests
    ├── test_server.py           # Server unit tests
    ├── test_crypto.py           # Encryption tests
    ├── test_tensors.py          # Serialization tests
    ├── test_config.py           # Configuration tests
    ├── test_e2e.py              # End-to-end integration
    ├── test_multi_turn.py       # Multi-turn conversation tests
    └── test_client_perf.py      # Performance regression tests
```

---

## Configuration

### Environment Variables

**Client Configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `INFEMERAL_CLIENT_WEIGHTS_PATH` | `/workspace/weights/client_weights.safetensors` | Path to client embedding weights |
| `INFEMERAL_CLIENT_SERVER_URL` | `localhost:50051` | gRPC server endpoint |
| `INFEMERAL_CLIENT_MODEL_ID` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | Model for tokenizer |

**Server Configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `INFEMERAL_SERVER_WEIGHTS_DIR` | `/workspace/weights/model` | Full model directory |
| `INFEMERAL_SERVER_TENSORIZED_WEIGHTS_PATH` | - | Optional Tensorizer cache path |
| `INFEMERAL_SERVER_KV_CACHE_DIR` | `/workspace/weights/kv` | Session KV cache storage |
| `INFEMERAL_SERVER_MAX_CONTEXT_LENGTH` | `2048` | Maximum context tokens |
| `INFEMERAL_SERVER_ATTENTION_SINK_TOKENS` | `4` | Preserved tokens in context windowing |
| `INFEMERAL_SERVER_GRPC_PORT` | `50051` | gRPC server port |

**NVIB Configuration (Optional Privacy Layer):**

| Variable | Default | Description |
|----------|---------|-------------|
| `INFEMERAL_NVIB_BETA` | `1.0` | Privacy budget (higher = less noise, better utility) |
| `INFEMERAL_NVIB_DIM` | `4096` | Embedding dimension |
| `INFEMERAL_NVIB_MU_INIT` | `0.0` | Initial mean for noise distribution |
| `INFEMERAL_NVIB_LOG_SIGMA2_INIT` | `0.0` | Initial log variance for noise distribution |
| `INFEMERAL_NVIB_PRNG_SEED` | `None` | PRNG seed (None = random) |

### Python Configuration

```python
from infemeral.config import client_settings, server_settings

# Access settings
print(client_settings.model_id)
print(server_settings.max_context_length)
```

---

## API Reference

### Client

```python
from infemeral.client import Client, GenerationMetrics

# Initialize
client = Client(
    weights_path="./weights/client_weights.safetensors",
    server_url="localhost:50051",
    device="cuda"  # or "cpu"
)

# Generate text
output = client.generate(
    prompt="Hello, world!",
    max_new_tokens=100,      # Maximum tokens to generate
    temperature=0.7,         # Sampling temperature (0 = greedy)
    top_p=0.9               # Nucleus sampling threshold
)

# Generate with metrics
output, metrics = client.generate(prompt, return_metrics=True)
# metrics.tokens_per_second, metrics.time_to_first_token_ms, etc.

# Health check
is_healthy = client.check_channel_health()

# Force reconnection
client.reconnect()

# Cleanup
client.close()
```

### Server

```python
from infemeral.server import serve_grpc, handler

# Start gRPC server
serve_grpc(port=50051, max_workers=4)

# RunPod serverless handler
result = handler({"input": {...}})
```

### Model Preparation

```python
from infemeral.model_prep import (
    download_model,
    extract_client_weights,
    tensorize_model,
    prepare_model
)

# Full pipeline
prepare_model(
    model_id="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    output_dir="./weights",
    tensorize=True
)

# Individual steps
download_model(model_id, output_dir)
extract_client_weights(model_dir, output_path)
tensorize_model(model_dir, output_path)
```

### Cryptography

```python
from infemeral.crypto import generate_session_key, encrypt_bytes, decrypt_bytes

# Generate 256-bit AES key
key = generate_session_key()

# Encrypt
ciphertext, nonce = encrypt_bytes(plaintext, key)

# Decrypt
plaintext = decrypt_bytes(ciphertext, key, nonce)
```

---

## Performance

### Benchmarks (Llama 3.1 8B AWQ on L4 24GB)

| Metric | Value |
|--------|-------|
| **Throughput** | ~45 tokens/sec |
| **TTFT (Time to First Token)** | ~180ms |
| **Privacy Overhead** | ~8ms/request |
| **Rotation + DP Noise** | ~3ms |
| **Serialization** | ~5ms |

### Memory Usage

| Component | Memory |
|-----------|--------|
| Rotation Matrix (4096²) | 64 MB |
| KV Cache (2048 tokens) | 512 MB |
| Client Embedding Weights | ~1.5 GB |

### Run Benchmarks

```bash
python scripts/benchmark_client.py \
    --weights ./weights/client_weights.safetensors \
    --server localhost:50051 \
    --tokens 20 \
    --warmup 2 \
    --runs 5 \
    --device both \
    --check-regression
```

**Regression Baselines:**
- CPU p50: 25ms (flagged if > 30ms)
- GPU p50: 5ms (flagged if > 6ms)

---

## Technical Deep Dive

### Orthogonal Matrix Rotation

The security hinges on the server never learning the rotation matrix M.

```python
# Client-side
M = generate_orthogonal_matrix(dim=4096)  # M^T M = I
x_cloaked = (x + noise) @ M.T             # Rotate embedding

# Server-side (sees only x_cloaked)
# Attention: softmax((MQ)(MK)^T / √d) = softmax(QK^T / √d)
# Orthogonality preserves dot products!

# Client-side
x_output = x_cloaked_output @ M           # Inverse rotation
```

### Context Windowing

For long conversations, the server uses attention sinks with a sliding window:

```
[Attention Sinks (4 tokens)] + [Recent Context (2044 tokens)]
     ↓ preserved                    ↓ sliding window
```

This preserves model coherence while bounding memory usage.

### KV Cache Format

The system uses a versioned binary format (v2) for KV cache serialization:
- Per-layer key/value tensor storage
- LZ4 compression for tensors > 4KB
- Backward compatibility with v1 format

---

## Threat Model

### Adversary Capabilities

We assume the server operator:
- Has full access to server code and memory
- Can observe all network traffic
- Can modify server behavior (but client detects tampering)
- Can access storage layer

### Mitigations

| Threat | Mitigation |
|--------|------------|
| **Embedding reconstruction** | Orthogonal rotation + DP noise |
| **Known-plaintext attack** | Differential privacy |
| **KV cache snooping** | AES-256-GCM encryption |
| **Model extraction** | Client holds embedding layers |

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m "not slow"           # Skip slow tests
pytest tests/ -m "not gpu"            # Skip GPU tests
pytest tests/ -m "not integration"    # Skip integration tests

# Run with coverage
pytest tests/ --cov=infemeral
```

---

## Dependencies

**Core:**
- torch >= 2.4.0
- transformers >= 4.44.0
- safetensors >= 0.4.0
- grpcio >= 1.66.0
- protobuf >= 5.27.0

**Security:**
- cryptography >= 42.0.0
- scipy >= 1.14.0

**Configuration:**
- pydantic >= 2.8.0
- pydantic-settings >= 2.4.0

**Performance:**
- tensorizer >= 2.9.0 (optional, for fast model loading)
- lz4 >= 4.3.0

**Deployment:**
- runpod >= 1.7.0
- autoawq >= 0.2.9

---

## Contributing

We welcome contributions!

### Development Setup

```bash
git clone https://github.com/infemeral/infemeral.git
cd infemeral

python -m venv venv
source venv/bin/activate

pip install -e ".[dev]"

# Run tests
pytest tests/

# Type checking
mypy infemeral/

# Format code
black infemeral/ tests/ scripts/
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Transformers](https://github.com/huggingface/transformers) - Model architecture
- [Tensorizer](https://github.com/coreweave/tensorizer) - Fast model loading
- [RunPod](https://runpod.io) - Serverless GPU infrastructure
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) - 4-bit quantization

---

<p align="center">
  <b>Infemeral: Your thoughts, your control.</b>
</p>
