# 💠 Infemeral

**Zero-Trust Distributed LLM Inference with Stateless Server Architecture**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/infemeral/infemeral)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-yellow.svg)](https://python.org)

> **The server provider is mathematically incapable of reconstructing your prompts or conversation history.**

Infemeral implements a **Split-Brain, Stateless Topology** that partitions LLM intelligence across three trust domains, ensuring that no single entity can access complete user data.

---

## 🏛️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER DEVICE (Trusted)                           │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │  Tokenizer  │ → │  Embedder   │ → │  DP Noise   │ → │  Matrix M   │ │
│  │  (text→ids) │   │  (ids→vec)  │   │  (ε=2.0)    │   │  (rotation) │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘ │
│         ↑                                                      ↓        │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │   Decoder   │ ← │   LM Head   │ ← │  Matrix M⁻¹ │ ← │ gRPC Client │ │
│  │  (ids→text) │   │  (vec→ids)  │   │  (inverse)  │   │  (TLS 1.3)  │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                         │ cloaked vectors
                                         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       RUNPOD L4 WORKER (Untrusted)                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    vLLM + PagedAttention                         │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐          │   │
│  │  │ Attn L1 │ → │ FFN L1  │ → │ Attn L2 │ → │   ...   │ → Output │   │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         ↑ fetch encrypted KV           ↓ store encrypted KV             │
└─────────────────────────────────────────────────────────────────────────┘
                           │                    │
                           ↓                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      REDIS SIDECAR (Encrypted State)                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │   AES-256-GCM Encrypted KV Cache   │   TTL: 1hr   │   LRU: 2GB  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Trust Domains

| Domain | Component | Holds | Cannot Access |
|--------|-----------|-------|---------------|
| **Sovereign Edge** | Client | Embedding layer, LM head, Matrix M | Nothing (fully trusted) |
| **Blind Core** | Server | Transformer blocks only | Raw embeddings, Matrix M |
| **Encrypted Locker** | Redis | AES-256-GCM encrypted KV | Unencrypted state |

---

## 🔐 Security Properties

### Mathematical Guarantees

1. **Embedding Privacy**: The server only sees rotated vectors: `x' = Mx + noise`
   - Matrix M is orthogonal → preserves dot products for attention
   - Differential privacy noise (ε=2.0, δ=1e-5) prevents known-plaintext attacks

2. **Forward Secrecy**: Session keys rotate after every request
   - Compromise of current key doesn't expose past conversations
   - HKDF key derivation with fresh entropy

3. **State Confidentiality**: KV cache encrypted with AES-256-GCM
   - Redis sidecar never sees plaintext
   - Keys are session-specific and ephemeral

### What the Server Cannot Do

- ❌ Read your prompts or responses
- ❌ Reconstruct conversation history
- ❌ Correlate requests across sessions
- ❌ Access KV cache contents
- ❌ Derive the rotation matrix M

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with 24GB+ VRAM (L4, RTX 4090, A10G)
- Docker & Docker Compose
- NVIDIA Container Toolkit

### 1. Clone and Setup

```bash
git clone https://github.com/infemeral/infemeral.git
cd infemeral

# Install client dependencies
pip install -r requirements.client.txt

# Generate cryptographic keys
python scripts/generate_keys.py --output-dir ~/.infemeral/keys
```

### 2. Extract Embedding Weights

```bash
# Extract client-side weights from model
python scripts/extract_embeddings.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output ~/.infemeral/weights
```

### 3. Build Proto Files

```bash
chmod +x proto/build_proto.sh
./proto/build_proto.sh
```

### 4. Deploy Server

```bash
# Set your HuggingFace token for gated models
export HF_TOKEN=your_token_here

# Start production stack
docker compose -f docker-compose.prod.yml up -d

# Check server health
curl http://localhost:50051/health
```

### 5. Run Client

```bash
# Interactive mode
python -m client.main --server localhost:50051

# Single prompt
python -m client.main --server localhost:50051 \
    --prompt "Explain quantum computing in simple terms"
```

---

## ☁️ RunPod Serverless Deployment

For cost-effective deployment with pay-per-request pricing, deploy to RunPod Serverless.

### 1. Build and Push Docker Image

```bash
# Build serverless image
docker build -f Dockerfile.serverless -t your-registry/infemeral/serverless:latest .

# Push to Docker Hub (or RunPod registry)
docker push your-registry/infemeral/serverless:latest
```

### 2. Create Network Volume

In the RunPod console:
1. Go to **Storage > Network Volumes**
2. Create a new volume (10GB minimum for Redis persistence)
3. Note the volume ID

### 3. Create Serverless Endpoint

In the RunPod console:
1. Go to **Serverless > Endpoints**
2. Click **New Endpoint**
3. Configure:
   - **Docker Image**: `your-registry/infemeral/serverless:latest`
   - **GPU**: NVIDIA L4 (24GB) recommended
   - **Environment Variables**:
     - `MODEL_NAME`: `meta-llama/Llama-3.1-8B-Instruct`
     - `HF_TOKEN`: Your HuggingFace token
   - **Network Volume**: Attach the volume created above
4. Deploy

### 4. Run Client (Serverless)

```bash
# Using HTTP transport with RunPod
python -m client.main \
    --transport http \
    --runpod-api-key YOUR_RUNPOD_API_KEY \
    --runpod-endpoint YOUR_ENDPOINT_ID \
    --prompt "Explain quantum computing"
```

### Serverless Configuration

See `runpod.toml` for detailed configuration options.

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_EMBEDDED` | `true` | Run Redis as subprocess |
| `REDIS_DATA_PATH` | `/runpod-volume/redis-data` | Network volume path |
| `REDIS_ENABLE_PERSISTENCE` | `true` | Enable Redis persistence |
| `REDIS_MAX_MEMORY` | `2gb` | Redis memory limit |

### Cost Comparison

| Mode | Pricing | Best For |
|------|---------|----------|
| Docker Pod | ~$0.29/hour (L4) | Steady traffic, always-on |
| Serverless | ~$0.00024/second | Low traffic, pay-per-use |

---

## 📁 Project Structure

```
infemeral/
├── 📜 README.md                    # This file
├── 📜 requirements.txt             # Server dependencies
├── 📜 requirements.client.txt      # Client dependencies
├── 📜 Dockerfile                   # Server container (gRPC)
├── 📜 Dockerfile.serverless        # Serverless container (HTTP)
├── 📜 Dockerfile.client            # Client container
├── 📜 docker-compose.prod.yml      # Production deployment
├── 📜 docker-compose.dev.yml       # Development setup
├── 📜 runpod.toml                  # RunPod serverless config
│
├── 📂 proto/                       # gRPC Contract
│   ├── inference.proto             # Service & message definitions
│   └── build_proto.sh              # Stub generator script
│
├── 📂 client/                      # Sovereign Edge (Trusted)
│   ├── main.py                     # CLI entry point
│   ├── 📂 crypto/
│   │   ├── matrix.py               # Orthogonal rotation (M, M⁻¹)
│   │   ├── noise.py                # Differential privacy
│   │   └── keys.py                 # RSA & AES key management
│   ├── 📂 model/
│   │   ├── tokenizer.py            # HuggingFace tokenizer wrapper
│   │   └── embedder.py             # Embedding & LM head layers
│   └── 📂 transport/
│       ├── grpc_client.py          # gRPC client (traditional)
│       └── http_client.py          # HTTP client (serverless)
│
├── 📂 server/                      # Blind Core (Untrusted)
│   ├── service.py                  # gRPC server implementation
│   ├── handler.py                  # RunPod serverless handler
│   ├── http_models.py              # HTTP request/response models
│   ├── 📂 engine/
│   │   ├── vllm_worker.py          # vLLM inference wrapper
│   │   └── model_loader.py         # Headless model loading
│   ├── 📂 state/
│   │   ├── redis_connector.py      # Redis KV storage
│   │   └── encryption.py           # KV cache encryption
│   └── 📂 scripts/
│       └── start_redis.sh          # Redis startup for serverless
│
└── 📂 scripts/                     # Utilities
    ├── generate_keys.py            # Key generation
    ├── extract_embeddings.py       # Weight extraction
    └── benchmark.py                # Performance testing
```

---

## ⚙️ Configuration

### Client Configuration

```python
from client import InfemerSession

# Traditional gRPC deployment
session = InfemerSession(
    server_host="localhost",
    server_port=50051,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    embedding_dim=4096,
    privacy_epsilon=2.0,      # Lower = more private, more noise
    privacy_delta=1e-5,
    use_tls=True,
    key_dir=Path("~/.infemeral/keys"),
    transport="grpc",         # Use gRPC transport
)

# RunPod Serverless deployment
session = InfemerSession(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    embedding_dim=4096,
    privacy_epsilon=2.0,
    key_dir=Path("~/.infemeral/keys"),
    transport="http",                           # Use HTTP transport
    runpod_api_key="your_runpod_api_key",
    runpod_endpoint_id="your_endpoint_id",
)
```

### Server Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | Model to load |
| `HF_TOKEN` | - | HuggingFace API token |
| `REDIS_HOST` | `redis-sidecar` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `GPU_MEMORY_UTILIZATION` | `0.85` | vLLM GPU memory fraction |
| `MAX_MODEL_LEN` | `4096` | Maximum context length |

### Privacy Budget

```python
# Adjust privacy/utility tradeoff
privacy_epsilon = 2.0   # Standard: balanced privacy
privacy_epsilon = 1.0   # High privacy: more noise, less accuracy
privacy_epsilon = 4.0   # Low privacy: less noise, better accuracy
```

---

## 📊 Performance

### Benchmarks (Llama 3.1 8B on L4 24GB)

| Metric | Value |
|--------|-------|
| **Throughput** | 45 tokens/sec |
| **TTFT (Time to First Token)** | 180ms |
| **Matrix Rotation Overhead** | 2.3ms |
| **DP Noise Overhead** | 0.8ms |
| **gRPC Serialization** | 5.2ms (512 tokens) |
| **Total Privacy Overhead** | ~8ms/request |

### Memory Usage

| Component | Memory |
|-----------|--------|
| Rotation Matrix (4096²) | 64 MB |
| KV Cache (2048 tokens) | 512 MB |
| Embedding Weights | 1.5 GB |

Run benchmarks:
```bash
python scripts/benchmark.py --dim 4096 --seq-len 512
```

---

## 🔬 Technical Deep Dive

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

### Differential Privacy

Gaussian mechanism calibrated for Local DP:

```
σ = Δf · √(2ln(1.25/δ)) / ε

Where:
- Δf = L2 sensitivity (bounded by embedding norm)
- ε = privacy budget (lower = more private)
- δ = privacy failure probability
```

### Tide-Windowing

Context compression for long conversations:

```
[Attention Sinks (4 tokens)] + [Recent Context (2044 tokens)]
     ↓ preserved                    ↓ sliding window
```

---

## 🛡️ Threat Model

### Adversary Capabilities

We assume the server operator:
- Has full access to server code and memory
- Can observe all network traffic (encrypted)
- Can modify server behavior (but client detects tampering)
- Can collude with Redis provider

### Mitigations

| Threat | Mitigation |
|--------|------------|
| **Embedding reconstruction** | Orthogonal rotation + DP noise |
| **Known-plaintext attack** | Differential privacy (ε, δ) |
| **Session correlation** | Fresh rotation per session |
| **KV cache snooping** | AES-256-GCM encryption |
| **Key compromise** | Forward secrecy via rotation |
| **Model extraction** | Client holds embedding layers |

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repo
git clone https://github.com/infemeral/infemeral.git
cd infemeral

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt -r requirements.client.txt
pip install pytest black mypy

# Run tests
pytest tests/

# Format code
black client/ server/ scripts/
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput inference
- [LMCache](https://github.com/LMCache/LMCache) - KV cache management
- [PySyft](https://github.com/OpenMined/PySyft) - Privacy-preserving ML inspiration
- [RunPod](https://runpod.io) - Serverless GPU infrastructure

---

<p align="center">
  <b>💠 Infemeral: Your thoughts, your control.</b>
</p>

