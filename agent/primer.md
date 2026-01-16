# Infemeral Agent Primer

**Generated**: 2026-01-15_20-13-26

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| ML Framework | PyTorch, Transformers, vLLM |
| Model Format | SafeTensors, Tensorizer (fast loading) |
| Quantization | AWQ (4-bit INT4) |
| Transport | gRPC + Protocol Buffers |
| Crypto | AES-256-GCM, Haar-distributed orthogonal matrices |
| Config | Pydantic Settings (env vars) |
| Deployment | RunPod Serverless |
| Testing | pytest |
| Linting | ruff |

## Architecture Pattern

**Split-Brain Zero-Trust Inference**

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLIENT (Sovereign Edge)                     │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Tokenizer │───▶│ embed_tokens│───▶│  Cloak   │───▶│   gRPC   │ │
│  └──────────┘    └───────────┘    │ (M @ x + ε)│    │  Stub    │ │
│                                    └──────────┘    └────┬─────┘ │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐         │       │
│  │  Sample  │◀───│  lm_head  │◀───│ Uncloak  │◀────────┘       │
│  └──────────┘    └───────────┘    └──────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                              │ gRPC
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SERVER (Blind Core)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │            Transformer Layers (AWQ Quantized)              │  │
│  │    Receives cloaked hidden states, never sees raw tokens   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         Encrypted KV Cache (AES-256-GCM @ Redis/Disk)      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Security Invariants**:
1. Raw tokens never leave the client
2. Cloaking uses orthogonal rotation (preserves attention dot products) + DP noise
3. Server processes only cloaked states; cannot reconstruct prompts/outputs
4. KV cache is AES-256-GCM encrypted with session keys

## Entry Points

| Context | Entry Point |
|---------|-------------|
| Client CLI | `infemeral/client.py:main()` |
| Server (RunPod) | `infemeral/server.py:handler()` |
| gRPC Service | `tensor_service.proto` → `TensorInference.Infer()` |

## 5 Critical Files for Development

1. **`infemeral/client.py`** - Client inference loop
   - `EmbeddingLayer`: Loads embed_tokens + lm_head from SafeTensors
   - `Client.generate()`: Tokenize → embed → cloak → call server → uncloak → de-embed → sample
   - `Client._call_server()`: gRPC transport with encryption

2. **`infemeral/server.py`** - Server inference handler
   - `load_model()`: Tensorizer (fast) → SafeTensors (fallback) with AWQ support
   - `forward_transformer()`: Bypasses embedding layer, feeds hidden states directly to transformer blocks
   - `handler()`: RunPod serverless entry; decrypts input, runs inference, encrypts output

3. **`infemeral/crypto.py`** - Cryptographic primitives
   - `generate_orthogonal_matrix()`: Haar-distributed via QR decomposition
   - `cloak()/uncloak()`: Orthogonal rotation + DP noise
   - `encrypt_bytes()/decrypt_bytes()`: AES-256-GCM

4. **`infemeral/config.py`** - Environment-based configuration
   - `CryptoSettings`: hidden_dim, dp_epsilon, dp_delta
   - `ClientSettings`: weights_path, server_url, model_id
   - `ServerSettings`: weights_path, tensorized_weights_path, kv_cache_dir, model_id

5. **`infemeral/tensors.py`** - Tensor serialization
   - `serialize_tensor()/deserialize_tensor()`: PyTorch ↔ bytes for gRPC
   - `pack_kv_cache()/unpack_kv_cache()`: Binary format for KV storage

## Key Data Flow

```
User Prompt
    │
    ▼
tokenizer.encode() → input_ids [1, seq_len]
    │
    ▼
embed_tokens(input_ids) → hidden [1, seq_len, 4096]
    │
    ▼
cloak(hidden, M, σ) → cloaked [1, seq_len, 4096]
    │
    ▼ (gRPC + AES-256-GCM)
forward_transformer(cloaked) → server_output [1, seq_len, 4096]
    │
    ▼ (gRPC + AES-256-GCM)
uncloak(server_output, M) → uncloaked [1, seq_len, 4096]
    │
    ▼
lm_head(uncloaked[:, -1:, :]) → logits [1, 1, vocab_size]
    │
    ▼
sample(logits) → next_token
```

## Configuration (Environment Variables)

```bash
# Crypto
INFEMERAL_CRYPTO_HIDDEN_DIM=4096
INFEMERAL_CRYPTO_DP_EPSILON=2.0
INFEMERAL_CRYPTO_DP_DELTA=1e-5

# Client
INFEMERAL_CLIENT_WEIGHTS_PATH=/workspace/weights/client_weights.safetensors
INFEMERAL_CLIENT_SERVER_URL=localhost:50051
INFEMERAL_CLIENT_MODEL_ID=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4

# Server
INFEMERAL_SERVER_TENSORIZED_WEIGHTS_PATH=/workspace/weights/model.tensors
INFEMERAL_SERVER_WEIGHTS_PATH=/workspace/weights/model.safetensors
INFEMERAL_SERVER_MODEL_ID=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
INFEMERAL_SERVER_KV_CACHE_DIR=/workspace/weights/kv
```

## Common Development Tasks

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Generate gRPC stubs (required after proto changes)
python -m grpc_tools.protoc -I. --python_out=infemeral --grpc_python_out=infemeral tensor_service.proto

# Run tests
pytest                          # All tests
pytest -m "not slow"            # Skip slow tests
pytest -m "not gpu"             # Skip GPU tests
pytest tests/test_crypto.py     # Single file

# Lint
ruff check .
ruff format .
```

## Current Development Notes

The server currently handles AWQ quantized models (Llama 3.1 8B). Recent work addressed:
- Rotary embedding dimension mismatches with transformers 4.46+ and AWQ models
- Position embeddings must be passed explicitly to attention layers
- Tensorizer integration for faster model loading (~10x vs SafeTensors)
