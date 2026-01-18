# Infemeral Agent Primer

**Generated**: 2026-01-16_22-35-26

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| ML Framework | PyTorch, Transformers, vLLM |
| Model Format | SafeTensors, Tensorizer (fast loading) |
| Quantization | AWQ (4-bit INT4) via AutoAWQ |
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
│  │         Encrypted KV Cache (AES-256-GCM @ Disk)            │  │
│  │         Per-layer storage with context windowing           │  │
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
| Server gRPC | `infemeral/server.py:serve_grpc()` |
| Server (RunPod) | `infemeral/server.py:handler()` |
| gRPC Service | `tensor_service.proto` → `TensorInference.Infer()` |

## 5 Critical Files for Development

1. **`infemeral/client.py`** - Client inference loop
   - `EmbeddingLayer`: Loads embed_tokens + lm_head from SafeTensors (supports tied embeddings)
   - `Client.generate()`: Two-phase generation (prompt phase → generation phase)
   - `Client._call_server()`: gRPC transport with AES-256-GCM encryption
   - `TokenTiming`/`GenerationMetrics`: Performance instrumentation

2. **`infemeral/server.py`** - Server inference handler
   - `load_model()`: Tensorizer (fast) → from_pretrained (fallback) with AWQ support
   - `forward_transformer()`: Bypasses embedding layer, feeds hidden states to transformer blocks
   - `apply_context_windowing()`: Attention sink + sliding window for KV cache management
   - `TensorInferenceServicer`: gRPC servicer with per-request KV cache load/save
   - `handler()`: RunPod serverless entry

3. **`infemeral/crypto.py`** - Cryptographic primitives
   - `generate_orthogonal_matrix()`: Haar-distributed via QR decomposition
   - `create_cloaking_context()`: Session-scoped matrix + DP sigma
   - `cloak()/uncloak()`: Orthogonal rotation + DP noise (einsum-based)
   - `encrypt_bytes()/decrypt_bytes()`: AES-256-GCM

4. **`infemeral/config.py`** - Environment-based configuration
   - `CryptoSettings`: hidden_dim, dp_epsilon, dp_delta
   - `ClientSettings`: weights_path, server_url, model_id
   - `ServerSettings`: weights_dir, tensorized_weights_path, kv_cache_dir, max_context_length, attention_sink_tokens

5. **`infemeral/tensors.py`** - Tensor serialization
   - `serialize_tensor()/deserialize_tensor()`: PyTorch ↔ bytes for gRPC (handles bfloat16)
   - `pack_kv_cache_v2()/unpack_kv_cache_v2()`: Per-layer KV binary format with version header
   - `compress_tensor_data()/decompress_tensor_data()`: Optional LZ4 compression

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
forward_transformer(cloaked, past_kv) → server_output [1, seq_len, 4096], new_kv
    │
    ▼ (gRPC + AES-256-GCM)
uncloak(server_output, M) → uncloaked [1, seq_len, 4096]
    │
    ▼
lm_head(uncloaked[:, -1:, :]) → logits [1, 1, vocab_size]
    │
    ▼
sample(logits) → next_token

[Generation phase: send only last token, server uses KV cache]
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
INFEMERAL_SERVER_WEIGHTS_DIR=/workspace/weights/model
INFEMERAL_SERVER_TENSORIZED_WEIGHTS_PATH=/workspace/weights/model.tensors
INFEMERAL_SERVER_MODEL_ID=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
INFEMERAL_SERVER_KV_CACHE_DIR=/workspace/weights/kv
INFEMERAL_SERVER_MAX_CONTEXT_LENGTH=2048
INFEMERAL_SERVER_ATTENTION_SINK_TOKENS=4
INFEMERAL_SERVER_GRPC_PORT=50051
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

**Recent Work (2026-01)**:
- **Two-phase generation**: Prompt phase sends full sequence, generation phase sends only new token (relies on server KV cache)
- **Per-layer KV cache**: v2 binary format stores (key, value) tuples per transformer layer with proper shape validation
- **Context windowing**: Attention sink (first N tokens) + sliding window to bound KV cache growth
- **DynamicCache integration**: Server converts tuple-based cache to transformers `DynamicCache` for layer compatibility
- **Performance instrumentation**: `TokenTiming` and `GenerationMetrics` classes for profiling per-token latency breakdown

**Known Considerations**:
- Rotary embeddings computed per-layer using each layer's `rotary_emb` module
- Position IDs must account for `past_len` when KV cache exists
- AWQ quantized models loaded via `AutoModelForCausalLM.from_pretrained` with `device_map`
- Tensorizer path checked first for faster cold starts (~10x vs SafeTensors)
