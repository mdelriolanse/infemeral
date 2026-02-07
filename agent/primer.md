# Infemeral Agent Primer

**Generated**: 2026-02-07_14-08-46

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| ML Framework | PyTorch, Transformers, vLLM |
| Model Format | SafeTensors, Tensorizer (fast loading) |
| Quantization | AWQ (4-bit INT4) via AutoAWQ, GPTQ (4-bit INT4) via auto-gptq |
| Transport | gRPC + Protocol Buffers |
| Crypto | AES-256-GCM, Haar-distributed orthogonal matrices, NVIB (optional) |
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
│  │            Transformer Layers (AWQ/GPTQ Quantized)         │  │
│  │    Receives cloaked hidden states, never sees raw tokens   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         Encrypted KV Cache (AES-256-GCM @ Disk/Memory)     │  │
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
   - NVIB cloaking support with auto-dimension detection

2. **`infemeral/server.py`** - Server inference handler
   - `load_model()`: Tensorizer (fast) → from_pretrained (fallback) with AWQ/GPTQ support
   - `forward_transformer()`: Bypasses embedding layer, feeds hidden states to transformer blocks
   - `apply_context_windowing()`: Attention sink + sliding window for KV cache management
   - `TensorInferenceServicer`: gRPC servicer with per-request KV cache load/save
   - `handler()`: RunPod serverless entry
   - In-memory KV cache with LRU eviction (configurable via `kv_cache_mode`)

3. **`infemeral/crypto.py`** - Cryptographic primitives
   - `generate_orthogonal_matrix()`: Haar-distributed via QR decomposition
   - `create_cloaking_context()`: Session-scoped matrix + DP sigma
   - `cloak()/uncloak()`: Orthogonal rotation + DP noise (einsum-based)
   - `encrypt_bytes()/decrypt_bytes()`: AES-256-GCM

4. **`infemeral/config.py`** - Environment-based configuration
   - `MODEL_PRESETS`: Predefined model configurations (Llama 3.1 8B, DeepSeek-R1-32B)
   - `CryptoSettings`: hidden_dim, dp_epsilon, dp_delta
   - `ClientSettings`: weights_path, server_url, model_id
   - `ServerSettings`: model_preset, weights_dir, tensorized_weights_path, kv_cache_dir, max_context_length, attention_sink_tokens, kv_cache_mode
   - `NVIBSettings`: beta, dim (auto-detect), simd_level

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
embed_tokens(input_ids) → hidden [1, seq_len, hidden_dim]
    │
    ▼
cloak(hidden, M, σ) → cloaked [1, seq_len, hidden_dim]
    │
    ▼ (gRPC + AES-256-GCM)
forward_transformer(cloaked, past_kv) → server_output [1, seq_len, hidden_dim], new_kv
    │
    ▼ (gRPC + AES-256-GCM)
uncloak(server_output, M) → uncloaked [1, seq_len, hidden_dim]
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
INFEMERAL_SERVER_MODEL_PRESET=llama-3.1-8b-awq  # or deepseek-r1-32b-gptq
INFEMERAL_SERVER_WEIGHTS_DIR=/workspace/weights/model
INFEMERAL_SERVER_TENSORIZED_WEIGHTS_PATH=/workspace/weights/model.tensors
INFEMERAL_SERVER_MODEL_ID=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
INFEMERAL_SERVER_KV_CACHE_DIR=/workspace/weights/kv
INFEMERAL_SERVER_MAX_CONTEXT_LENGTH=2048  # 4096 for DeepSeek-R1-32B on RTX 4090
INFEMERAL_SERVER_ATTENTION_SINK_TOKENS=4
INFEMERAL_SERVER_GRPC_PORT=50051
INFEMERAL_SERVER_KV_CACHE_MODE=memory  # memory/disk/hybrid
INFEMERAL_SERVER_MAX_CACHED_SESSIONS=10

# NVIB (optional privacy layer)
INFEMERAL_NVIB_DIM=0  # 0 = auto-detect from model
INFEMERAL_NVIB_BETA=100.0
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

---

## CRITICAL: GPU Memory Bottleneck (Current Focus)

### The Problem

**Attempting to run DeepSeek-R1-32B-GPTQ-INT4 on RTX 4090 (24GB VRAM) hits memory constraints:**

- **Model weights**: ~17GB VRAM (32B INT4 quantized)
- **Remaining VRAM**: ~7GB for activations, KV cache, and overhead
- **Context limit**: ~4,096 tokens (vs 32K theoretical max)
- **Risk**: OOM (Out of Memory) errors during model loading or inference

### Current Model Support

| Model | Size | VRAM Usage | Max Context (RTX 4090) | Status |
|-------|------|------------|------------------------|--------|
| Llama 3.1 8B AWQ | ~4GB | ~6GB | 2K-4K tokens | ✅ Working |
| DeepSeek-R1-32B GPTQ | ~17GB | ~17GB | ~4K tokens | ⚠️ Memory constrained |

### Configuration for DeepSeek-R1-32B on RTX 4090

**CRITICAL SETTINGS:**
```bash
# Must set these to avoid OOM
export INFEMERAL_SERVER_MODEL_PRESET=deepseek-r1-32b-gptq
export INFEMERAL_SERVER_MAX_CONTEXT_LENGTH=4096  # DO NOT exceed
export INFEMERAL_SERVER_KV_CACHE_MODE=memory  # Use memory cache
export INFEMERAL_SERVER_MAX_CACHED_SESSIONS=1  # Single session only
```

### Memory Breakdown (DeepSeek-R1-32B on RTX 4090)

```
Total VRAM: 24GB
├── Model weights: ~17GB (GPTQ INT4)
├── KV cache (4K tokens): ~2-3GB
├── Activations: ~1-2GB
└── PyTorch overhead: ~1GB
────────────────────────────
Total: ~21-23GB (tight fit!)
```

### Known Issues & Workarounds

1. **OOM during model loading**
   - **Symptom**: `RuntimeError: CUDA out of memory` when calling `load_model()`
   - **Workaround**: Ensure no other processes using GPU, clear cache: `torch.cuda.empty_cache()`
   - **Check**: `nvidia-smi` should show <2GB used before loading

2. **OOM during inference**
   - **Symptom**: OOM error after several tokens generated
   - **Cause**: KV cache growing beyond available memory
   - **Fix**: Ensure `INFEMERAL_SERVER_MAX_CONTEXT_LENGTH=4096` is set
   - **Monitor**: Watch `nvidia-smi` during inference

3. **Context window too large**
   - **Symptom**: OOM when prompt exceeds ~4K tokens
   - **Fix**: Truncate prompts or use attention windowing more aggressively
   - **Config**: Reduce `INFEMERAL_SERVER_ATTENTION_SINK_TOKENS` if needed

### GPU Utilization Issue (Separate from Memory)

**Current Status**: GPU shows 1.9% utilization (expected 40-80%)
- **Documented in**: `agent/issues/gpu-underutilization-2026-01-31_20-50-22.md`
- **Root cause**: Likely CPU-bound bottleneck (GIL, serialization)
- **Impact**: Slow inference (~1 tok/s vs expected 40-100 tok/s)
- **Status**: OPEN - needs investigation

### Recommendations

1. **For RTX 4090**: Stick with Llama 3.1 8B for now (more headroom, better performance)
2. **For DeepSeek-R1-32B**: Use A100-40GB or higher for full 32K context
3. **If using DeepSeek on RTX 4090**: 
   - Set `MAX_CONTEXT_LENGTH=4096` (mandatory)
   - Use single session (`MAX_CACHED_SESSIONS=1`)
   - Monitor VRAM usage closely
   - Consider gradient checkpointing if available

---

## RunPod Environment Setup (CRITICAL REFERENCE)

### SSH Connection

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@<pod_ip>
```

### Virtual Environment (uv)

The project uses `uv` as the package manager. The virtual environment is located at `/mnt/.venv` (persistent storage).

**Critical**: Set the environment variable before any `uv` commands:
```bash
export UV_PROJECT_ENVIRONMENT=/mnt/.venv
```

### Running Python Commands

**Option 1**: Use full path to Python (recommended for SSH commands):
```bash
/mnt/.venv/bin/python -m infemeral.server
```

**Option 2**: Use `uv run` with env var set:
```bash
export UV_PROJECT_ENVIRONMENT=/mnt/.venv
uv run -- python -m infemeral.server
```

**Option 3**: For local testing with uv on RunPod:
```bash
export UV_PROJECT_ENVIRONMENT=/mnt/.venv
cd /workspace/infemeral-src
uv run -- python -c "from infemeral.client import Client; print('OK')"
```

### Dependency Management

Check installed packages:
```bash
export UV_PROJECT_ENVIRONMENT=/mnt/.venv
uv pip list
```

Install a specific package:
```bash
export UV_PROJECT_ENVIRONMENT=/mnt/.venv
uv pip install 'transformers==4.51.3' --python /mnt/.venv/bin/python
```

**Warning**: Running `uv run` without setting `UV_PROJECT_ENVIRONMENT` may create a new venv and reinstall all dependencies, potentially upgrading packages to incompatible versions.

### Server Management

Start server (background):
```bash
ssh -p <port> root@<pod_ip> "cd /workspace/infemeral-src && /mnt/.venv/bin/python -m infemeral.server > /tmp/server.log 2>&1 &"
```

Check server logs:
```bash
ssh -p <port> root@<pod_ip> "tail -50 /tmp/server.log"
```

Kill server:
```bash
ssh -p <port> root@<pod_ip> "pkill -9 python"
```

Check GPU memory:
```bash
ssh -p <port> root@<pod_ip> "nvidia-smi"
```

Clear KV cache:
```bash
ssh -p <port> root@<pod_ip> "rm -rf /workspace/weights/kv/*"
```

### Known Issues

1. **Transformers version**: AutoAWQ is deprecated and only tested with `transformers==4.51.3`. The `uv run` command may auto-upgrade to newer versions that break AWQ imports. Always pin: `uv pip install 'transformers==4.51.3'`

2. **Client embedding loading**: Loading the 2GB `client_weights.safetensors` takes ~50-60 seconds. Use appropriate timeouts.

3. **GPU memory (Llama 8B)**: The server uses ~6GB VRAM. The client embeddings need ~2GB. Both can run on a single RTX 4090 (24GB).

4. **GPU memory (DeepSeek 32B)**: The server uses ~17GB VRAM. **CRITICAL**: Set `MAX_CONTEXT_LENGTH=4096` and `MAX_CACHED_SESSIONS=1` to avoid OOM.

### Directory Structure on RunPod

```
/workspace/
├── infemeral-src/          # Source code (synced via scp)
│   └── infemeral/
│       ├── client.py
│       ├── server.py
│       └── ...
└── weights/
    ├── client_weights.safetensors  # Client embedding weights (2.1GB)
    ├── model/                       # Full AWQ model for server (Llama)
    ├── deepseek-r1-32b/            # DeepSeek-R1-32B GPTQ model (~17GB)
    ├── tokenizer/                   # Tokenizer files
    └── kv/                          # Encrypted KV cache storage
/mnt/
└── .venv/                  # Persistent virtual environment
```

### Quick Test Workflow

```bash
# 1. Deploy updated code
./scp_to_runpod.sh <pod_ip> <port> infemeral/server.py

# 2. Restart server on RunPod
ssh -p <port> root@<pod_ip> "pkill -9 python; sleep 2 && cd /workspace/infemeral-src && /mnt/.venv/bin/python -m infemeral.server > /tmp/server.log 2>&1 &"

# 3. Wait for model load (~20s for Llama, ~60s for DeepSeek)
sleep 20

# 4. Check server started
ssh -p <port> root@<pod_ip> "tail -5 /tmp/server.log"
# Should show: "gRPC server started on port 50051"

# 5. Run client test
ssh -p <port> root@<pod_ip> "cd /workspace/infemeral-src && timeout 180 /mnt/.venv/bin/python -c \"
from infemeral.client import Client
client = Client(weights_path='/workspace/weights/client_weights.safetensors', device='cpu')
result = client.generate('Hello', max_new_tokens=10)
print(f'Result: {repr(result)}')
client.close()
\""
```

---

## Active RunPod Configuration (2026-02-07)

**Pod Specs**: 1x L40S (48 GB VRAM), 94 GB RAM, 16 vCPU
- **IP**: 203.57.40.185
- **Port**: 10105
- **SSH**: `ssh -p 10105 root@203.57.40.185`

**Current Setup**:
- Virtual environment: `/mnt/.venv` (managed with `uv`)
- Source code: `/workspace/infemeral-src` (synced via git)
- Weights: `/workspace/weights/`
  - Client: `client_weights.safetensors` (2.1GB)
  - Server (Llama): `model/` (5.4GB)
  - Server (DeepSeek): `deepseek-r1-32b/` (18GB)
  - Tokenizer: `tokenizer/` (19MB)
  - KV cache: `kv/` (persistent storage)

**Package Versions (Critical)**:
- Python: 3.11.10
- PyTorch: 2.9.1+cu128
- Transformers: 4.51.3 (pinned - newer versions break AutoAWQ)
- AutoAWQ: 0.2.7.post3 (pinned for compatibility)

**Server Management**:
```bash
# Start server
ssh -p 10105 root@203.57.40.185 "cd /workspace/infemeral-src && nohup /mnt/.venv/bin/python -m infemeral.server > /tmp/server.log 2>&1 &"

# Check logs
ssh -p 10105 root@203.57.40.185 "tail -50 /tmp/server.log"

# Kill server
ssh -p 10105 root@203.57.40.185 "pkill -f 'python -m infemeral.server'"

# Check GPU
ssh -p 10105 root@203.57.40.185 "nvidia-smi"

# Sync code
cd /home/mdelr/apps/infemeral
git add -A && git commit -m "..." && git push origin main
ssh -p 10105 root@203.57.40.185 "cd /workspace/infemeral-src && git pull origin main"
```

**Model Configuration (L40S)**:

For **Llama 3.1 8B AWQ** (current setup):
- VRAM usage: ~6GB
- Max context: 2048 tokens (can increase to 4096+)
- Headroom: 42GB available for KV cache and activations
- Status: ✅ Running smoothly

For **DeepSeek-R1-32B GPTQ** (available):
- VRAM usage: ~17GB
- Max context: Up to 16K tokens (vs 4K limit on RTX 4090)
- Headroom: 31GB for KV cache
- Status: ⚠️ Not tested yet on L40S

**To switch to DeepSeek-R1-32B**:
```bash
export INFEMERAL_SERVER_MODEL_PRESET=deepseek-r1-32b-gptq
export INFEMERAL_SERVER_WEIGHTS_DIR=/workspace/weights/deepseek-r1-32b
export INFEMERAL_SERVER_MAX_CONTEXT_LENGTH=8192
```

## Current Development Notes

**Recent Work (2026-01 to 2026-02)**:
- **Two-phase generation**: Prompt phase sends full sequence, generation phase sends only new token (relies on server KV cache)
- **Per-layer KV cache**: v2 binary format stores (key, value) tuples per transformer layer with proper shape validation
- **Context windowing**: Attention sink (first N tokens) + sliding window to bound KV cache growth
- **DynamicCache integration**: Server converts tuple-based cache to transformers `DynamicCache` for layer compatibility
- **Performance instrumentation**: `TokenTiming` and `GenerationMetrics` classes for profiling per-token latency breakdown
- **Model presets**: Support for multiple models (Llama 3.1 8B, DeepSeek-R1-32B) via `MODEL_PRESETS` configuration
- **NVIB integration**: Optional privacy layer with auto-dimension detection
- **In-memory KV cache**: LRU eviction with configurable max sessions
- **GPTQ support**: Added `auto-gptq` for DeepSeek-R1-32B model loading

**Known Considerations**:
- Rotary embeddings computed per-layer using each layer's `rotary_emb` module
- Position IDs must account for `past_len` when KV cache exists
- AWQ quantized models loaded via `AutoModelForCausalLM.from_pretrained` with `device_map`
- GPTQ models loaded via `AutoModelForCausalLM.from_pretrained` with `auto-gptq` integration
- Tensorizer path checked first for faster cold starts (~10x vs SafeTensors)
- **GPU memory is tight for DeepSeek-R1-32B on RTX 4090** - must limit context to 4096 tokens

## File Deployment

After modifying server files, deploy to RunPod using:
```bash
./scp_to_runpod.sh <pod_ip> <port> <filename> [remote_target_dir]
```

Example:
```bash
./scp_to_runpod.sh 203.57.40.146 10017 infemeral/server.py
```

---

## Related Documentation

- **GPU Underutilization Issue**: `agent/issues/gpu-underutilization-2026-01-31_20-50-22.md`
- **DeepSeek-R1-32B Plan**: `agent/plans/deepseek-r1-32b-int4-support-2026-02-01_00-40-21.md`
- **GPU Utilization Benchmark**: `agent/plans/gpu-utilization-benchmark-2026-01-31_19-37-09.md`
- **Performance Bottleneck Remediation**: `agent/plans/performance-bottleneck-remediation-2026-01-31_18-59-28.md`
