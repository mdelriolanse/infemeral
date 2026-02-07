# DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4 Model Support Plan

**Feature**: Add support for int4 quantized DeepSeek-R1-32B with NVIB integration
**Created**: 2026-02-01_00-40-21
**Status**: Ready for Implementation

## Overview

Extend Infemeral to support the GPTQ-INT4 quantized DeepSeek-R1-Distill-Qwen-32B model from HuggingFace. This includes downloading to RunPod network volume, updating configuration to support model selection, and validating NVIB cloaking compatibility with the new model's architecture.

### Target Model
- **HuggingFace ID**: `dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4`
- **Base Architecture**: Qwen2.5-32B (dense transformer)
- **Quantization**: GPTQ INT4 (W4A16 - 4-bit weights, 16-bit activations)
- **Hidden Dimension**: 5120 (Qwen2.5-32B standard)
- **Layers**: 64 transformer blocks
- **Context Length**: 32,768 tokens (model max), **4,096 tokens on RTX 4090**
- **License**: MIT

### Hardware Note (RTX 4090 - 24GB VRAM)
The 32B INT4 model uses ~17GB for weights, leaving ~7GB for activations and KV cache. This constrains:
- **Max context**: ~4,096 tokens (vs 32K theoretical max)
- **Batch size**: 1 (single request only)
- **Recommendation**: Set `INFEMERAL_SERVER_MAX_CONTEXT_LENGTH=4096`

### Key Differences from Current Model

| Aspect | Current (Llama 3.1 8B AWQ) | New (DeepSeek-R1-32B GPTQ) |
|:-------|:---------------------------|:---------------------------|
| Architecture | Llama-style | Qwen-style |
| Hidden dim | 4096 | 5120 |
| Layers | 32 | 64 |
| Quantization | AWQ INT4 | GPTQ INT4 |
| Size on disk | ~4GB | ~17GB |

---

## Clarification Questions

**Resolved via HuggingFace analysis:**

1. **GPTQ vs AWQ compatibility**: The current codebase uses AWQ quantization. GPTQ models require different loading - `transformers` library supports both, but loading paths differ slightly.
   - **Answer**: `AutoModelForCausalLM.from_pretrained()` handles GPTQ natively via `auto-gptq` integration. Verified compatible.

2. **Hidden dimension change (4096 → 5120)**: NVIB cloaker is initialized with `dim=4096` by default.
   - **Impact**: NVIB must dynamically detect hidden dim from model, or be configurable per-model.

3. **RunPod network volume location**: Where should the model be stored?
   - **Answer**: Use `/workspace/weights/deepseek-r1-32b/` directory structure, mirroring existing model layout.

---

## Dependency Mapping

| Dependency | Required For | Blocker? |
|:-----------|:-------------|:---------|
| `auto-gptq` library | Loading GPTQ models | Yes - must install |
| `optimum` library | GPTQ integration | No - optional, improves loading |
| RunPod network volume space (~20GB) | Storing model weights | Yes - verify space |
| NVIB C library recompilation | None needed | No - dim is runtime config |
| HuggingFace token (if gated) | Model download | No - MIT license, public |

### External Dependencies
- **auto-gptq**: `pip install auto-gptq>=0.6.0` (required for GPTQ quantization)
- **optimum**: `pip install optimum` (optional, for accelerated loading)

---

## Phase 1: MVP/Foundational

**Goal**: Model downloadable and loadable on RunPod with basic inference working

### Task 1.1: Install GPTQ Dependencies

**File**: `requirements.txt` or `pyproject.toml`

**Changes**:
```txt
# Add to requirements
auto-gptq>=0.6.0
optimum>=1.16.0  # Optional but recommended
```

**Verification**:
```bash
pip install auto-gptq>=0.6.0 optimum>=1.16.0
python -c "from auto_gptq import AutoGPTQForCausalLM; print('GPTQ OK')"
```

**Success Criteria**:
- [ ] `auto-gptq` imports successfully
- [ ] No version conflicts with existing dependencies

---

### Task 1.2: Download Model to RunPod Network Volume

**Script**: `scripts/download_deepseek_r1.sh`

**Content**:
```bash
#!/bin/bash
# Download DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4 to RunPod network volume

MODEL_ID="dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4"
OUTPUT_DIR="/workspace/weights/deepseek-r1-32b"

echo "Downloading $MODEL_ID to $OUTPUT_DIR..."

# Ensure directory exists
mkdir -p "$OUTPUT_DIR"

# Download using huggingface-cli (fastest method)
huggingface-cli download "$MODEL_ID" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

# Verify download
if [ -f "$OUTPUT_DIR/config.json" ]; then
    echo "Download complete!"
    ls -lh "$OUTPUT_DIR"
else
    echo "ERROR: Download failed - config.json not found"
    exit 1
fi
```

**Alternative Python Script**: `scripts/download_deepseek_r1.py`

```python
#!/usr/bin/env python3
"""Download DeepSeek-R1-32B-GPTQ-INT4 to network volume."""

from huggingface_hub import snapshot_download
from pathlib import Path

MODEL_ID = "dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4"
OUTPUT_DIR = Path("/workspace/weights/deepseek-r1-32b")

print(f"Downloading {MODEL_ID} to {OUTPUT_DIR}...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(OUTPUT_DIR),
    local_dir_use_symlinks=False,
)

print(f"Download complete! Files in {OUTPUT_DIR}:")
for f in OUTPUT_DIR.iterdir():
    print(f"  {f.name}: {f.stat().st_size / 1e9:.2f} GB" if f.is_file() else f"  {f.name}/")
```

**Success Criteria**:
- [ ] Model files present in `/workspace/weights/deepseek-r1-32b/`
- [ ] `config.json`, `*.safetensors` files exist
- [ ] Total size ~17GB

---

### Task 1.3: Update Configuration for Model Selection

**File**: `infemeral/config.py`

**Changes**:

Add model presets support:

```python
from typing import Literal

# Model presets with architecture-specific defaults
MODEL_PRESETS = {
    "llama-3.1-8b-awq": {
        "model_id": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "hidden_dim": 4096,
        "num_layers": 32,
        "architecture": "llama",
    },
    "deepseek-r1-32b-gptq": {
        "model_id": "dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4",
        "hidden_dim": 5120,
        "num_layers": 64,
        "architecture": "qwen",
    },
}

class ServerSettings(BaseSettings):
    # ... existing fields ...

    model_preset: str = Field(
        default="llama-3.1-8b-awq",
        description="Model preset name (llama-3.1-8b-awq, deepseek-r1-32b-gptq)",
    )

    @property
    def model_config_preset(self) -> dict:
        """Get preset configuration for the selected model."""
        return MODEL_PRESETS.get(self.model_preset, MODEL_PRESETS["llama-3.1-8b-awq"])
```

**Update** `NVIBSettings` to support dynamic dimension:

```python
class NVIBSettings(BaseSettings):
    # ... existing fields ...

    dim: int = Field(
        default=0,  # 0 = auto-detect from model
        description="Embedding dimension (0 = auto-detect from model hidden_size)",
    )
```

**Success Criteria**:
- [ ] `MODEL_PRESETS` dict contains both models
- [ ] `server_settings.model_preset` selectable via env var
- [ ] `INFEMERAL_NVIB_DIM=0` triggers auto-detection

---

### Task 1.4: Update run_inference.sh for Model Selection

**File**: `run_inference.sh`

**Changes**:
```bash
#!/bin/bash
# Run inference on RunPod from local machine
# Usage: ./run_inference.sh "Your prompt here" [max_tokens] [model]

POD_IP="203.57.40.175"
POD_PORT="10271"
PROMPT="${1:-Hello, how are you?}"
MAX_TOKENS="${2:-20}"
MODEL="${3:-llama}"  # "llama" or "deepseek"

# Set model-specific paths
if [ "$MODEL" = "deepseek" ]; then
    WEIGHTS_PATH="/workspace/weights/deepseek-r1-32b"
    CLIENT_WEIGHTS="/workspace/weights/deepseek-r1-32b-client/client_weights.safetensors"
    MODEL_ID="dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4"
else
    WEIGHTS_PATH="/workspace/weights/model"
    CLIENT_WEIGHTS="/workspace/weights/client_weights.safetensors"
    MODEL_ID="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
fi

ssh -o StrictHostKeyChecking=no -p "$POD_PORT" root@"$POD_IP" "
cd /workspace/infemeral-src
export INFEMERAL_SERVER_WEIGHTS_DIR=$WEIGHTS_PATH
export INFEMERAL_SERVER_MODEL_ID=$MODEL_ID
export INFEMERAL_CLIENT_WEIGHTS_PATH=$CLIENT_WEIGHTS
export INFEMERAL_CLIENT_MODEL_ID=$MODEL_ID
/mnt/.venv/bin/python -c \"
from infemeral.client import Client

client = Client(
    weights_path='$CLIENT_WEIGHTS',
    server_url='localhost:50051',
    device='cuda'
)

result = client.generate('$PROMPT', max_new_tokens=$MAX_TOKENS)
print(result)
client.close()
\"
"
```

**Success Criteria**:
- [ ] `./run_inference.sh "Hello" 20 llama` uses Llama model
- [ ] `./run_inference.sh "Hello" 20 deepseek` uses DeepSeek model
- [ ] Environment variables correctly propagate

---

### Task 1.5: Extract Client Weights for DeepSeek

**File**: `scripts/extract_deepseek_client_weights.py`

**Content**:
```python
#!/usr/bin/env python3
"""Extract client weights for DeepSeek-R1-32B."""

from infemeral.model_prep import extract_client_weights
from pathlib import Path

MODEL_DIR = Path("/workspace/weights/deepseek-r1-32b")
OUTPUT_DIR = Path("/workspace/weights/deepseek-r1-32b-client")

if __name__ == "__main__":
    extract_client_weights(
        model_id="dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4",
        model_dir=MODEL_DIR,
        output_dir=str(OUTPUT_DIR),
        device="cuda",
    )
```

**Alternatively**, use existing `model_prep.py`:
```bash
python -m infemeral.model_prep \
    --model-id "dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4" \
    --output-dir /workspace/weights/deepseek-r1-32b-client \
    --client-only
```

**Success Criteria**:
- [ ] `client_weights.safetensors` exists in output dir
- [ ] File contains `embed_tokens.weight` with shape `[vocab_size, 5120]`
- [ ] Tokenizer saved in `tokenizer/` subdirectory

---

### Task 1.6: Update NVIB for Dynamic Dimension Detection

**File**: `infemeral/client.py`

**Changes in `Client.__init__()`**:

```python
# Initialize NVIB cloaker (optional privacy layer)
self.nvib_cloaker = None
if HAS_NVIB:
    try:
        from infemeral.config import nvib_settings

        # Auto-detect dimension from embedding layer if dim=0
        nvib_dim = nvib_settings.dim
        if nvib_dim == 0:
            # Get hidden_size from embedding layer
            nvib_dim = self.embedding.embed_tokens.weight.shape[1]

        self.nvib_cloaker = NVIBCloaker(
            dim=nvib_dim,
            beta=nvib_settings.beta,
            mu_init=nvib_settings.mu_init,
            log_sigma2_init=nvib_settings.log_sigma2_init,
            seed=nvib_settings.prng_seed,
        )
    except Exception as e:
        import warnings
        warnings.warn(f"NVIB cloaking unavailable: {e}. Continuing without NVIB.")
```

**Success Criteria**:
- [ ] NVIB auto-detects dimension from model
- [ ] Works with both 4096 (Llama) and 5120 (DeepSeek)
- [ ] Explicit `INFEMERAL_NVIB_DIM` overrides auto-detection

---

## Phase 2: Testing & Validation

**Goal**: Verify model works correctly with NVIB integration

### Task 2.1: Create DeepSeek Integration Test

**File**: `tests/test_deepseek_integration.py`

```python
"""DeepSeek-R1-32B integration tests."""

import os
import pytest
import torch

# Skip if model not available
DEEPSEEK_WEIGHTS = "/workspace/weights/deepseek-r1-32b"
DEEPSEEK_CLIENT_WEIGHTS = "/workspace/weights/deepseek-r1-32b-client/client_weights.safetensors"

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEEPSEEK_WEIGHTS),
    reason="DeepSeek model not downloaded"
)


class TestDeepSeekModelLoading:
    """Tests for DeepSeek model loading."""

    def test_server_loads_gptq_model(self):
        """Server should load GPTQ model without errors."""
        os.environ["INFEMERAL_SERVER_WEIGHTS_DIR"] = DEEPSEEK_WEIGHTS
        os.environ["INFEMERAL_SERVER_MODEL_ID"] = "dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4"

        from infemeral.server import load_model
        model = load_model()

        assert model is not None
        # Check architecture
        assert hasattr(model, "model")
        assert hasattr(model.model, "layers")

    def test_client_loads_deepseek_embeddings(self):
        """Client should load DeepSeek embedding weights."""
        from infemeral.client import EmbeddingLayer

        embedding = EmbeddingLayer(DEEPSEEK_CLIENT_WEIGHTS, device="cuda")

        # Verify hidden dimension
        assert embedding.embed_tokens.weight.shape[1] == 5120


class TestDeepSeekWithNVIB:
    """Tests for DeepSeek + NVIB integration."""

    def test_nvib_auto_dimension_detection(self):
        """NVIB should auto-detect 5120 dimension for DeepSeek."""
        os.environ["INFEMERAL_NVIB_DIM"] = "0"  # Auto-detect

        from infemeral.client import Client
        client = Client(
            weights_path=DEEPSEEK_CLIENT_WEIGHTS,
            server_url="localhost:50051",
        )

        if client.nvib_cloaker is not None:
            assert client.nvib_cloaker.dim == 5120

        client.close()

    def test_nvib_cloaking_5120_dim(self):
        """NVIB should work with 5120-dimension embeddings."""
        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=5120, beta=100.0, seed=42)

        embedding = torch.randn(1, 5120, dtype=torch.float32)
        cloaked = cloaker.cloak(embedding)

        assert cloaked.shape == embedding.shape
        assert not torch.allclose(cloaked, embedding)

    def test_end_to_end_inference_deepseek(self):
        """Full inference with DeepSeek + NVIB should produce output."""
        pytest.importorskip("infemeral.nvib")

        os.environ["INFEMERAL_SERVER_WEIGHTS_DIR"] = DEEPSEEK_WEIGHTS
        os.environ["INFEMERAL_NVIB_DIM"] = "0"

        from infemeral.client import Client

        client = Client(
            weights_path=DEEPSEEK_CLIENT_WEIGHTS,
            server_url="localhost:50051",
        )

        # Skip if no server running
        if not client.check_channel_health():
            pytest.skip("Server not running")

        result = client.generate("<think>\nWhat is 2+2?", max_new_tokens=20)

        assert result is not None
        assert len(result) > 0

        client.close()
```

**Success Criteria**:
- [ ] Model loading tests pass
- [ ] NVIB dimension auto-detection works
- [ ] End-to-end inference produces coherent output

---

### Task 2.2: Update conftest.py with DeepSeek Fixtures

**File**: `tests/conftest.py`

**Add**:
```python
@pytest.fixture
def deepseek_client_weights(tmp_path):
    """Mock DeepSeek client weights for testing."""
    import torch
    from safetensors.torch import save_file

    # Create mock weights with DeepSeek dimensions
    hidden_size = 5120
    vocab_size = 152064  # Qwen vocab size

    weights = {
        "embed_tokens.weight": torch.randn(vocab_size, hidden_size, dtype=torch.float16),
    }

    weights_path = tmp_path / "deepseek_client_weights.safetensors"
    save_file(weights, weights_path, metadata={"tied_embeddings": "true"})

    return str(weights_path)


@pytest.fixture
def nvib_cloaker_5120():
    """Get NVIB cloaker with 5120 dimension."""
    try:
        from infemeral.nvib import NVIBCloaker
        return NVIBCloaker(dim=5120, beta=100.0, seed=42)
    except (ImportError, RuntimeError):
        pytest.skip("NVIB not available")
```

---

### Task 2.3: Create NVIB Multi-Model Test

**File**: `tests/test_nvib_multi_model.py`

```python
"""Test NVIB compatibility across different model dimensions."""

import pytest
import torch


class TestNVIBMultiModel:
    """Tests for NVIB across model architectures."""

    @pytest.fixture(params=[4096, 5120, 8192])
    def dim(self, request):
        return request.param

    def test_nvib_various_dimensions(self, dim):
        """NVIB should work with common model dimensions."""
        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=dim, beta=100.0, seed=42)
        embedding = torch.randn(1, dim, dtype=torch.float32)

        cloaked = cloaker.cloak(embedding)

        assert cloaked.shape == (1, dim)
        assert not torch.allclose(cloaked, embedding)

    def test_nvib_batch_processing(self):
        """NVIB should handle batch embeddings correctly."""
        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=5120, beta=100.0, seed=42)

        # Batch of 8 embeddings
        batch = torch.randn(8, 5120, dtype=torch.float32)
        cloaked = cloaker.cloak(batch)

        assert cloaked.shape == (8, 5120)
```

---

## Phase 3: Documentation & Polish

**Goal**: Complete documentation and production readiness

### Task 3.1: Update SYSTEM_RUNDOWN.md

**File**: `SYSTEM_RUNDOWN.md`

**Add section**:
```markdown
## Supported Models

### Llama 3.1 8B (Default)
- **Model ID**: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`
- **Quantization**: AWQ INT4
- **Hidden Dim**: 4096
- **Layers**: 32

### DeepSeek-R1-32B (Optional)
- **Model ID**: `dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4`
- **Quantization**: GPTQ INT4
- **Hidden Dim**: 5120
- **Layers**: 64
- **Context**: 32K tokens

#### Switching Models

Set environment variables:
```bash
export INFEMERAL_SERVER_MODEL_PRESET=deepseek-r1-32b-gptq
export INFEMERAL_SERVER_WEIGHTS_DIR=/workspace/weights/deepseek-r1-32b
export INFEMERAL_CLIENT_WEIGHTS_PATH=/workspace/weights/deepseek-r1-32b-client/client_weights.safetensors
export INFEMERAL_CLIENT_MODEL_ID=dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4
```
```

---

### Task 3.2: Create Model Download Instructions

**File**: `docs/models/deepseek-r1-32b.md` (or section in README)

```markdown
# DeepSeek-R1-32B Setup Guide

## Prerequisites
- GPU with >= 24GB VRAM (RTX 4090 or A100)
- ~20GB network volume space
- `auto-gptq` library installed

> **RTX 4090 Users**: The 32B model fits but leaves limited headroom. Context is capped at ~4K tokens. For full 32K context, use A100-40GB or higher.

## Installation

1. Install dependencies:
```bash
pip install auto-gptq>=0.6.0
```

2. Download model:
```bash
./scripts/download_deepseek_r1.sh
```

3. Extract client weights:
```bash
python -m infemeral.model_prep \
    --model-id "dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4" \
    --output-dir /workspace/weights/deepseek-r1-32b-client \
    --client-only
```

4. Start server with DeepSeek:
```bash
export INFEMERAL_SERVER_WEIGHTS_DIR=/workspace/weights/deepseek-r1-32b
export INFEMERAL_SERVER_MAX_CONTEXT_LENGTH=4096  # Required for RTX 4090
python -m infemeral.server --mode grpc
```

5. Test inference:
```bash
./run_inference.sh "<think>\nWhat is 2+2?" 50 deepseek
```

## NVIB Configuration

NVIB auto-detects the 5120 dimension. To explicitly set:
```bash
export INFEMERAL_NVIB_DIM=5120
```

## Known Differences

1. **No system prompt**: DeepSeek-R1 should not use system prompts
2. **Reasoning format**: Prefix prompts with `<think>\n` for best results
3. **Temperature**: Use 0.5-0.7 for stable output
```

---

## Impacted Files

| File | Action | Description |
|:-----|:-------|:------------|
| `requirements.txt` | Modify | Add `auto-gptq>=0.6.0` |
| `infemeral/config.py` | Modify | Add `MODEL_PRESETS`, update `NVIBSettings.dim` default |
| `infemeral/client.py` | Modify | Add NVIB dimension auto-detection |
| `run_inference.sh` | Modify | Add model selection flag |
| `scripts/download_deepseek_r1.sh` | Create | Model download script |
| `scripts/download_deepseek_r1.py` | Create | Alternative Python download script |
| `tests/test_deepseek_integration.py` | Create | DeepSeek-specific tests |
| `tests/test_nvib_multi_model.py` | Create | Multi-dimension NVIB tests |
| `tests/conftest.py` | Modify | Add DeepSeek fixtures |
| `SYSTEM_RUNDOWN.md` | Modify | Document supported models |
| `docs/models/deepseek-r1-32b.md` | Create | Setup guide |

---

## Risk Assessment

| Risk | Impact | Mitigation Strategy |
|:-----|:-------|:--------------------|
| GPTQ loading incompatibility | High | Test with `auto-gptq` before deployment. Fallback to vLLM if needed. |
| Memory pressure (32B model on RTX 4090) | High | Limit context to 4096 tokens. Set `INFEMERAL_SERVER_MAX_CONTEXT_LENGTH=4096`. Single request only. |
| Hidden dim mismatch breaks NVIB | Medium | Implement auto-detection. Add explicit validation at startup. |
| Qwen tokenizer differences | Low | Use model-specific tokenizer (already handled by `AutoTokenizer`). |
| NVIB performance at 5120 dim | Low | 5120 is only ~25% larger than 4096. C library handles any dimension. |
| Client weights extraction fails | Medium | Test `model_prep.py` with DeepSeek architecture. May need Qwen-specific handling. |

---

## Success Criteria

### Functional
- [ ] Model downloads successfully to `/workspace/weights/deepseek-r1-32b/`
- [ ] `auto-gptq` loads the GPTQ model without errors
- [ ] Client weights extracted with correct shapes (`embed_tokens: [vocab, 5120]`)
- [ ] NVIB initializes with auto-detected 5120 dimension
- [ ] End-to-end inference produces coherent DeepSeek-R1 output
- [ ] `run_inference.sh "prompt" 50 deepseek` works

### Performance
- [ ] Model loads in <60s (cold start)
- [ ] NVIB overhead <3ms per embedding (5120 dim)
- [ ] Memory usage fits in 24GB VRAM (RTX 4090)
- [ ] Context limited to 4096 tokens (memory constraint)

### Integration
- [ ] Switching between Llama and DeepSeek via env vars works
- [ ] No breaking changes to existing Llama workflow
- [ ] All existing tests pass
- [ ] New DeepSeek tests pass

---

## Suggested Tests

### Unit Tests

1. **`test_gptq_model_loads`**: Verify GPTQ model loads with `auto-gptq`
2. **`test_deepseek_hidden_dim`**: Verify hidden_size is 5120
3. **`test_nvib_auto_dimension_5120`**: Verify NVIB detects 5120 from model
4. **`test_nvib_5120_performance`**: Benchmark NVIB at 5120 dim (<3ms)
5. **`test_client_weights_extraction`**: Verify embed_tokens shape correct
6. **`test_qwen_tokenizer_loads`**: Verify tokenizer for DeepSeek works

### Integration Tests

1. **`test_deepseek_inference_coherent`**: Generate text, verify not garbage
2. **`test_deepseek_with_nvib_e2e`**: Full pipeline with NVIB enabled
3. **`test_model_switching`**: Switch Llama↔DeepSeek via env vars
4. **`test_deepseek_reasoning_format`**: Verify `<think>` tag handling

### Build/Deployment Verification

```bash
# On RunPod:

# 1. Download model
./scripts/download_deepseek_r1.sh

# 2. Extract client weights
python -m infemeral.model_prep \
    --model-id "dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4" \
    --output-dir /workspace/weights/deepseek-r1-32b-client \
    --client-only

# 3. Start server
export INFEMERAL_SERVER_WEIGHTS_DIR=/workspace/weights/deepseek-r1-32b
export INFEMERAL_SERVER_MODEL_ID=dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4
python -m infemeral.server --mode grpc &

# 4. Test inference (from local machine)
./run_inference.sh "<think>\nExplain why the sky is blue." 100 deepseek

# 5. Verify NVIB is active
# Should see "NVIB cloaking enabled with dim=5120" in logs

# 6. Run test suite
pytest tests/test_deepseek_integration.py -v
pytest tests/test_nvib_multi_model.py -v
```

---

## Task List Summary

| Phase | Task | Priority | Dependencies |
|:------|:-----|:---------|:-------------|
| 1 | Install GPTQ dependencies | P0 | None |
| 1 | Download model to network volume | P0 | Task 1.1 |
| 1 | Update config for model selection | P0 | None |
| 1 | Update run_inference.sh | P1 | Task 1.3 |
| 1 | Extract client weights | P0 | Task 1.2 |
| 1 | Update NVIB dimension auto-detection | P0 | Task 1.5 |
| 2 | Create DeepSeek integration tests | P0 | Phase 1 |
| 2 | Update conftest.py with fixtures | P1 | Phase 1 |
| 2 | Create multi-model NVIB tests | P1 | Phase 1 |
| 3 | Update SYSTEM_RUNDOWN.md | P1 | Phase 2 |
| 3 | Create setup guide documentation | P2 | Phase 2 |
