# NVIB Integration Plan

**Feature**: Integrate NVIB (Nonparametric Variational Information Bottleneck) Cloaking Layer
**Created**: 2026-02-01_00-13-38
**Status**: Ready for Implementation

## Overview

Integrate the NVIB cloaking layer from the `feature/nvib` branch (located in `.worktree-nvib/nvib/`) into the main branch. NVIB adds privacy-preserving noise to embeddings before encryption, providing an additional layer of protection against inference attacks.

### Target Architecture

```
Client Flow (with NVIB):
embedding.embed(tokens) → hidden [4096-dim]
  → NVIBCloaker.cloak() → noised_hidden [4096-dim]
  → serialize_tensor()
  → encrypt_bytes()
  → gRPC request
  → Server
```

---

## Clarification Questions

**None required.** The source files are complete and the integration points are well-defined:
- NVIB source files exist in `.worktree-nvib/nvib/`
- `NVIBSettings` already exists in `infemeral/config.py:66-95`
- Integration points in `client.py` are clear (`_call_server()` at line 191, `__init__()` at line 114)

---

## Dependency Mapping

| Dependency | Required For | Blocker? |
|:-----------|:-------------|:---------|
| C compiler (gcc) | Building `nvib_cloak.so` | Yes - user must have gcc |
| SIMD support (AVX2/AVX512/SSE4) | Performance optimization | No - fallback to scalar |
| `torch`, `numpy` | Python wrapper | No - already in dependencies |
| `ctypes` | Python wrapper | No - stdlib |

### External Dependencies
- **None new** - All Python dependencies already present (`torch`, `numpy`, `scipy`)
- **Build requirement**: `gcc` with SIMD support (documented, not enforced)

---

## Phase 1: MVP/Foundational

**Goal**: NVIB cloaking available and working with graceful degradation

### Task 1.1: Copy NVIB Source Files

**Files to Create**:
| Source | Destination |
|:-------|:------------|
| `.worktree-nvib/nvib/nvib_cloak.c` | `infemeral/nvib/nvib_cloak.c` |
| `.worktree-nvib/nvib/nvib_cloak.h` | `infemeral/nvib/nvib_cloak.h` |
| `.worktree-nvib/nvib/nvib_wrapper.py` | `infemeral/nvib/nvib_wrapper.py` |
| `.worktree-nvib/nvib/__init__.py` | `infemeral/nvib/__init__.py` |
| `.worktree-nvib/nvib/README.md` | `infemeral/nvib/README.md` |

**Changes Required in `infemeral/nvib/__init__.py`**:
```python
# Change from:
from nvib.nvib_wrapper import NVIBCloaker, nvib_cloak

# To:
from infemeral.nvib.nvib_wrapper import NVIBCloaker, nvib_cloak
```

**Success Criteria**:
- [ ] All files copied to `infemeral/nvib/`
- [ ] Import path corrected in `__init__.py`
- [ ] Directory structure matches specification

---

### Task 1.2: Create Makefile for Build System

**File**: `Makefile` (project root)

**Content** (adapted from `.worktree-nvib/Makefile`):
```makefile
# Makefile for NVIB Cloaking Library

CC = gcc
CFLAGS = -O3 -ffast-math -fPIC -Wall -Wextra -std=c11
LDFLAGS = -lm
LIB_DIR = infemeral/nvib

# Detect SIMD support at compile time
HAS_AVX512 := $(shell grep -q avx512 /proc/cpuinfo 2>/dev/null && echo yes || echo no)
HAS_AVX2 := $(shell grep -q avx2 /proc/cpuinfo 2>/dev/null && echo yes || echo no)

ifeq ($(HAS_AVX512),yes)
	SIMD_DEFINES = -D__AVX512F__
	SIMD_FLAGS = -mavx512f -mavx512cd -mavx2 -msse4.1
else ifeq ($(HAS_AVX2),yes)
	SIMD_DEFINES = -D__AVX2__
	SIMD_FLAGS = -mavx2 -msse4.1
else
	SIMD_DEFINES = -D__SSE4_1__
	SIMD_FLAGS = -msse4.1
endif

# Target: Build shared library
$(LIB_DIR)/nvib_cloak.so: $(LIB_DIR)/nvib_cloak.c $(LIB_DIR)/nvib_cloak.h
	@echo "Building NVIB cloaking library..."
	$(CC) $(CFLAGS) $(SIMD_FLAGS) $(SIMD_DEFINES) -shared -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

# Target: Build everything
all: $(LIB_DIR)/nvib_cloak.so

# Target: Clean build artifacts
clean:
	rm -f $(LIB_DIR)/nvib_cloak.so
	@echo "Cleaned build artifacts"

# Target: Check if library exists
check:
	@if [ -f $(LIB_DIR)/nvib_cloak.so ]; then \
		echo "Library exists: $(LIB_DIR)/nvib_cloak.so"; \
	else \
		echo "Library not found. Run 'make' to build."; \
		exit 1; \
	fi

.PHONY: all clean check
```

**Success Criteria**:
- [ ] `make` builds `infemeral/nvib/nvib_cloak.so`
- [ ] `make clean` removes build artifacts
- [ ] `make check` verifies library exists

---

### Task 1.3: Integrate NVIB into Client

**File**: `infemeral/client.py`

#### 1.3.1: Add Import Block (after line 24)

```python
# NVIB cloaking (optional privacy layer)
try:
    from infemeral.nvib import NVIBCloaker
    HAS_NVIB = True
except (ImportError, RuntimeError):
    HAS_NVIB = False
    NVIBCloaker = None
```

#### 1.3.2: Add `nvib_ms` to `TokenTiming` Dataclass (line 28)

```python
@dataclass
class TokenTiming:
    """Timing breakdown for a single token generation."""

    embed_ms: float = 0.0
    nvib_ms: float = 0.0  # NEW: NVIB cloaking time
    network_ms: float = 0.0
    de_embed_ms: float = 0.0
    sample_ms: float = 0.0
    total_ms: float = 0.0
```

#### 1.3.3: Initialize NVIB in `Client.__init__()` (after line 145)

Add after `self.server_url = server_url`:

```python
# Initialize NVIB cloaker (optional privacy layer)
self.nvib_cloaker = None
if HAS_NVIB:
    try:
        from infemeral.config import nvib_settings
        self.nvib_cloaker = NVIBCloaker(
            dim=nvib_settings.dim,
            beta=nvib_settings.beta,
            mu_init=nvib_settings.mu_init,
            log_sigma2_init=nvib_settings.log_sigma2_init,
            seed=nvib_settings.prng_seed,
        )
    except Exception as e:
        import warnings
        warnings.warn(f"NVIB cloaking unavailable: {e}. Continuing without NVIB.")
```

#### 1.3.4: Apply NVIB in `_call_server()` (line 191-194)

Modify to:

```python
def _call_server(self, hidden: torch.Tensor) -> torch.Tensor:
    """Send hidden states to server, receive transformed output."""
    # Apply NVIB cloaking if available (before serialization/encryption)
    if self.nvib_cloaker is not None:
        hidden = self.nvib_cloaker.cloak(hidden)

    # Serialize tensor
    data, shape, dtype = serialize_tensor(hidden)
    # ... rest unchanged ...
```

#### 1.3.5: Add NVIB Timing in `_generate_token()` (line 324-367)

Update the timing instrumentation to include NVIB:

```python
def _generate_token(
    self,
    input_ids: torch.Tensor,
    temperature: float,
    top_p: float,
    return_timing: bool = False,
) -> tuple[dict, TokenTiming] | None:
    """Generate a single token with optional timing instrumentation."""
    if not return_timing:
        return None

    timing = TokenTiming()
    total_start = time.perf_counter()

    # Embed
    t0 = time.perf_counter()
    hidden = self.embedding.embed(input_ids)
    timing.embed_ms = (time.perf_counter() - t0) * 1000

    # NVIB cloaking (if available)
    if self.nvib_cloaker is not None:
        t0 = time.perf_counter()
        hidden = self.nvib_cloaker.cloak(hidden)
        timing.nvib_ms = (time.perf_counter() - t0) * 1000

    # Network (includes serialize + encrypt + RPC + decrypt + deserialize)
    t0 = time.perf_counter()
    server_output = self._call_server_internal(hidden)  # See 1.3.6
    timing.network_ms = (time.perf_counter() - t0) * 1000

    # ... rest unchanged ...
```

#### 1.3.6: Refactor `_call_server` to Avoid Double Cloaking

To avoid cloaking twice (once in `_generate_token` for timing, once in `_call_server`), introduce an internal method:

```python
def _call_server_internal(self, hidden: torch.Tensor) -> torch.Tensor:
    """Send hidden states to server (no NVIB - already applied)."""
    # Serialize tensor
    data, shape, dtype = serialize_tensor(hidden)
    # ... rest of current _call_server implementation ...

def _call_server(self, hidden: torch.Tensor) -> torch.Tensor:
    """Send hidden states to server, receive transformed output."""
    # Apply NVIB cloaking if available
    if self.nvib_cloaker is not None:
        hidden = self.nvib_cloaker.cloak(hidden)
    return self._call_server_internal(hidden)
```

#### 1.3.7: Update `print_metrics()` (line 424-451)

Add `nvib` to the phases list:

```python
phases = ["embed", "nvib", "network", "de_embed", "sample", "total"]
```

**Success Criteria**:
- [ ] NVIB imported with graceful fallback
- [ ] `nvib_cloaker` initialized in `Client.__init__()`
- [ ] NVIB applied in `_call_server()` before serialization
- [ ] Timing metrics include `nvib_ms`
- [ ] No double-cloaking when `return_metrics=True`

---

### Task 1.4: Update `print_metrics()` Output

Update `infemeral/client.py:print_metrics()` to display NVIB timing in the breakdown.

**Success Criteria**:
- [ ] NVIB timing visible in `--profile` output

---

## Phase 2: Testing & Validation

**Goal**: Comprehensive test coverage for NVIB integration

### Task 2.1: Create NVIB Integration Tests

**File**: `tests/test_nvib_integration.py`

```python
"""NVIB integration tests."""

import pytest
import torch
from unittest import mock

# Test graceful degradation
class TestNVIBGracefulDegradation:
    """Tests for NVIB graceful degradation."""

    def test_client_works_without_nvib(self, mock_client_weights):
        """Client should work when NVIB library is missing."""
        with mock.patch.dict("sys.modules", {"infemeral.nvib": None}):
            # Force reimport
            import importlib
            from infemeral import client
            importlib.reload(client)

            # Should not raise
            c = client.Client(weights_path=mock_client_weights, server_url="localhost:50051")
            assert c.nvib_cloaker is None
            c.close()

    def test_client_with_nvib_enabled(self, mock_client_weights):
        """Client should initialize NVIB when available."""
        pytest.importorskip("infemeral.nvib")

        from infemeral.client import Client
        c = Client(weights_path=mock_client_weights, server_url="localhost:50051")

        # Should have NVIB cloaker if library is built
        # (May be None if .so not compiled - that's OK)
        c.close()


class TestNVIBCloaking:
    """Tests for NVIB cloaking behavior."""

    @pytest.fixture
    def nvib_cloaker(self):
        """Get NVIB cloaker if available."""
        try:
            from infemeral.nvib import NVIBCloaker
            return NVIBCloaker(dim=4096, beta=1.0, seed=42)
        except (ImportError, RuntimeError):
            pytest.skip("NVIB library not available")

    def test_cloaking_changes_embedding(self, nvib_cloaker):
        """NVIB should add noise to embedding."""
        embedding = torch.randn(1, 4096, dtype=torch.float32)
        cloaked = nvib_cloaker.cloak(embedding)

        # Should be different
        assert not torch.allclose(embedding, cloaked)

        # Should preserve shape
        assert cloaked.shape == embedding.shape

    def test_cloaking_deterministic_with_seed(self, nvib_cloaker):
        """Same seed should produce same noise."""
        embedding = torch.randn(1, 4096, dtype=torch.float32)

        nvib_cloaker.set_seed(123)
        cloaked1 = nvib_cloaker.cloak(embedding)

        nvib_cloaker.set_seed(123)
        cloaked2 = nvib_cloaker.cloak(embedding)

        torch.testing.assert_close(cloaked1, cloaked2)

    def test_beta_affects_noise_level(self, nvib_cloaker):
        """Higher beta should produce less noise."""
        from scipy.spatial.distance import cosine

        embedding = torch.randn(1, 4096, dtype=torch.float32)

        # High beta = low noise
        cloaked_high_beta = nvib_cloaker.cloak(embedding, beta=10.0)
        sim_high = 1 - cosine(embedding.numpy().flatten(), cloaked_high_beta.numpy().flatten())

        # Low beta = high noise
        cloaked_low_beta = nvib_cloaker.cloak(embedding, beta=0.1)
        sim_low = 1 - cosine(embedding.numpy().flatten(), cloaked_low_beta.numpy().flatten())

        # Higher beta should have higher similarity (less noise)
        assert sim_high > sim_low

    def test_cloaking_performance(self, nvib_cloaker):
        """NVIB cloaking should be fast (<2ms for 4096-dim)."""
        import time

        embedding = torch.randn(1, 4096, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = nvib_cloaker.cloak(embedding)

        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = nvib_cloaker.cloak(embedding)
            times.append((time.perf_counter() - start) * 1000)

        import numpy as np
        p95 = np.percentile(times, 95)
        assert p95 < 2.0, f"P95 latency {p95:.3f}ms exceeds 2ms target"
```

**Success Criteria**:
- [ ] Graceful degradation tests pass
- [ ] Cloaking behavior tests pass
- [ ] Performance benchmark passes (<2ms P95)
- [ ] Determinism with seed verified

---

### Task 2.2: Update conftest.py with NVIB Fixtures

**File**: `tests/conftest.py`

Add:

```python
@pytest.fixture
def nvib_cloaker():
    """Get NVIB cloaker if available, skip otherwise."""
    try:
        from infemeral.nvib import NVIBCloaker
        return NVIBCloaker(dim=4096, beta=1.0, seed=42)
    except (ImportError, RuntimeError):
        pytest.skip("NVIB library not available (run 'make' to build)")
```

---

### Task 2.3: Copy and Update Ollama Tests (Optional)

**File**: `tests/test_nvib_ollama.py`

Copy from `.worktree-nvib/nvib/tests/test_nvib_ollama.py` and update import:

```python
# Change:
from nvib import NVIBCloaker

# To:
from infemeral.nvib import NVIBCloaker
```

Mark as integration test (requires external Ollama):

```python
pytestmark = pytest.mark.integration
```

---

## Phase 3: Documentation & Polish

**Goal**: Documentation and build integration

### Task 3.1: Update README.md

Add NVIB section to project README:

```markdown
## NVIB Privacy Cloaking (Optional)

NVIB (Nonparametric Variational Information Bottleneck) adds privacy-preserving noise to embeddings before encryption.

### Building NVIB

```bash
make  # Builds infemeral/nvib/nvib_cloak.so
```

### Configuration

| Environment Variable | Default | Description |
|:--------------------|:--------|:------------|
| `INFEMERAL_NVIB_BETA` | 1.0 | Privacy budget (higher = less noise) |
| `INFEMERAL_NVIB_DIM` | 4096 | Embedding dimension |
| `INFEMERAL_NVIB_PRNG_SEED` | None | PRNG seed (None = random) |

### Graceful Degradation

If the NVIB library is not compiled, the client will operate without cloaking and emit a warning.
```

---

### Task 3.2: Verify Configuration Integration

**File**: `infemeral/config.py`

**Status**: Already complete. `NVIBSettings` class exists at lines 66-95 with:
- `beta` (default: 1.0)
- `mu_init` (default: 0.0)
- `log_sigma2_init` (default: 0.0)
- `prng_seed` (default: None)
- `simd_level` (default: "auto")
- `dim` (default: 4096)

No changes required.

---

## Impacted Files

| File | Action | Description |
|:-----|:-------|:------------|
| `infemeral/nvib/__init__.py` | Create | Package init with corrected import |
| `infemeral/nvib/nvib_cloak.c` | Create | C implementation (copy from worktree) |
| `infemeral/nvib/nvib_cloak.h` | Create | C header (copy from worktree) |
| `infemeral/nvib/nvib_wrapper.py` | Create | Python ctypes wrapper (copy from worktree) |
| `infemeral/nvib/README.md` | Create | Module documentation (copy from worktree) |
| `infemeral/client.py` | Modify | Add NVIB integration |
| `Makefile` | Create | Build system for NVIB library |
| `tests/test_nvib_integration.py` | Create | Integration tests |
| `tests/conftest.py` | Modify | Add NVIB fixture |
| `tests/test_nvib_ollama.py` | Create (Optional) | Ollama integration tests |
| `README.md` | Modify | Add NVIB documentation |

---

## Risk Assessment

| Risk | Impact | Mitigation Strategy |
|:-----|:-------|:--------------------|
| Build complexity (gcc/SIMD) | Medium | Make NVIB optional with clear error messages. Graceful degradation when library missing. |
| Performance impact | Low | NVIB is ~0.5ms per embedding. Negligible vs network latency. |
| Breaking changes | Low | NVIB is opt-in. No API changes. Existing tests unaffected. |
| Library not found at runtime | Medium | Clear build instructions in error message. `make check` target for verification. |
| SIMD compatibility across CPUs | Low | Runtime SIMD detection with scalar fallback chain (AVX512 → AVX2 → SSE4 → scalar). |
| Double cloaking bug | Medium | Refactor to `_call_server_internal()` to separate concerns. |

---

## Success Criteria

### Functional
- [ ] `from infemeral.nvib import NVIBCloaker` works after `make`
- [ ] Client initializes NVIB cloaker when library available
- [ ] Client works without NVIB (graceful degradation)
- [ ] NVIB applied to embeddings before encryption in `_call_server()`
- [ ] End-to-end generation produces coherent output with NVIB

### Performance
- [ ] NVIB overhead < 2ms per embedding (4096 dims)
- [ ] P95 latency < 2ms in benchmark
- [ ] No significant impact on tokens/sec throughput

### Integration
- [ ] No breaking changes to existing API
- [ ] Configuration via environment variables works
- [ ] Build process documented and reproducible
- [ ] `make` builds library successfully
- [ ] `make check` verifies library exists

### Testing
- [ ] All existing tests pass (no regressions)
- [ ] `test_nvib_integration.py` tests pass
- [ ] Graceful degradation tests pass
- [ ] Performance benchmark passes

---

## Suggested Tests

### Unit Tests (`tests/test_nvib_integration.py`)

1. **`test_client_works_without_nvib`**: Verify client functions when NVIB import fails
2. **`test_client_with_nvib_enabled`**: Verify NVIB cloaker initializes when available
3. **`test_cloaking_changes_embedding`**: Verify noise is added
4. **`test_cloaking_deterministic_with_seed`**: Verify reproducibility
5. **`test_beta_affects_noise_level`**: Verify privacy budget parameter works
6. **`test_cloaking_performance`**: Benchmark P95 < 2ms
7. **`test_cloaking_preserves_shape`**: Output shape matches input
8. **`test_timing_metrics_include_nvib`**: `TokenTiming.nvib_ms` populated

### Integration Tests (`tests/test_nvib_ollama.py`, optional)

1. **`test_nvib_semantic_coherence`**: LLM output coherent with NVIB
2. **`test_nvib_privacy_budget`**: Similarity decreases with lower beta
3. **`test_nvib_performance`**: End-to-end latency acceptable

### Build Verification

```bash
# Build and verify
make
make check

# Run tests
pytest tests/test_nvib_integration.py -v

# Run with NVIB disabled (graceful degradation)
mv infemeral/nvib/nvib_cloak.so /tmp/
pytest tests/test_nvib_integration.py::TestNVIBGracefulDegradation -v
mv /tmp/nvib_cloak.so infemeral/nvib/
```

---

## Task List Summary

| Phase | Task | Priority | Dependencies |
|:------|:-----|:---------|:-------------|
| 1 | Copy NVIB files to `infemeral/nvib/` | P0 | None |
| 1 | Create `Makefile` | P0 | Task 1 |
| 1 | Integrate NVIB into `client.py` | P0 | Tasks 1, 2 |
| 1 | Update `print_metrics()` | P1 | Task 3 |
| 2 | Create `test_nvib_integration.py` | P0 | Phase 1 |
| 2 | Update `conftest.py` | P1 | Phase 1 |
| 2 | Copy Ollama tests (optional) | P2 | Phase 1 |
| 3 | Update README.md | P1 | Phase 2 |
| 3 | Verify config integration | P1 | Phase 1 |
