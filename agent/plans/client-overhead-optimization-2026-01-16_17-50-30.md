# Client-Side Performance Optimization Plan

**Created:** 2026-01-16
**Updated:** 2026-01-16 (clarifications resolved)
**Scope:** Reduce computational overhead in client-side inference pipeline
**Depends On:** Phase 1 (gRPC) and Phase 2 (KV Cache) - both complete
**Target:** CUDA-enabled clients with CPU fallback
**Priority:** Minimize time-to-first-token (TTFT)

---

## Executive Summary

The client performs several compute-intensive operations per token:

| Operation | FLOPs | CPU Time | GPU Time | Speedup |
|-----------|-------|----------|----------|---------|
| Embedding lookup | ~0 | <0.1ms | <0.1ms | - |
| Cloak (orthogonal rotate) | 16.7M | ~3ms | ~0.1ms | 30× |
| Encrypt (AES-GCM) | - | ~0.5ms | CPU-only | - |
| Uncloak (orthogonal rotate) | 16.7M | ~3ms | ~0.1ms | 30× |
| **De-embed (lm_head)** | **524M** | **~15ms** | **~0.5ms** | **30×** |
| Sample (softmax + multinomial) | 0.3M | ~1ms | ~0.2ms | 5× |
| **Total compute** | **558M** | **~23ms** | **~1ms** | **~20×** |

**Critical bottleneck:** De-embedding via `lm_head` is a **128k × 4096 matmul** = 524M FLOPs per token.

**Key insight:** GPU acceleration yields ~20ms savings per token. Network RTT (10-50ms) then becomes the dominant factor.

---

## Clarifications Resolved

| Question | Answer | Impact on Plan |
|----------|--------|----------------|
| Target hardware | CUDA + CPU fallback | Add GPU detection, keep tensors on device |
| Latency priority | **Time-to-first-token** | Remove batched de-embed (increases TTFT) |
| Memory budget | 200 MB acceptable | Deprioritize disk offload, blockwise matrix |
| Tied embeddings | Unknown at runtime | Detect in `model_prep.py`, skip saving duplicate |

---

## GPU vs CPU: Detailed Analysis

### Per-Token Breakdown (Single Token Generation)

```
                         CPU (M3/i9)          GPU (RTX 3060)
                         ───────────          ──────────────
Embed lookup             <0.1ms               <0.1ms
Cloak (4096² matmul)     3ms                  0.1ms + 0.3ms transfer
Serialize                0.5ms                0.5ms (CPU)
Encrypt (AES-GCM)        0.5ms                0.5ms (CPU)
─── Network RTT ───      10-50ms              10-50ms
Decrypt                  0.5ms                0.5ms (CPU)
Deserialize              0.3ms                0.3ms + 0.2ms transfer
Uncloak                  3ms                  0.1ms
De-embed (128k×4096)     15ms                 0.5ms
Sample                   1ms                  0.2ms
                         ───────────          ──────────────
Total (excl. network)    ~24ms                ~3ms
```

**Net GPU benefit: ~20ms/token saved**

### Why Not 30× Speedup End-to-End?

1. **AES encryption stays on CPU** - CUDA AES exists but adds complexity
2. **Data transfer overhead** - Small tensors (32KB) don't amortize PCIe latency
3. **Network dominates** - RTT is 10-50ms regardless of device

### Recommendation

CUDA support is straightforward (PyTorch handles device placement). Worth implementing.

```python
# Already works - just ensure tensors stay on device
device = "cuda" if torch.cuda.is_available() else "cpu"
self.embedding = EmbeddingLayer(weights_path, device)  # Weights on GPU
```

The only code change needed: ensure cloaking matrix is also on GPU.

---

## Tied Embeddings Discovery

Checked `model_prep.py:97-100`:

```python
# Currently ALWAYS saves lm_head separately
client_state_dict["lm_head.weight"] = model.lm_head.weight.data.clone().cpu()
```

This wastes ~500MB if weights are identical (many models tie them).

**Fix:** Compare tensors during prep, conditionally skip saving.

---

## Phase 1: Measurement & Profiling (MVP)

**Goal:** Establish baseline metrics before optimization.

### Task 1.1: Add Per-Token Timing Instrumentation

**File:** `infemeral/client.py`
**Lines:** ~40 new

```python
@dataclass
class TokenTiming:
    embed_ms: float
    cloak_ms: float
    network_ms: float
    uncloak_ms: float
    de_embed_ms: float
    sample_ms: float
    total_ms: float

@dataclass
class GenerationMetrics:
    timings: list[TokenTiming]
    device: str
    peak_memory_mb: float
    tokens_per_sec: float
```

**Subtasks:**
1. Create timing dataclasses
2. Add `time.perf_counter()` around each phase in `generate()`
3. Add `return_metrics: bool = False` parameter to `generate()`
4. Add `--profile` CLI flag to print timing breakdown

### Task 1.2: Add Memory Profiling Hook

**File:** `infemeral/client.py`
**Lines:** ~20 new

**Subtasks:**
1. Track peak GPU memory via `torch.cuda.max_memory_allocated()`
2. Track CPU memory via `tracemalloc`
3. Include in `GenerationMetrics`

### Task 1.3: Create Benchmark Script

**File:** `scripts/benchmark_client.py` (new)
**Lines:** ~120

**Subtasks:**
1. Run `generate()` with fixed prompt (e.g., "The quick brown fox")
2. Generate fixed token count (e.g., 20 tokens)
3. Warmup run + N measurement runs
4. Report: tokens/sec, p50/p95/p99 per-phase latency, memory
5. Compare CPU vs GPU if CUDA available
6. Save results to `benchmark_results.json`

**Impacted Files:**
| File | Change Type | Description |
|------|-------------|-------------|
| `infemeral/client.py` | Modified | Add timing instrumentation |
| `scripts/benchmark_client.py` | New | Benchmark harness |

---

## Phase 2: GPU & Matrix Optimization

**Goal:** Ensure GPU acceleration works, reduce matrix memory.

### Task 2.1: Verify GPU Acceleration Path

**File:** `infemeral/crypto.py`
**Lines:** ~10 modified

**Current issue:** Cloaking matrix created on CPU, moved to device during `cloak()`.

```python
# Current: matrix on CPU, moved each call
matrix = ctx.matrix.to(device=device, dtype=hidden.dtype)
```

**Optimization:** Store matrix on target device from creation.

**Subtasks:**
1. Add `device` parameter to `create_cloaking_context()`
2. Create matrix directly on target device
3. Avoid per-call `.to()` overhead

### Task 2.2: Float16 Matrix Storage

**File:** `infemeral/crypto.py`
**Lines:** ~15 modified

**Current:** 4096×4096 × float32 = 64 MB

**Optimization:** Store as float16 = 32 MB

**Subtasks:**
1. Generate matrix in float32 (for numerical stability in QR)
2. Convert to float16 for storage
3. Cast to computation dtype during `cloak()`/`uncloak()`
4. Add validation test: `|M @ M.T - I| < 1e-3` (relaxed for float16)

**Risk:** Orthogonality degradation. Mitigate with explicit test.

### Task 2.3: Lazy Matrix Materialization (Optional)

**File:** `infemeral/crypto.py`
**Lines:** ~40 modified

**Optimization:** Store only seed (32 bytes), regenerate on first use.

**Subtasks:**
1. Modify `CloakingContext` to store seed
2. Add `get_matrix()` with lazy generation and caching
3. ~50ms one-time cost acceptable

**Benefit:** Lower memory until first token; useful for multi-client scenarios.

---

## Phase 3: De-embedding Optimization

**Goal:** Accelerate the 128k × 4096 matmul bottleneck.

**Note:** Batched de-embedding REMOVED - conflicts with TTFT priority.

### Task 3.1: Detect and Leverage Tied Embeddings

**File:** `infemeral/model_prep.py` + `infemeral/client.py`
**Lines:** ~25 modified

**Subtasks (model_prep.py):**
1. After loading model, check `torch.equal(embed.weight, lm_head.weight)`
2. If equal, skip saving `lm_head.weight`
3. Add metadata flag `tied_embeddings: true` to weights file

**Subtasks (client.py):**
1. Check for `lm_head.weight` in state_dict
2. If missing, set `self.lm_head.weight = self.embed_tokens.weight`
3. Log: "Using tied embeddings (512 MB saved)"

**Benefit:** 500 MB memory saved if weights are tied.

### Task 3.2: Top-K Vocabulary Subset (Optional)

**File:** `infemeral/client.py`
**Lines:** ~60 new

**Insight:** Most tokens come from top ~10k vocabulary. Full 128k rarely needed.

**Subtasks:**
1. Create reduced lm_head projection: `nn.Linear(4096, 10000)`
2. Map reduced indices to full vocabulary
3. Sample from reduced; if low confidence, fallback to full
4. Expected 90% hit rate on reduced vocab

**Tradeoff:** Complexity vs ~10× de-embed speedup for 90% of tokens.

**Risk:** Distribution shift could reduce hit rate. Add monitoring.

### Task 3.3: INT8 Quantized lm_head (Optional)

**File:** `infemeral/client.py`
**Lines:** ~30 new
**Dependency:** `torch.ao.quantization` (built-in) or `bitsandbytes`

**Subtasks:**
1. Quantize lm_head weights to INT8 post-training
2. Use `torch.nn.quantized.functional.linear`
3. Benchmark perplexity impact

**Tradeoff:** ~2% quality loss for 2× speed + 50% memory reduction.

---

## Phase 4: Network & Serialization Optimization

**Goal:** Reduce per-token network overhead (important since network dominates after GPU).

### Task 4.1: gRPC Connection Keepalive

**File:** `infemeral/client.py`
**Lines:** ~15 modified

**Current:** Basic channel without keepalive.

**Optimization:** Add keepalive to prevent connection drops.

```python
self._channel = grpc.insecure_channel(
    self.server_url,
    options=[
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 10000),        # Ping every 10s
        ("grpc.keepalive_timeout_ms", 5000),      # 5s timeout
        ("grpc.keepalive_permit_without_calls", 1),
    ],
)
```

**Subtasks:**
1. Add keepalive options
2. Add connection health check method
3. Auto-reconnect on channel failure

### Task 4.2: LZ4 Tensor Compression

**File:** `infemeral/tensors.py`
**Lines:** ~40 new
**Dependency:** `lz4` package

**Current:** Raw bytes ~32 KB per token.

**Optimization:** LZ4 compression before encryption.

**Subtasks:**
1. Add `compress_tensor()` using `lz4.frame.compress()`
2. Add `decompress_tensor()` for response
3. Only compress if size > 4KB (avoid overhead for tiny payloads)
4. Update proto with optional compression flag

**Expected:** 30-50% size reduction, <0.5ms latency.

### Task 4.3: Request Pipelining (Deferred)

**Complexity:** High (async rewrite)
**Benefit:** Hide network latency by overlapping compute

Defer until synchronous path fully optimized.

---

## Phase 5: Model Preparation Optimization

**Goal:** Reduce client weight file size and improve load time.

### Task 5.1: Conditional lm_head Saving

**File:** `infemeral/model_prep.py`
**Lines:** ~20 modified

**Subtasks:**
1. Check if `embed_tokens.weight` and `lm_head.weight` are identical
2. If identical, don't save `lm_head.weight`
3. Add `"tied_embeddings": true` to safetensors metadata
4. Log savings: "Tied embeddings detected, saving 512 MB"

### Task 5.2: Float16 Weight Storage

**File:** `infemeral/model_prep.py`
**Lines:** ~10 modified

**Current:** Weights saved as float32 (default).

**Optimization:** Save as float16 (half size, sufficient precision).

**Subtasks:**
1. Add `.half()` before saving
2. Verify no precision issues in embedding lookup
3. Update client to handle float16 weights

---

## Impacted Files Summary

| File | Phase | Change Type | Description |
|------|-------|-------------|-------------|
| `infemeral/client.py` | 1, 2, 3, 4 | Modified | Timing, GPU path, tied weights, keepalive |
| `infemeral/crypto.py` | 2 | Modified | Device placement, float16 matrix |
| `infemeral/tensors.py` | 4 | Modified | LZ4 compression |
| `infemeral/model_prep.py` | 3, 5 | Modified | Tied weight detection, float16 save |
| `scripts/benchmark_client.py` | 1 | New | Benchmark harness |
| `tests/test_client_perf.py` | All | New | Performance regression tests |
| `pyproject.toml` | 4 | Modified | Add `lz4` dependency |

---

## Dependency Graph

```
Phase 1 (Measurement) ─────────────────────────────────────┐
    │                                                       │
    ├──► Task 2.1 (GPU path) ──► Task 2.2 (float16 matrix) │
    │                                                       │
    ├──► Task 3.1 (Tied embeddings) ◄── Task 5.1 (prep)    │
    │                                                       │
    ├──► Task 4.1 (Keepalive) ──► Task 4.2 (LZ4)           │
    │                                                       │
    └──► Task 5.2 (float16 weights)                         │
                                                            │
    [OPTIONAL based on profiling results] ◄─────────────────┘
    ├──► Task 2.3 (Lazy matrix)
    ├──► Task 3.2 (Top-K vocab)
    └──► Task 3.3 (INT8 quant)
```

**Critical Path for TTFT:** Phase 1 → Task 2.1 → Task 4.1

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Profiling overhead skews measurements | Medium | Use `perf_counter()`, disable in production |
| Float16 matrix loses orthogonality | High | Add `torch.allclose(M @ M.T, I, atol=1e-3)` test |
| GPU memory fragmentation | Medium | Call `torch.cuda.empty_cache()` periodically |
| LZ4 adds latency for small tensors | Low | Only compress if size > 4KB |
| Tied embedding detection fails | Low | Fallback to loading both weights |
| Top-K filtering misses correct token | High | Always fallback to full vocab if confidence < threshold |

---

## Success Criteria

### Phase 1 (Measurement)
- [ ] Benchmark script produces reproducible metrics
- [ ] Timing breakdown identifies ≥80% of per-token latency
- [ ] Baseline documented: tokens/sec, latency breakdown, memory
- [ ] CPU vs GPU comparison available

### Phase 2 (GPU & Matrix)
- [ ] GPU path verified working (no CPU fallback when CUDA available)
- [ ] Matrix memory reduced by 50% (64 MB → 32 MB)
- [ ] Orthogonality preserved: `|M @ M.T - I| < 1e-3`
- [ ] No per-call `.to()` overhead

### Phase 3 (De-embedding)
- [ ] Tied embeddings detected automatically during model prep
- [ ] Client loads single weight set when tied (512 MB saved)
- [ ] Output quality unchanged

### Phase 4 (Network)
- [ ] Connection persists across requests (no reconnect)
- [ ] LZ4 reduces payload by ≥30%
- [ ] Compression latency < 0.5ms

### Phase 5 (Model Prep)
- [ ] Weight file 50% smaller with float16
- [ ] Client load time improved

---

## Suggested Tests

### Performance Regression Tests
```python
# tests/test_client_perf.py

def test_gpu_path_used_when_available():
    """Verify tensors stay on CUDA when available."""
    client = Client(device="cuda")
    assert client.embedding.embed_tokens.weight.device.type == "cuda"
    assert client.cloaking_ctx.matrix.device.type == "cuda"

def test_no_device_transfer_during_cloak():
    """Cloak should not call .to() if matrix already on device."""
    # Mock torch.Tensor.to and verify not called

def test_orthogonal_matrix_float16_valid():
    """Float16 orthogonal matrix should satisfy M @ M.T ≈ I."""
    ctx = create_cloaking_context(seed=42, dtype=torch.float16)
    M = ctx.matrix.float()  # Upcast for precision
    I = torch.eye(M.shape[0])
    assert torch.allclose(M @ M.T, I, atol=1e-3)

def test_tied_embeddings_detected():
    """model_prep should detect and skip duplicate weights."""
    # Create model with tied weights, run prep, verify single weight in output

def test_tied_embeddings_loaded_correctly():
    """Client should use single weight for both embed and lm_head."""
    # Load weights without lm_head, verify lm_head.weight is embed.weight

def test_lz4_compression_roundtrip():
    """Compressed tensor should decompress to identical bytes."""
    data = serialize_tensor(torch.randn(1, 1, 4096))
    compressed = compress_tensor(data)
    decompressed = decompress_tensor(compressed)
    assert data == decompressed

def test_lz4_skipped_for_small_tensors():
    """Compression should be skipped for tensors < 4KB."""
    small_data = b"x" * 1000
    result = compress_tensor(small_data)
    assert result == small_data  # No compression applied

def test_grpc_keepalive_configured():
    """Verify keepalive options are set on channel."""
    client = Client()
    _ = client.stub  # Trigger channel creation
    # Verify channel options (may need internal inspection)
```

### Benchmark Assertions
```python
# scripts/benchmark_client.py

BASELINE_CPU_MS = 25.0  # Expected per-token on CPU
BASELINE_GPU_MS = 5.0   # Expected per-token on GPU

def assert_performance_regression():
    """Fail CI if regression detected."""
    results = run_benchmark()

    if results["device"] == "cuda":
        assert results["p50_latency_ms"] <= BASELINE_GPU_MS * 1.2
    else:
        assert results["p50_latency_ms"] <= BASELINE_CPU_MS * 1.2
```

---

## Out of Scope (Deferred)

- **Batched de-embedding** - Conflicts with TTFT priority
- **Request pipelining** - High complexity async rewrite
- **Blockwise cloaking** - Memory not a constraint at 200MB
- **Disk offload** - Memory not a constraint
- **Speculative decoding** - Requires server changes
- **WebAssembly port** - Different target platform

---

## Recommended Execution Order (TTFT Priority)

1. **Phase 1 (all tasks)** - Get baseline metrics first
2. **Task 2.1** - GPU path verification (biggest win: ~20ms/token)
3. **Task 4.1** - gRPC keepalive (reduces connection overhead)
4. **Task 2.2** - Float16 matrix (32 MB memory saved)
5. **Task 3.1 + 5.1** - Tied embeddings (512 MB saved if applicable)
6. **Task 4.2** - LZ4 compression (network payload reduction)
7. **Task 5.2** - Float16 weights (faster load, smaller file)
8. **Optional tasks** - Based on profiling showing remaining bottlenecks

---

## Architecture After Optimization

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Client (TTFT Optimized)                           │
├─────────────────────────────────────────────────────────────────────┤
│  INIT (one-time):                                                    │
│  ├─ Detect CUDA, set device                                         │
│  ├─ Load embed_tokens (float16, ~256 MB)                            │
│  ├─ Share lm_head weight if tied (0 MB extra)                       │
│  ├─ Generate cloaking matrix on device (float16, ~32 MB)            │
│  └─ Establish gRPC channel with keepalive                           │
│                                                                      │
│  PER-TOKEN (GPU path):                                               │
│  1. Embed [1, 1, 4096] ─────────────────────────────── 0.1ms        │
│  2. Cloak (matrix on device, no transfer) ──────────── 0.1ms        │
│  3. Serialize + LZ4 + AES encrypt ──────────────────── 1.0ms (CPU)  │
│  4. gRPC call (keepalive, no reconnect) ────────────── RTT          │
│  5. Decrypt + decompress + deserialize ─────────────── 1.0ms (CPU)  │
│  6. Uncloak ────────────────────────────────────────── 0.1ms        │
│  7. De-embed (shared weights on GPU) ───────────────── 0.5ms        │
│  8. Sample ─────────────────────────────────────────── 0.2ms        │
│                                                         ─────────    │
│                                             Compute:    ~3ms         │
│                                             + Network:  10-50ms      │
│                                                                      │
│  Memory: ~300 MB GPU (down from ~600 MB)                             │
│  Latency: ~15ms/token GPU, ~35ms/token CPU (excl. network)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Second-Order Effects

| If you implement... | Then... |
|---------------------|---------|
| GPU acceleration | Network RTT becomes dominant bottleneck |
| Tied embedding detection | Need to update model_prep.py AND client.py |
| LZ4 compression | Proto change optional but recommended for version compat |
| Float16 matrix | Need to relax orthogonality tolerance in tests |
| Top-K vocabulary | Need to ship token frequency data with client |
