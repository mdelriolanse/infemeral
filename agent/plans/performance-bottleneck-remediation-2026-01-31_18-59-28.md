# Performance Bottleneck Remediation Plan

**Created:** 2026-01-31_18-59-28
**Objective:** Reduce inference latency from ~1 token/sec to 40-100 tokens/sec for 150-token generation on RTX 4090

## Executive Summary

Current performance: **147 seconds** for 150 tokens (~1 token/sec)
Target performance: **1.5-3.75 seconds** for 150 tokens (40-100 tokens/sec)
Expected improvement: **50-100x speedup**

---

## Clarification Questions

1. **In-memory cache persistence:** Should the in-memory KV cache survive server restarts, or is it acceptable to lose it on restart? (Affects Phase 1 complexity)
2. **Session timeout behavior:** What should happen when a session expires while KV cache is in memory? Silent discard or persist to disk?
3. **Network architecture constraints:** For Phase 2 (server-side generation), is it acceptable to change the gRPC proto schema, or must it remain backward compatible?
4. **Logging requirements:** Are there compliance/audit requirements that mandate retaining the verbose logging in Phase 3?

---

## Dependency Mapping

```
Phase 1 (KV Cache) ─────┬─────> Phase 3 (Logging) ─────> Phase 4 (empty_cache)
                        │
                        └─────> Phase 5 (CPU↔GPU) ─────> Phase 6 (Crypto)
                                       │
Phase 2 (Server-side Gen) ◄────────────┘
```

**Blockers:**
- Phase 1 must complete before Phase 2 (server-side generation needs in-memory KV cache)
- Network infrastructure changes in Phase 2 may affect Phase 5 testing
- All phases require RunPod server access for integration testing

---

## Phase 1: In-Memory KV Cache (Critical - 50-90% improvement)

### Problem Statement
`infemeral/server.py:555-563` reads/writes KV cache to disk **every token**. At token 150, this involves ~24MB per write with encryption overhead.

**Current flow:**
```
Token N → load_kv_cache() [disk read] → forward_transformer() → save_kv_cache() [disk write]
```

### Implementation Tasks

| Task | File | Lines | Description |
|------|------|-------|-------------|
| 1.1 | `infemeral/server.py` | New | Create `SessionKVCache` class with in-memory dict storage |
| 1.2 | `infemeral/server.py` | 555-563 | Replace disk I/O with in-memory cache lookup in `Infer()` |
| 1.3 | `infemeral/server.py` | 183-205 | Make `save_kv_cache()` optional (checkpoint-only mode) |
| 1.4 | `infemeral/server.py` | 133-180 | Add fallback from memory to disk in `load_kv_cache()` |
| 1.5 | `infemeral/server.py` | New | Add `persist_session()` method for explicit checkpoint |
| 1.6 | `infemeral/server.py` | 214-243 | Update `cleanup_old_sessions()` to handle memory cache |
| 1.7 | `infemeral/config.py` | New | Add `kv_cache_mode: Literal["memory", "disk", "hybrid"]` setting |

### Impacted Files
- `infemeral/server.py` (primary)
- `infemeral/config.py` (settings)
- `tests/test_server.py` (unit tests)

### Success Criteria
- [ ] KV cache remains in GPU memory during multi-token generation
- [ ] No disk I/O between tokens within same session
- [ ] Memory usage bounded by `max_sessions * max_context_length * cache_size`
- [ ] Graceful fallback to disk when memory pressure detected

### Suggested Tests

**File:** `tests/test_kv_cache_memory.py`

```python
"""Integration tests for in-memory KV cache - queries real server."""

import time
import pytest
from infemeral.client import Client

class TestInMemoryKVCache:
    """Tests that verify KV cache stays in memory during inference."""

    @pytest.fixture
    def remote_client(self, runpod_server_url):
        """Create client connected to real RunPod server."""
        client = Client(
            weights_path='/workspace/weights/client_weights.safetensors',
            server_url=runpod_server_url,
            device='cuda'
        )
        yield client
        client.close()

    def test_no_disk_io_between_tokens(self, remote_client, runpod_ssh):
        """Verify no disk writes occur between token generations."""
        # Get initial disk write count
        initial_writes = runpod_ssh.exec("cat /proc/diskstats | awk '{sum+=$10} END {print sum}'")

        # Generate 20 tokens
        result, metrics = remote_client.generate(
            "Hello",
            max_new_tokens=20,
            return_metrics=True
        )

        # Get final disk write count
        final_writes = runpod_ssh.exec("cat /proc/diskstats | awk '{sum+=$10} END {print sum}'")

        # Allow 2 writes (session start + end), not 40 (2 per token)
        disk_writes = int(final_writes) - int(initial_writes)
        assert disk_writes <= 4, f"Expected ≤4 disk writes, got {disk_writes}"

    def test_latency_improvement_20_tokens(self, remote_client):
        """Verify per-token latency < 100ms (was 500-1000ms with disk I/O)."""
        result, metrics = remote_client.generate(
            "The quick brown fox",
            max_new_tokens=20,
            return_metrics=True
        )

        # Calculate median network latency (includes server processing)
        network_times = [t.network_ms for t in metrics.timings]
        median_network_ms = sorted(network_times)[len(network_times) // 2]

        # Should be < 100ms per token (was 500-1000ms with disk I/O)
        assert median_network_ms < 100, f"Median network latency {median_network_ms:.1f}ms exceeds 100ms"

    def test_memory_bounded(self, remote_client, runpod_ssh):
        """Verify GPU memory stays bounded during long generation."""
        # Get baseline GPU memory
        baseline_mem = runpod_ssh.exec("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")

        # Generate 100 tokens across 5 sessions
        for _ in range(5):
            remote_client.session_id = secrets.token_hex(16)  # New session
            remote_client.generate("Test prompt", max_new_tokens=100)

        # Check memory didn't grow unboundedly
        final_mem = runpod_ssh.exec("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")

        # Allow 2GB growth max (5 sessions * ~400MB each)
        mem_growth = int(final_mem) - int(baseline_mem)
        assert mem_growth < 2048, f"Memory grew by {mem_growth}MB, expected < 2048MB"
```

### Deployment Steps (Builder Agent)
1. **rsync updated files to server:**
   ```bash
   rsync -avz --progress infemeral/ root@$POD_IP:/workspace/infemeral-src/infemeral/ -e "ssh -p $POD_PORT"
   ```
2. **Restart server process:**
   ```bash
   ssh -p $POD_PORT root@$POD_IP "pkill -f 'python.*server' && cd /workspace/infemeral-src && nohup /mnt/.venv/bin/python -m infemeral.server --mode grpc &"
   ```
3. **Run integration tests:**
   ```bash
   ssh -p $POD_PORT root@$POD_IP "cd /workspace/infemeral-src && /mnt/.venv/bin/pytest tests/test_kv_cache_memory.py -v"
   ```

---

## Phase 2: Server-Side Generation Loop (Architectural - 20-40% improvement)

### Problem Statement
`infemeral/client.py:296-315` sends one gRPC round-trip per token. Each round-trip includes serialize → encrypt → network → decrypt → deserialize overhead.

**Current flow:**
```
Client                          Server
  │                               │
  ├──── token 1 request ─────────►│
  │◄─── token 1 response ─────────┤
  ├──── token 2 request ─────────►│
  │◄─── token 2 response ─────────┤
  ... (150 round trips)
```

**Target flow:**
```
Client                          Server
  │                               │
  ├──── generate(prompt, n=150) ─►│
  │     (streaming response)      │ ◄── tokens generated server-side
  │◄─── token 1 ──────────────────┤
  │◄─── token 2 ──────────────────┤
  ... (1 request, 150 streamed responses)
```

### Implementation Tasks

| Task | File | Lines | Description |
|------|------|-------|-------------|
| 2.1 | `tensor_service.proto` | New | Add `StreamingInferenceRequest` and `stream InferenceResponse` |
| 2.2 | `infemeral/server.py` | New | Add `StreamingInfer()` method with server-side generation loop |
| 2.3 | `infemeral/client.py` | 296-315 | Add `generate_streaming()` that uses new RPC |
| 2.4 | `infemeral/client.py` | 193-231 | Extract `_call_server_streaming()` for stream handling |
| 2.5 | Regenerate | - | Regenerate protobuf stubs after proto changes |
| 2.6 | `infemeral/server.py` | New | Implement server-side sampling (copy from client) |

### Impacted Files
- `tensor_service.proto` (schema change)
- `infemeral/server.py` (streaming implementation)
- `infemeral/client.py` (streaming client)
- `infemeral/tensor_service_pb2.py` (regenerated)
- `infemeral/tensor_service_pb2_grpc.py` (regenerated)

### Success Criteria
- [ ] Single gRPC request initiates full generation
- [ ] Tokens stream back as they're generated
- [ ] Latency reduced by eliminating per-token network overhead
- [ ] Backward compatible: old `Infer()` still works

### Suggested Tests

**File:** `tests/test_streaming_inference.py`

```python
"""Integration tests for streaming server-side generation."""

import time
import pytest
from infemeral.client import Client

class TestStreamingInference:
    """Tests for server-side generation with streaming response."""

    @pytest.fixture
    def remote_client(self, runpod_server_url):
        """Create client connected to real RunPod server."""
        client = Client(
            weights_path='/workspace/weights/client_weights.safetensors',
            server_url=runpod_server_url,
            device='cuda'
        )
        yield client
        client.close()

    def test_streaming_reduces_total_latency(self, remote_client):
        """Verify streaming is faster than per-token requests."""
        prompt = "The quick brown fox"
        max_tokens = 50

        # Time per-token mode
        start = time.perf_counter()
        result_per_token = remote_client.generate(prompt, max_new_tokens=max_tokens)
        per_token_time = time.perf_counter() - start

        # Time streaming mode
        start = time.perf_counter()
        result_streaming = remote_client.generate_streaming(prompt, max_new_tokens=max_tokens)
        streaming_time = time.perf_counter() - start

        # Streaming should be at least 20% faster
        improvement = (per_token_time - streaming_time) / per_token_time
        assert improvement >= 0.20, f"Streaming only {improvement*100:.1f}% faster, expected ≥20%"

    def test_streaming_tokens_arrive_incrementally(self, remote_client):
        """Verify tokens are yielded as they're generated, not all at once."""
        tokens_received = []
        timestamps = []

        for token in remote_client.generate_streaming_iter("Hello", max_new_tokens=10):
            tokens_received.append(token)
            timestamps.append(time.perf_counter())

        # Verify tokens arrived at different times (not batched)
        assert len(timestamps) >= 2
        time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        # Each token should take some time (not all instant)
        avg_diff = sum(time_diffs) / len(time_diffs)
        assert avg_diff > 0.001, "Tokens arrived too quickly - likely batched, not streamed"

    def test_streaming_handles_eos(self, remote_client):
        """Verify streaming stops at EOS token."""
        # Short prompt that should trigger EOS
        result = remote_client.generate_streaming("Say 'hi'", max_new_tokens=100)

        # Should be much shorter than 100 tokens
        token_count = len(remote_client.tokenizer.encode(result))
        assert token_count < 50, f"Generated {token_count} tokens, expected < 50 (EOS not detected?)"

    def test_100_tokens_under_5_seconds(self, remote_client):
        """Performance target: 100 tokens in under 5 seconds (20+ tok/s)."""
        start = time.perf_counter()
        result = remote_client.generate_streaming(
            "Write a short story about a robot.",
            max_new_tokens=100
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"100 tokens took {elapsed:.1f}s, expected < 5.0s"
```

### Deployment Steps (Builder Agent)
1. **rsync updated files including proto:**
   ```bash
   rsync -avz --progress infemeral/ tensor_service.proto root@$POD_IP:/workspace/infemeral-src/ -e "ssh -p $POD_PORT"
   ```
2. **Regenerate protobuf on server:**
   ```bash
   ssh -p $POD_PORT root@$POD_IP "cd /workspace/infemeral-src && python -m grpc_tools.protoc -I. --python_out=infemeral --grpc_python_out=infemeral tensor_service.proto"
   ```
3. **Restart server and run tests:**
   ```bash
   ssh -p $POD_PORT root@$POD_IP "pkill -f 'python.*server' && cd /workspace/infemeral-src && nohup /mnt/.venv/bin/python -m infemeral.server --mode grpc &"
   ssh -p $POD_PORT root@$POD_IP "cd /workspace/infemeral-src && /mnt/.venv/bin/pytest tests/test_streaming_inference.py -v"
   ```

---

## Phase 3: Remove Excessive Logging (High - 5-15% improvement)

### Problem Statement
`infemeral/server.py:392-493` logs at `INFO` level for every layer on every token. For 150 tokens × 32 layers = 4,800+ log entries with string formatting overhead.

**Current code (server.py:392-393, 423-435, 449, 464-484):**
```python
logger.info(f"Loading layer {layer_idx} cache: key={k_contig.shape}, ...")
logger.info(f"Layer {i} input - cache has {cache_len} entries")
logger.info(f"Layer {i}: calling with hidden_states={hidden_states.shape}, ...")
logger.info(f"Layer {i}: returned tuple len={len(layer_out)}, cache type=...")
```

### Implementation Tasks

| Task | File | Lines | Description |
|------|------|-------|-------------|
| 3.1 | `infemeral/server.py` | 392-393 | Change to `logger.debug()` or remove |
| 3.2 | `infemeral/server.py` | 423-435 | Change to `logger.debug()` or remove |
| 3.3 | `infemeral/server.py` | 449 | Change to `logger.debug()` or remove |
| 3.4 | `infemeral/server.py` | 464-484 | Change to `logger.debug()` or remove |
| 3.5 | `infemeral/server.py` | 400, 402, 497, 512-513 | Change remaining verbose logs to debug |
| 3.6 | `infemeral/config.py` | New | Add `log_level: str = "INFO"` setting |
| 3.7 | `infemeral/server.py` | 604-608 | Use configurable log level |

### Impacted Files
- `infemeral/server.py` (logging changes)
- `infemeral/config.py` (optional log level setting)

### Success Criteria
- [ ] No `logger.info()` calls inside `forward_transformer()` loop
- [ ] Debug logging available via `LOG_LEVEL=DEBUG` environment variable
- [ ] Per-token latency reduced by 5-20ms

### Suggested Tests

**File:** `tests/test_logging_performance.py`

```python
"""Integration tests verifying logging doesn't impact performance."""

import time
import pytest
from infemeral.client import Client

class TestLoggingPerformance:
    """Tests that verify logging overhead is minimal."""

    @pytest.fixture
    def remote_client(self, runpod_server_url):
        """Create client connected to real RunPod server."""
        client = Client(
            weights_path='/workspace/weights/client_weights.safetensors',
            server_url=runpod_server_url,
            device='cuda'
        )
        yield client
        client.close()

    def test_no_excessive_logging(self, remote_client, runpod_ssh):
        """Verify INFO-level logs don't scale with token count."""
        # Clear logs
        runpod_ssh.exec("truncate -s 0 /var/log/infemeral.log")

        # Generate 50 tokens
        remote_client.generate("Test", max_new_tokens=50)

        # Count log lines
        log_count = int(runpod_ssh.exec("wc -l < /var/log/infemeral.log"))

        # Should be < 100 lines (2 per request, not 50*32*4 = 6400)
        assert log_count < 100, f"Generated {log_count} log lines for 50 tokens, expected < 100"

    def test_logging_latency_under_5ms(self, remote_client):
        """Verify per-token overhead from logging is < 5ms."""
        # Generate with metrics
        result, metrics = remote_client.generate(
            "Hello",
            max_new_tokens=20,
            return_metrics=True
        )

        # Network latency should now exclude logging overhead
        # Baseline without logging was measured at ~50ms, with logging was ~70ms
        network_times = [t.network_ms for t in metrics.timings[1:]]  # Skip first (prompt)
        median_network_ms = sorted(network_times)[len(network_times) // 2]

        # If Phase 1 is done, this should be < 100ms
        # Logging removal should show ~15-20% improvement
        assert median_network_ms < 100, f"Median {median_network_ms:.1f}ms, logging may still be excessive"
```

### Deployment Steps (Builder Agent)
1. **rsync updated server.py:**
   ```bash
   rsync -avz infemeral/server.py root@$POD_IP:/workspace/infemeral-src/infemeral/ -e "ssh -p $POD_PORT"
   ```
2. **Restart server:**
   ```bash
   ssh -p $POD_PORT root@$POD_IP "pkill -f 'python.*server' && cd /workspace/infemeral-src && nohup /mnt/.venv/bin/python -m infemeral.server --mode grpc &"
   ```
3. **Run tests:**
   ```bash
   ssh -p $POD_PORT root@$POD_IP "cd /workspace/infemeral-src && /mnt/.venv/bin/pytest tests/test_logging_performance.py -v"
   ```

---

## Phase 4: Remove `torch.cuda.empty_cache()` (Medium - 1-5% improvement)

### Problem Statement
`infemeral/server.py:572-574` calls `torch.cuda.empty_cache()` after every inference, forcing a CUDA synchronization barrier.

**Current code:**
```python
# Memory wipe
del hidden, output, new_kv
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # SYNC BARRIER EVERY TOKEN
```

### Implementation Tasks

| Task | File | Lines | Description |
|------|------|-------|-------------|
| 4.1 | `infemeral/server.py` | 572-574 | Remove `del` and `empty_cache()` calls |
| 4.2 | `infemeral/server.py` | 735-738 | Remove same pattern from `handler()` |
| 4.3 | `infemeral/config.py` | New | Add optional `cuda_memory_management: bool` for debugging |

### Impacted Files
- `infemeral/server.py`
- `infemeral/config.py` (optional)

### Success Criteria
- [ ] No `torch.cuda.empty_cache()` in hot path
- [ ] GPU memory still bounded (PyTorch allocator handles this)
- [ ] 1-5ms reduction in per-token latency

### Suggested Tests

**File:** `tests/test_cuda_memory.py`

```python
"""Integration tests for CUDA memory management after removing empty_cache()."""

import time
import pytest
from infemeral.client import Client

class TestCudaMemoryManagement:
    """Tests verifying memory stays bounded without explicit cache clearing."""

    @pytest.fixture
    def remote_client(self, runpod_server_url):
        """Create client connected to real RunPod server."""
        client = Client(
            weights_path='/workspace/weights/client_weights.safetensors',
            server_url=runpod_server_url,
            device='cuda'
        )
        yield client
        client.close()

    def test_memory_stable_over_many_generations(self, remote_client, runpod_ssh):
        """Verify GPU memory doesn't grow unboundedly without empty_cache()."""
        # Baseline
        baseline = int(runpod_ssh.exec("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"))

        # Generate 500 tokens total across 10 requests
        for i in range(10):
            remote_client.generate(f"Request {i}", max_new_tokens=50)

        # Check memory
        final = int(runpod_ssh.exec("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"))

        # Allow 500MB growth max (reasonable for KV cache)
        growth = final - baseline
        assert growth < 500, f"Memory grew {growth}MB without empty_cache(), may have leak"

    def test_no_cuda_sync_in_hot_path(self, remote_client, runpod_ssh):
        """Verify no CUDA synchronization barriers in inference path."""
        # This is a proxy test - we measure latency variance
        # empty_cache() causes sync which increases latency variance

        result, metrics = remote_client.generate(
            "Hello world",
            max_new_tokens=30,
            return_metrics=True
        )

        network_times = [t.network_ms for t in metrics.timings[1:]]

        # Calculate variance
        mean = sum(network_times) / len(network_times)
        variance = sum((t - mean) ** 2 for t in network_times) / len(network_times)
        std_dev = variance ** 0.5

        # With sync barriers, std_dev is typically > 10ms
        # Without, it should be < 5ms
        assert std_dev < 10, f"Latency std_dev {std_dev:.1f}ms suggests sync barriers still present"
```

### Deployment Steps (Builder Agent)
1. **rsync server.py:**
   ```bash
   rsync -avz infemeral/server.py root@$POD_IP:/workspace/infemeral-src/infemeral/ -e "ssh -p $POD_PORT"
   ```
2. **Restart and test:**
   ```bash
   ssh -p $POD_PORT root@$POD_IP "pkill -f 'python.*server' && cd /workspace/infemeral-src && nohup /mnt/.venv/bin/python -m infemeral.server --mode grpc &"
   ssh -p $POD_PORT root@$POD_IP "cd /workspace/infemeral-src && /mnt/.venv/bin/pytest tests/test_cuda_memory.py -v"
   ```

---

## Phase 5: Optimize CPU↔GPU Tensor Transfers (Medium - 5-10% improvement)

### Problem Statement
`infemeral/tensors.py:50-83` forces CPU round-trips for all tensor serialization:
```python
# serialize_tensor (line 50)
data = tensor.detach().cpu().numpy().tobytes()  # GPU → CPU → NumPy

# deserialize_tensor (line 75)
tensor = torch.from_numpy(arr.copy())  # Extra copy
return tensor.to(device)  # CPU → GPU
```

### Implementation Tasks

| Task | File | Lines | Description |
|------|------|-------|-------------|
| 5.1 | `infemeral/tensors.py` | 34-53 | Use `torch.save` with `io.BytesIO` for faster serialize |
| 5.2 | `infemeral/tensors.py` | 56-83 | Use `torch.load` with direct device placement |
| 5.3 | `infemeral/tensors.py` | 75 | Remove unnecessary `.copy()` for read-only buffer |
| 5.4 | `infemeral/tensors.py` | New | Add `serialize_tensor_fast()` using CUDA-aware methods |
| 5.5 | `infemeral/tensors.py` | 148-166 | Optimize `pack_kv_cache_v2` to avoid CPU transfer |

### Impacted Files
- `infemeral/tensors.py` (primary)
- `infemeral/client.py` (if API changes)
- `infemeral/server.py` (if API changes)

### Success Criteria
- [ ] Tensors that start on GPU avoid CPU round-trip where possible
- [ ] 2-10ms reduction in per-token latency
- [ ] No behavior change for existing callers

### Suggested Tests

**File:** `tests/test_tensor_transfer.py`

```python
"""Integration tests for optimized tensor serialization."""

import time
import pytest
import torch
from infemeral.tensors import serialize_tensor, deserialize_tensor
from infemeral.client import Client

class TestTensorTransferOptimization:
    """Tests for optimized CPU↔GPU tensor handling."""

    @pytest.fixture
    def remote_client(self, runpod_server_url):
        """Create client connected to real RunPod server."""
        client = Client(
            weights_path='/workspace/weights/client_weights.safetensors',
            server_url=runpod_server_url,
            device='cuda'
        )
        yield client
        client.close()

    def test_serialization_roundtrip_correctness(self):
        """Verify optimized serialization preserves tensor values."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        original = torch.randn(1, 100, 4096, dtype=torch.float16, device='cuda')

        data, shape, dtype = serialize_tensor(original)
        recovered = deserialize_tensor(data, shape, dtype, device='cuda')

        torch.testing.assert_close(original.cpu(), recovered.cpu())

    def test_serialization_latency_improved(self, remote_client):
        """Verify serialization overhead is reduced."""
        # Generate with metrics to measure embed + de_embed (includes serialization)
        result, metrics = remote_client.generate(
            "Hello",
            max_new_tokens=20,
            return_metrics=True
        )

        # Embed and de_embed should be < 5ms each (include local serialization)
        embed_times = [t.embed_ms for t in metrics.timings]
        de_embed_times = [t.de_embed_ms for t in metrics.timings]

        median_embed = sorted(embed_times)[len(embed_times) // 2]
        median_de_embed = sorted(de_embed_times)[len(de_embed_times) // 2]

        assert median_embed < 5, f"Embed median {median_embed:.1f}ms, expected < 5ms"
        assert median_de_embed < 5, f"De-embed median {median_de_embed:.1f}ms, expected < 5ms"

    def test_large_tensor_serialization_performance(self):
        """Verify large tensor serialization is efficient."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Large tensor (similar to KV cache)
        large = torch.randn(1, 32, 2048, 128, dtype=torch.float16, device='cuda')

        start = time.perf_counter()
        for _ in range(10):
            data, shape, dtype = serialize_tensor(large)
            _ = deserialize_tensor(data, shape, dtype, device='cuda')
        elapsed = time.perf_counter() - start

        # 10 round-trips should take < 500ms (< 50ms each)
        assert elapsed < 0.5, f"10 serialize/deserialize took {elapsed*1000:.0f}ms, expected < 500ms"
```

### Deployment Steps (Builder Agent)
1. **rsync tensors.py:**
   ```bash
   rsync -avz infemeral/tensors.py root@$POD_IP:/workspace/infemeral-src/infemeral/ -e "ssh -p $POD_PORT"
   ```
2. **Run local unit tests first:**
   ```bash
   pytest tests/test_tensor_transfer.py::TestTensorTransferOptimization::test_serialization_roundtrip_correctness -v
   ```
3. **Deploy and test on server:**
   ```bash
   ssh -p $POD_PORT root@$POD_IP "cd /workspace/infemeral-src && /mnt/.venv/bin/pytest tests/test_tensor_transfer.py -v"
   ```

---

## Phase 6: Cache AESGCM Cipher Instance (Low - <1% improvement)

### Problem Statement
`infemeral/crypto.py:28-46` creates new `AESGCM(key)` objects for every encrypt/decrypt call:
```python
def encrypt_bytes(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)  # NEW OBJECT EVERY CALL
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
```

### Implementation Tasks

| Task | File | Lines | Description |
|------|------|-------|-------------|
| 6.1 | `infemeral/crypto.py` | New | Create `CryptoSession` class that caches `AESGCM` instance |
| 6.2 | `infemeral/client.py` | 142-143 | Replace `session_key` with `CryptoSession` instance |
| 6.3 | `infemeral/client.py` | 199, 223 | Use cached cipher for encrypt/decrypt |
| 6.4 | `infemeral/server.py` | New | Cache cipher per session_id (optional) |

### Impacted Files
- `infemeral/crypto.py` (new class)
- `infemeral/client.py` (use cached cipher)
- `infemeral/server.py` (optional optimization)

### Success Criteria
- [ ] Single `AESGCM` instance per session
- [ ] 0.5-1ms reduction in per-token latency
- [ ] Backward compatible API

### Suggested Tests

**File:** `tests/test_crypto_caching.py`

```python
"""Integration tests for cached crypto operations."""

import time
import pytest
from infemeral.crypto import CryptoSession, generate_session_key
from infemeral.client import Client

class TestCryptoCaching:
    """Tests for cached AESGCM cipher instances."""

    def test_crypto_session_caches_cipher(self):
        """Verify CryptoSession reuses AESGCM instance."""
        key = generate_session_key()
        session = CryptoSession(key)

        # Multiple encrypts should use same cipher
        cipher_id_1 = id(session._cipher)
        _ = session.encrypt(b"test1")
        cipher_id_2 = id(session._cipher)
        _ = session.encrypt(b"test2")
        cipher_id_3 = id(session._cipher)

        assert cipher_id_1 == cipher_id_2 == cipher_id_3

    def test_crypto_session_roundtrip(self):
        """Verify cached cipher produces correct results."""
        key = generate_session_key()
        session = CryptoSession(key)

        original = b"test data " * 1000
        ciphertext, nonce = session.encrypt(original)
        recovered = session.decrypt(ciphertext, nonce)

        assert recovered == original

    def test_crypto_latency_reduced(self):
        """Verify caching reduces crypto overhead."""
        key = generate_session_key()
        session = CryptoSession(key)
        data = b"x" * 16384  # 16KB payload

        # Warm up
        for _ in range(10):
            ct, nonce = session.encrypt(data)
            session.decrypt(ct, nonce)

        # Measure
        start = time.perf_counter()
        for _ in range(1000):
            ct, nonce = session.encrypt(data)
            session.decrypt(ct, nonce)
        elapsed = time.perf_counter() - start

        # 1000 encrypt+decrypt should take < 100ms (< 0.1ms each)
        assert elapsed < 0.1, f"1000 crypto ops took {elapsed*1000:.0f}ms, expected < 100ms"

    @pytest.fixture
    def remote_client(self, runpod_server_url):
        """Create client connected to real RunPod server."""
        client = Client(
            weights_path='/workspace/weights/client_weights.safetensors',
            server_url=runpod_server_url,
            device='cuda'
        )
        yield client
        client.close()

    def test_client_uses_cached_crypto(self, remote_client):
        """Verify client benefits from crypto caching."""
        result, metrics = remote_client.generate(
            "Test",
            max_new_tokens=20,
            return_metrics=True
        )

        # Network time includes crypto - should be minimal
        # This tests the full integration
        network_times = [t.network_ms for t in metrics.timings[1:]]
        median = sorted(network_times)[len(network_times) // 2]

        # Just verify it works - crypto is small part of total
        assert median < 100, f"Network median {median:.1f}ms"
```

### Deployment Steps (Builder Agent)
1. **rsync updated files:**
   ```bash
   rsync -avz infemeral/crypto.py infemeral/client.py root@$POD_IP:/workspace/infemeral-src/infemeral/ -e "ssh -p $POD_PORT"
   ```
2. **Run tests:**
   ```bash
   ssh -p $POD_PORT root@$POD_IP "cd /workspace/infemeral-src && /mnt/.venv/bin/pytest tests/test_crypto_caching.py -v"
   ```

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|---------------------|
| In-memory KV cache causes OOM | High | Medium | Implement LRU eviction, configurable max sessions |
| Streaming RPC breaks existing clients | High | Low | Keep old `Infer()` method, add new `StreamingInfer()` |
| Logging removal hides production issues | Medium | Medium | Keep debug-level logging, add structured error tracking |
| empty_cache removal causes fragmentation | Low | Low | Monitor memory over time, add periodic cleanup |
| Tensor optimization breaks precision | High | Low | Extensive roundtrip testing with torch.testing.assert_close |
| Crypto caching creates security issue | High | Very Low | Key is session-scoped, cipher instance doesn't leak key |

---

## Validation Checklist

### Per-Phase Validation
- [ ] All unit tests pass locally
- [ ] Files synced to RunPod server
- [ ] Server restarted cleanly
- [ ] Integration tests pass on server
- [ ] Performance improvement measured

### End-to-End Validation (After All Phases)

```bash
# Final performance benchmark
ssh -p $POD_PORT root@$POD_IP "cd /workspace/infemeral-src && time /mnt/.venv/bin/python -c \"
from infemeral.client import Client
client = Client(
    weights_path='/workspace/weights/client_weights.safetensors',
    server_url='localhost:50051',
    device='cuda'
)
result, metrics = client.generate('Hello how are you?', max_new_tokens=150, return_metrics=True)
print(f'Generated {metrics.total_tokens} tokens at {metrics.tokens_per_sec:.1f} tok/s')
client.close()
\""

# Target: 150 tokens in < 5 seconds (30+ tok/s)
```

---

## Summary

| Phase | Bottleneck | Expected Improvement | Complexity |
|-------|------------|---------------------|------------|
| 1 | KV Cache Disk I/O | 50-90% | Medium |
| 2 | Per-Token Network RT | 20-40% | High |
| 3 | Excessive Logging | 5-15% | Low |
| 4 | empty_cache() Sync | 1-5% | Low |
| 5 | CPU↔GPU Transfers | 5-10% | Medium |
| 6 | Crypto Object Creation | <1% | Low |

**Cumulative expected improvement:** 81-161%+ → achieving target of 40-100 tok/s from baseline of ~1 tok/s
