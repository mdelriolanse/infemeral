# KV Cache Multi-Turn Support Plan - Phase 2

**Created:** 2026-01-16
**Scope:** Enable multi-turn conversations with KV cache reuse
**Depends On:** Phase 1 (gRPC Server MVP) must be complete
**Target:** RunPod GPU pods

---

## Executive Summary

Current implementation has a **critical format mismatch**: KV cache is stored as flat `(keys, values)` tensors, but transformers expect per-layer structure `((k₀,v₀), ..., (k₃₁,v₃₁))`. Additionally, clients send the **full sequence every token** instead of just new tokens.

This plan fixes both issues to enable efficient multi-turn conversations.

---

## The Problem Illustrated

```
CURRENT (BROKEN):
┌─────────────────────────────────────────────────────────────┐
│ Token 1: Client sends [1, 100, 4096] (100 tokens)           │
│ Token 2: Client sends [1, 101, 4096] (101 tokens) ← WASTE   │
│ Token 3: Client sends [1, 102, 4096] (102 tokens) ← WASTE   │
│ ...                                                          │
│ Token 50: Client sends [1, 149, 4096] (149 tokens) ← WASTE  │
│                                                              │
│ Total data sent: ~300MB for 50 tokens                       │
│ KV cache: Loaded but NEVER USED (format mismatch)           │
└─────────────────────────────────────────────────────────────┘

FIXED (PHASE 2):
┌─────────────────────────────────────────────────────────────┐
│ Token 1: Client sends [1, 100, 4096] (100 tokens)           │
│ Token 2: Client sends [1, 1, 4096] (1 token) + session_id   │
│ Token 3: Client sends [1, 1, 4096] (1 token) + session_id   │
│ ...                                                          │
│ Token 50: Client sends [1, 1, 4096] (1 token) + session_id  │
│                                                              │
│ Total data sent: ~6MB for 50 tokens (50x reduction!)        │
│ KV cache: Loaded, converted, USED for attention             │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites & Dependencies

### Must Complete First
- [x] Phase 1: gRPC Server MVP (client-server communication works)

### Blockers
| Blocker | Resolution |
|---------|------------|
| Per-layer KV format unknown | Inspect model.config.num_hidden_layers |
| Storage format breaking change | Versioned format header in packed bytes |

---

## Task List

### Task 1: Redesign KV Cache Storage Format

**File:** `infemeral/tensors.py`
**Estimated Lines:** ~60 new/modified

**Current Format (Broken):**
```
[4B: key_len][4B: ndim][shape...][key_bytes][value_bytes]
```
- Stores ONE key tensor, ONE value tensor
- No layer information

**New Format:**
```
[1B: version = 0x02]
[4B: num_layers (uint32)]
For each layer:
  [4B: key_len][4B: ndim][shape...][key_bytes]
  [4B: val_len][4B: ndim][shape...][value_bytes]
```

**Subtasks:**
1. Add `pack_kv_cache_v2(kv_tuples: tuple[tuple[Tensor, Tensor], ...]) -> bytes`
   - Input: `((k₀, v₀), (k₁, v₁), ..., (k₃₁, v₃₁))`
   - Output: Versioned binary format with all layers
2. Add `unpack_kv_cache_v2(data: bytes) -> tuple[tuple[Tensor, Tensor], ...]`
   - Input: Versioned binary format
   - Output: Per-layer tuple structure
3. Keep old `pack_kv_cache()` and `unpack_kv_cache()` for backward compatibility
4. Add version detection in `unpack_kv_cache()` to auto-select parser

**Backward Compatibility:**
- Version 0x01 (implicit): Old flat format
- Version 0x02: New per-layer format
- `unpack_kv_cache()` checks first byte, delegates to correct parser

---

### Task 2: Fix Server KV Cache Loading (Line 545)

**File:** `infemeral/server.py`
**Location:** Lines 542-555

**Current Code:**
```python
kv_cache = load_kv_cache(session_id, session_key, str(device))
past_key_values = None
if kv_cache is not None:
    # TODO: Convert to proper past_key_values format
    pass  # ← THE GAP
```

**Fixed Code:**
```python
kv_cache = load_kv_cache(session_id, session_key, str(device))
past_key_values = None
if kv_cache is not None:
    # kv_cache is now tuple of (k, v) per layer from unpack_kv_cache_v2
    past_key_values = kv_cache  # Direct assignment, format matches
```

**Subtasks:**
1. Modify `load_kv_cache()` to use `unpack_kv_cache_v2()`
2. Return type changes from `tuple[Tensor, Tensor] | None` to `tuple[tuple[Tensor, Tensor], ...] | None`
3. Update type hints
4. Add validation: check layer count matches model.config.num_hidden_layers

---

### Task 3: Fix Server KV Cache Saving (Line 554)

**File:** `infemeral/server.py`
**Location:** Lines 554-560

**Current Code (Broken):**
```python
# Line 554 area - currently doesn't save properly
# new_kv is tuple of tuples, but save_kv_cache expects flat tensors
```

**Fixed Code:**
```python
if new_kv:
    save_kv_cache(session_id, new_kv, session_key)
```

**Subtasks:**
1. Modify `save_kv_cache(session_id, kv_tuples, session_key)` signature
   - Change from `(session_id, keys, values, session_key)`
   - To `(session_id, kv_tuples, session_key)` where kv_tuples is per-layer
2. Use `pack_kv_cache_v2()` inside `save_kv_cache()`
3. Update all callers (only handler/Infer uses it)

---

### Task 4: Optimize Client for Incremental Tokens

**File:** `infemeral/client.py`
**Location:** Lines 196-221 (generate loop)

**Current Code (Inefficient):**
```python
for _ in range(max_new_tokens):
    hidden = self.embedding.embed(generated_ids)  # ← Full sequence!
    cloaked = cloak(hidden, self.cloaking_ctx)
    server_output = self._call_server(cloaked)    # ← Sends everything
    # ...
```

**Fixed Code:**
```python
# First token: send full prompt
hidden = self.embedding.embed(generated_ids)
cloaked = cloak(hidden, self.cloaking_ctx)
server_output = self._call_server(cloaked, is_prompt=True)

for _ in range(max_new_tokens - 1):
    # Subsequent tokens: send only new token
    last_token = generated_ids[:, -1:]
    hidden = self.embedding.embed(last_token)     # ← Just 1 token!
    cloaked = cloak(hidden, self.cloaking_ctx)
    server_output = self._call_server(cloaked, is_prompt=False)
    # ...
```

**Subtasks:**
1. Split generate loop into prompt phase and generation phase
2. Prompt phase: send full sequence, server builds initial KV cache
3. Generation phase: send only new token embedding
4. Server uses session_id to load KV cache for context
5. Track position offset in client for proper attention positioning

**Proto Change (Optional):**
Add `position_offset` field to `InferenceRequest`:
```protobuf
message InferenceRequest {
    // ... existing fields ...
    int32 position_offset = 9;  // For incremental generation
}
```

---

### Task 5: Handle Position Embeddings Correctly

**File:** `infemeral/server.py`
**Location:** `forward_transformer()` rotary embedding computation

**Issue:** When processing a single new token with KV cache, position embeddings must account for cached sequence length.

**Current Code (Partial):**
```python
# Lines 350-380: Rotary embedding computation
position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
```

**Fixed Code:**
```python
# Account for cached sequence length
cached_seq_len = past_key_values[0][0].shape[2] if past_key_values else 0
position_ids = torch.arange(
    cached_seq_len,
    cached_seq_len + seq_len,
    device=device
).unsqueeze(0)
```

**Subtasks:**
1. Compute position offset from KV cache shape
2. Pass correct position_ids to rotary embedding
3. Test with single-token generation after long prompt

---

### Task 6: Add Context Length Windowing

**File:** `infemeral/server.py`
**Location:** After KV cache loading

**Issue:** KV cache grows unbounded. Need to implement sliding window or tide-windowing for contexts > max_context_length.

**Strategy: Attention Sink + Sliding Window**
```
[sink tokens: 4] [... window tokens: max_context - 4 ...]
     ↑                           ↑
  Always keep          Sliding window of recent tokens
```

**Subtasks:**
1. Check total KV length after loading cache
2. If exceeds `server_settings.max_context_length`:
   - Keep first `attention_sink_tokens` (default: 4)
   - Keep last `max_context_length - attention_sink_tokens` tokens
   - Discard middle tokens from KV cache
3. Update position embeddings accordingly
4. Log warning when windowing occurs

**Config Used:**
```python
# Already in config.py
max_context_length: int = 2048
attention_sink_tokens: int = 4
```

---

### Task 7: Multi-Turn Integration Tests

**File:** `tests/test_multi_turn.py` (new)
**Estimated Lines:** ~150

**Test Cases:**

```python
def test_kv_cache_format_roundtrip():
    """pack_kv_cache_v2 → unpack_kv_cache_v2 preserves structure."""

def test_kv_cache_backward_compatibility():
    """Old format files still loadable."""

def test_multi_token_generation():
    """Generate 10 tokens, verify KV cache grows correctly."""

def test_incremental_vs_full_equivalence():
    """Output identical whether sending full seq or incremental."""

def test_context_windowing():
    """Sequences > max_context_length handled gracefully."""

def test_position_embeddings_incremental():
    """Position IDs correct after KV cache loaded."""

def test_session_isolation_multi_turn():
    """Different sessions don't share KV cache state."""
```

---

### Task 8: Session Cleanup (Deferred Partial Implementation)

**File:** `infemeral/server.py`

**MVP Cleanup:** Delete KV cache files older than 1 hour on server startup.

```python
def cleanup_old_sessions(max_age_seconds: int = 3600):
    """Delete KV cache files older than max_age."""
    kv_dir = Path(server_settings.kv_cache_dir)
    now = time.time()
    for f in kv_dir.glob("*.bin"):
        if now - f.stat().st_mtime > max_age_seconds:
            f.unlink()
```

**Subtasks:**
1. Add `cleanup_old_sessions()` function
2. Call at server startup in `serve_grpc()`
3. Log number of sessions cleaned

**Full Session Management (Deferred to Phase 3):**
- Redis-based session store
- Explicit session expiration API
- Session metadata (created_at, last_accessed)

---

## Impacted Files

| File | Change Type | Description |
|------|-------------|-------------|
| `infemeral/tensors.py` | Modified | Add `pack_kv_cache_v2()`, `unpack_kv_cache_v2()` |
| `infemeral/server.py` | Modified | Fix lines 545, 554; add windowing; position fix |
| `infemeral/client.py` | Modified | Incremental token sending in generate loop |
| `tensor_service.proto` | Modified (optional) | Add position_offset field |
| `tests/test_multi_turn.py` | New | Multi-turn integration tests |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Format change breaks existing KV files | Low | Version header + backward compat in unpack |
| Position embedding off-by-one | High | Extensive testing with known outputs |
| Memory spike during format conversion | Medium | Stream-process layers, don't load all at once |
| Windowing loses important context | Medium | Keep sink tokens; document limitation |
| Client/server KV length mismatch | High | Validate shapes match before forward pass |

---

## Success Criteria

### Definition of Done
- [ ] `pack_kv_cache_v2()` and `unpack_kv_cache_v2()` roundtrip correctly
- [ ] Server loads KV cache and passes to `forward_transformer()` (not None)
- [ ] Client sends only new token after initial prompt
- [ ] 50-token generation uses ~50x less bandwidth than current
- [ ] Position embeddings correct for incremental tokens
- [ ] Context > 2048 tokens handled via windowing
- [ ] All tests in `test_multi_turn.py` pass

### Validation Steps
1. Generate 10 tokens with full sequence (baseline)
2. Generate 10 tokens with incremental (new implementation)
3. Compare outputs - must be identical
4. Measure bandwidth: incremental should be ~10x less
5. Test with 3000-token context, verify windowing kicks in
6. Multi-session test: two clients don't interfere

---

## Architecture After Phase 2

```
┌─────────────────────────────────────────────────────────────────┐
│                      Client (Incremental)                        │
├─────────────────────────────────────────────────────────────────┤
│  PROMPT PHASE:                                                   │
│  1. Embed full prompt [1, N, 4096]                               │
│  2. Cloak + encrypt                                              │
│  3. Send to server (position_offset=0)                           │
│                                                                  │
│  GENERATION PHASE (per token):                                   │
│  1. Embed new token only [1, 1, 4096]                            │
│  2. Cloak + encrypt                                              │
│  3. Send to server (position_offset=N+i)                         │
│  4. Receive output, sample next token                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Server (KV Cache Aware)                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Load KV cache by session_id                                  │
│     → unpack_kv_cache_v2() → ((k₀,v₀), ..., (k₃₁,v₃₁))          │
│  2. Apply windowing if > max_context_length                      │
│  3. Compute position_ids from cache length + offset              │
│  4. forward_transformer(hidden, past_key_values=kv_cache)        │
│  5. Append new KV to cache                                       │
│  6. save_kv_cache() with pack_kv_cache_v2()                      │
│  7. Return output tensor                                         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      KV Cache File (v2 Format)                   │
├─────────────────────────────────────────────────────────────────┤
│  [version: 0x02]                                                 │
│  [num_layers: 32]                                                │
│  [layer 0: k₀ shape + data, v₀ shape + data]                     │
│  [layer 1: k₁ shape + data, v₁ shape + data]                     │
│  ...                                                             │
│  [layer 31: k₃₁ shape + data, v₃₁ shape + data]                  │
│                                                                  │
│  Encrypted with AES-256-GCM using session_key                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Out of Scope (Deferred to Phase 3)

- RSA-OAEP session key wrapping
- TLS transport security
- Redis-based session store (instead of files)
- Explicit session expiration API
- Health check / readiness endpoints
- Metrics and observability
- gRPC-Web for browser support

---

## Estimated Effort

| Task | Complexity | Notes |
|------|------------|-------|
| Task 1: KV format redesign | Medium | Core change, needs careful testing |
| Task 2: Fix load (line 545) | Low | Simple once format is right |
| Task 3: Fix save (line 554) | Low | Signature change + call v2 |
| Task 4: Client incremental | Medium | Loop restructure, position tracking |
| Task 5: Position embeddings | Medium | Easy to get off-by-one |
| Task 6: Context windowing | Medium | Sink tokens + sliding window |
| Task 7: Integration tests | Medium | Many edge cases |
| Task 8: Session cleanup | Low | Simple file age check |

---

## Dependency Graph

```
Task 1 (KV format)
    ↓
Task 2 (load fix) ←──────┐
    ↓                    │
Task 3 (save fix)        │
    ↓                    │
Task 5 (position fix) ───┤
    ↓                    │
Task 6 (windowing)       │
    ↓                    │
Task 4 (client incremental)
    ↓
Task 7 (tests)
    ↓
Task 8 (cleanup)
```

**Critical Path:** Tasks 1 → 2 → 3 → 5 → 4 → 7
