# Plan: Purge Cloaking Mechanism

**Created**: 2026-01-31_15-07-56
**Status**: Ready for Approval
**Category**: Refactoring / Code Cleanup

---

## Clarification Questions

None. Requirements are clear:
1. Remove all cloaking-related code from the codebase
2. Preserve working inference functionality (tested working without cloaking)
3. Do NOT commit changes until verified

---

## Summary

Remove the broken cloaking mechanism (orthogonal matrix rotation + DP noise) from the Infemeral codebase. The cloaking approach has been proven mathematically incompatible with transformer architecture (LayerNorm, softmax, and activations are not equivariant to orthogonal rotations). Inference works correctly when cloaking is disabled.

**Key Principle**: The removal should be surgical - only cloaking-specific code is removed. AES-256-GCM encryption (which protects data in transit and KV cache at rest) is **retained**.

---

## Dependency Mapping

### Prerequisites (None - self-contained refactor)

The codebase already works without cloaking (tested per `agent/issues/cloaking-breaks-inference-2026-01-19_23-53-11.md`).

### Blockers

- **None identified**: All changes are internal code removals with no external dependencies.

### Execution Order

Files must be modified in this order due to import dependencies:

1. `infemeral/crypto.py` - Remove cloaking primitives (other modules import from here)
2. `infemeral/config.py` - Remove DP-related settings
3. `infemeral/client.py` - Remove cloaking usage
4. `tests/test_crypto.py` - Remove cloaking tests
5. `tests/test_client.py` - Remove cloaking-related tests
6. `tests/conftest.py` - Remove cloaking fixtures
7. `scripts/benchmark_client.py` - Remove cloak/uncloak timing
8. Documentation updates (primer.md, SYSTEM_RUNDOWN.md)

---

## Phase 1: MVP/Foundational - Core Code Removal

### Task 1.1: Clean `infemeral/crypto.py`

**Remove**:
- `CloakingContext` dataclass (lines 19-27)
- `generate_orthogonal_matrix()` function (lines 30-57)
- `compute_dp_sigma()` function (lines 60-73)
- `create_cloaking_context()` function (lines 76-109)
- `cloak()` function (lines 112-142)
- `uncloak()` function (lines 145-167)
- Related imports: `numpy as np`, parts of `torch`

**Retain**:
- `generate_session_key()` (line 170-172)
- `encrypt_bytes()` (lines 175-188)
- `decrypt_bytes()` (lines 191-203)
- Required imports for AES-GCM

**Impacted Lines**: ~150 lines removed

### Task 1.2: Clean `infemeral/config.py`

**Remove from `CryptoSettings`**:
- `hidden_dim` field (line 12)
- `dp_epsilon` field (line 13)
- `dp_delta` field (line 14)

**Note**: The `CryptoSettings` class may become empty or can be removed entirely if no crypto settings remain. Consider renaming to reflect actual purpose (session key generation only).

**Alternative**: Remove the entire `CryptoSettings` class and `crypto_settings` singleton since no configuration is needed for AES key generation.

**Impacted Lines**: ~8 lines removed

### Task 1.3: Clean `infemeral/client.py`

**Remove**:
- Import of `cloak`, `create_cloaking_context`, `uncloak` (lines 46-53)
- Import of `crypto_settings` if `CryptoSettings` removed (line 18)
- `TokenTiming` fields: `cloak_ms`, `uncloak_ms` (lines 26, 28)
- `self.cloaking_ctx` initialization in `__init__` (lines 149-154)
- All `cloak()` calls in `generate()` (lines 296, 323, 365)
- All `uncloak()` calls in `generate()` (lines 298, 325, 375)
- Timing instrumentation for cloak/uncloak in `_generate_token()` (lines 364-366, 374-376)
- Remove `cloaked`/`uncloaked` from intermediates dict (lines 391-392)

**Modify**:
- Rename `cloaked` variable to `hidden` (data flow: `hidden` → server → `server_output`)
- Update `_call_server()` to accept `hidden` instead of `cloaked`
- Remove `cloaked`/`uncloaked` from metrics timing breakdown

**Impact**: The client now sends raw embeddings (encrypted with AES-256-GCM) directly to the server without rotation/noise.

### Task 1.4: Update `print_metrics()` in `client.py`

**Remove**:
- "cloak" and "uncloak" from phases list (line 473)

---

## Phase 2: Test Updates

### Task 2.1: Clean `tests/test_crypto.py`

**Remove entire test classes**:
- `TestOrthogonalMatrix` (lines 20-58)
- `TestDifferentialPrivacy` (lines 61-79)
- `TestCloaking` (lines 82-127)
- `TestCloakingEdgeCases` (lines 218-261)
- `TestSecurityProperties` (lines 264-308)

**Remove imports**:
- `CloakingContext`
- `cloak`, `uncloak`
- `compute_dp_sigma`
- `create_cloaking_context`
- `generate_orthogonal_matrix`

**Retain**:
- `TestEncryption` (lines 130-215) - AES-256-GCM tests

**Impacted Lines**: ~200 lines removed

### Task 2.2: Clean `tests/test_client.py`

**Remove**:
- Import of `create_cloaking_context` (line 8)
- `TestClientCloakingFlow` class (lines 181-216)
- `test_cloaking_context_created` test in `TestClientSession` (lines 172-178)

**Retain**:
- `TestEmbeddingLayer`
- `TestClientSampling`
- `TestClientSession` (minus cloaking test)
- `TestClientGrpcCalls`
- `TestClientEncryption`

### Task 2.3: Clean `tests/conftest.py`

**Remove**:
- Import of `create_cloaking_context` (line 10)
- `cloaking_context` fixture (lines 25-28)

**Retain**:
- `device` fixture
- `session_key` fixture
- `sample_hidden_states` fixtures
- `temp_weights_dir` fixture
- `mock_client_weights` fixture
- `mock_kv_cache` fixture
- pytest configuration hooks

### Task 2.4: Update other test files

**Files to check**:
- `tests/test_server.py` - May reference "cloaked" variable names
- `tests/test_e2e.py` - May have cloaking-related tests
- `tests/test_grpc_integration.py` - Variable naming
- `tests/test_client_perf.py` - Timing metrics

For each: rename `cloaked` variables to `hidden` if present, remove cloaking-specific tests.

---

## Phase 3: Benchmark & Documentation Updates

### Task 3.1: Update `scripts/benchmark_client.py`

**Remove from `BenchmarkResult` dataclass**:
- `cloak_p50`, `cloak_p95`, `cloak_p99` fields (lines 44-47)
- `uncloak_p50`, `uncloak_p95`, `uncloak_p99` fields (lines 51-54)

**Update**:
- `phases` list: remove "cloak" and "uncloak" (lines 183, 237, 258)

### Task 3.2: Update `agent/primer.md`

**Update sections**:
- Architecture Pattern: Remove "Cloak" and "Uncloak" from diagram
- Crypto section: Remove cloaking references
- Key Data Flow: Remove cloak/uncloak steps
- Configuration: Remove DP-related env vars

### Task 3.3: Update `SYSTEM_RUNDOWN.md`

**Major updates needed**:
- Architecture diagram: Remove cloaking step
- Security Architecture section: Remove cloaking subsection
- Processing Locations: Update client processing list
- Threat Model: Update mitigations list

### Task 3.4: Update `README.md`

Check for and remove any cloaking references in the public-facing README.

---

## Impacted Files Summary

| File | Action | Lines Changed (est.) |
|------|--------|---------------------|
| `infemeral/crypto.py` | Remove cloaking functions | -150 |
| `infemeral/config.py` | Remove DP settings | -8 |
| `infemeral/client.py` | Remove cloaking usage | -40 |
| `tests/test_crypto.py` | Remove cloaking tests | -200 |
| `tests/test_client.py` | Remove cloaking tests | -40 |
| `tests/conftest.py` | Remove cloaking fixture | -5 |
| `scripts/benchmark_client.py` | Remove cloak timing | -15 |
| `agent/primer.md` | Update documentation | ~30 |
| `SYSTEM_RUNDOWN.md` | Update documentation | ~50 |
| `README.md` | Update if needed | TBD |

**Total estimated**: ~500+ lines removed

---

## Risk Assessment

| Risk | Impact | Mitigation Strategy |
|------|--------|---------------------|
| Breaking inference after removal | High | Run local test with disabled cloaking first (already verified working) |
| Missing a cloaking reference | Medium | Use grep to find all `cloak` references before and after |
| Test failures from missing fixtures | Medium | Run pytest after each phase to catch immediately |
| Import errors from circular deps | Low | Follow execution order (crypto.py first) |
| Documentation drift | Low | Update docs in same PR for consistency |

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `infemeral/crypto.py` exports only: `generate_session_key`, `encrypt_bytes`, `decrypt_bytes`
- [ ] `infemeral/client.py` has no imports of `cloak`, `uncloak`, `create_cloaking_context`
- [ ] `infemeral/config.py` has no DP-related settings
- [ ] `grep -r "cloak" infemeral/` returns no matches (excluding comments)

### Phase 2 Complete When:
- [ ] `pytest tests/test_crypto.py` passes with only encryption tests
- [ ] `pytest tests/test_client.py` passes without cloaking tests
- [ ] `pytest tests/` passes all remaining tests

### Phase 3 Complete When:
- [ ] `scripts/benchmark_client.py` runs without cloak timing
- [ ] Documentation reflects new architecture (no cloaking)

### Final Validation:
- [ ] `grep -rn "cloak" . --include="*.py" --include="*.md"` returns only historical references in `agent/issues/`
- [ ] Server can be started: `python -m infemeral.server --mode grpc`
- [ ] Client inference works end-to-end on RunPod

---

## Suggested Tests

After completing the refactor, the following tests should validate correctness:

### Unit Tests (already exist, verify they pass):
```bash
pytest tests/test_crypto.py::TestEncryption -v
pytest tests/test_tensors.py -v
pytest tests/test_config.py -v
```

### Integration Tests:
```bash
# Local (if server running):
pytest tests/test_grpc_integration.py -v

# Or manual test:
python -c "
from infemeral.client import Client
client = Client(weights_path='/workspace/weights/client_weights.safetensors', device='cpu')
result = client.generate('Hello', max_new_tokens=10)
print(f'Result: {repr(result)}')
client.close()
"
```

### Grep Validation:
```bash
# Should return empty (no cloaking code in source):
grep -rn "cloak" infemeral/ --include="*.py" | grep -v "# " | grep -v '"""'

# Should only show historical files:
grep -rn "cloak" . --include="*.py" --include="*.md"
```

---

## Rollback Plan

If issues arise:
1. All changes are in local files (not committed)
2. Use `git checkout -- <file>` to restore individual files
3. Use `git checkout -- .` to restore all files

---

## Notes for Implementer

1. **Do NOT commit** until user verifies on RunPod
2. **Preserve AES-256-GCM encryption** - only cloaking (orthogonal rotation + DP noise) is removed
3. The `cloaked_embedding` field name in protobuf can remain for backward compatibility (it just sends encrypted hidden states now)
4. After verification, use `/write-commits` to create atomic commits
