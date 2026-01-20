# Debugging Snapshot: Cloaking Breaks Inference Output

**Created**: 2026-01-19_23-53-11
**Status**: Root Cause Identified
**Category**: Logic Error / Architectural Design Flaw

---

## Problem Statement

The cloaking mechanism in Infemeral produces garbled/gibberish output from the LLM, even when the underlying inference pipeline (transformer layers, KV cache) is working correctly.

### Symptoms

- **With cloaking enabled**: `'Hello yet_hooksデルollyAf createSelectorbaraaná barrier_best'`
- **With cloaking disabled**: `'Hello, how are you? I am doing well, thanks for asking...'` (coherent)

### Error Category

**Architectural Design Flaw**: The cloaking mechanism assumes transformer layers are equivariant to orthogonal rotations, but they are not.

---

## Evidence Collection

### Test Results

**Test 1: With cloaking (default behavior)**
```python
client = Client(weights_path='/workspace/weights/client_weights.safetensors', device='cpu')
result = client.generate('Hello', max_new_tokens=10)
# Result: 'Hello yet_hooksデルollyAf createSelectorbaraaná barrier_best'
```

**Test 2: Without cloaking (identity functions)**
```python
import infemeral.client as client_module
client_module.cloak = lambda x, ctx: x    # Identity
client_module.uncloak = lambda x, ctx: x  # Identity

result = client.generate('Hello, how are you?', max_new_tokens=30)
# Result: 'Hello, how are you? I am doing well, thanks for asking...'
```

### Files Modified During Session

1. `infemeral/server.py` - KV cache fixes (unrelated to cloaking issue)
2. `infemeral/tensors.py` - Added `.contiguous()` calls (unrelated)
3. `agent/primer.md` - Documentation updates

### Relevant Code

**Cloaking (`infemeral/crypto.py:112-142`)**:
```python
def cloak(hidden: torch.Tensor, ctx: CloakingContext, add_noise: bool = True) -> torch.Tensor:
    """Apply cloaking: DP noise + orthogonal rotation."""
    matrix = ctx.matrix
    if add_noise:
        noise = torch.randn_like(hidden) * ctx.sigma
        hidden = hidden + noise
    # Apply orthogonal rotation: x' = x @ M.T
    return torch.einsum("ij,...j->...i", matrix, hidden)

def uncloak(cloaked: torch.Tensor, ctx: CloakingContext) -> torch.Tensor:
    """Remove orthogonal rotation (DP noise cannot be removed)."""
    matrix_t = ctx.matrix_t
    # Apply inverse rotation: x = x' @ M
    return torch.einsum("ij,...j->...i", matrix_t, cloaked)
```

**Data Flow**:
```
Client: hidden = embed(tokens)
Client: cloaked = cloak(hidden)           # x' = M @ x + noise
Server: output = transformer(cloaked)     # f(M @ x + noise)
Client: uncloaked = uncloak(output)       # M.T @ f(M @ x + noise)
Client: logits = lm_head(uncloaked)
```

---

## Root Cause Analysis

### The Mathematical Problem

For cloaking to work, we need:
```
uncloak(transformer(cloak(x))) ≈ transformer(x)
```

This requires the transformer to be **equivariant** to orthogonal rotations. However:

| Operation | Equivariant to Rotation? | Reason |
|-----------|-------------------------|--------|
| Linear layer `Wx + b` | Partially (bias breaks it) | `W(Mx) + b ≠ M(Wx + b)` |
| LayerNorm | **No** | Mean/std computed per-vector changes |
| Attention softmax | **No** | `softmax(QK^T)` is nonlinear |
| GELU/SiLU activation | **No** | Element-wise nonlinearity |
| RMSNorm | **No** | Norm computation is not rotation-invariant |

### Why It Fails

1. **LayerNorm**: `LayerNorm(Mx) ≠ M * LayerNorm(x)` because the mean and standard deviation are computed over the rotated vector, producing different normalization.

2. **Attention**: `softmax((M @ Q)(M @ K)^T) = softmax(M @ Q @ K^T @ M^T)` - the softmax is applied to a differently-oriented matrix, changing attention patterns.

3. **Activations**: `GELU(Mx) ≠ M * GELU(x)` because GELU operates element-wise on the rotated coordinates.

### DP Noise Compounds the Problem

Even if rotation were somehow handled, the DP noise `ε` added during cloaking:
- Cannot be removed (by design for privacy)
- Accumulates through the network
- Further corrupts the output

---

## Dead Ends (Attempted Solutions)

These were explored during the debugging session but are **not solutions to the cloaking issue** (they fixed separate KV cache problems):

1. **KV cache parameter name fix**: Changed `past_key_values` to `past_key_value` (singular) - fixed cache population but not cloaking
2. **Eager attention implementation**: Avoided SDPA contiguity errors but not cloaking
3. **4D causal attention mask**: Fixed mask shape errors but not cloaking
4. **Direct cache list assignment**: Avoided `torch.cat()` non-contiguity but not cloaking

---

## Proposed Solutions

### Standard Approaches (Not Yet Attempted)

1. **Disable Cloaking for Development**
   - Add `INFEMERAL_DISABLE_CLOAKING=true` environment variable
   - Make cloak/uncloak identity functions when set
   - Allows functional testing while cloaking is redesigned

2. **Reduce DP Noise**
   - Set `dp_epsilon` to a very high value (e.g., 100) to minimize noise
   - Test if output improves (would confirm noise is a major factor)
   - Current: `dp_epsilon=2.0`, `dp_delta=1e-5`

3. **Verify Matrix Orthogonality**
   - Add assertions that `M @ M.T ≈ I`
   - Check if numerical precision in float16 causes drift
   - Log condition number of the matrix

### Novel Approaches (Outside-the-Box)

1. **Homomorphic Encryption Alternative**
   - Replace orthogonal cloaking with actual homomorphic encryption
   - Libraries: TenSEAL, Microsoft SEAL
   - Allows computation on encrypted data with mathematical guarantees
   - Trade-off: Significant performance overhead

2. **Split Model Architecture**
   - Instead of cloaking hidden states, split the model differently:
     - Client runs: embedding + first N layers + last M layers + lm_head
     - Server runs: middle layers only
   - Hidden states in middle layers are less interpretable
   - No cloaking needed; privacy via architecture
   - Similar to "split learning" in federated ML

3. **Cloaking-Aware Fine-Tuning**
   - Fine-tune the model to be more robust to rotations
   - Train with augmented data where inputs are randomly rotated
   - May improve tolerance but won't fully solve the mathematical issue

4. **Linear-Only Server Computation**
   - Modify server to only run linear projections (Q, K, V, O projections)
   - Move all nonlinear ops (LayerNorm, softmax, activations) to client
   - Cloaking would work for linear ops
   - Significant latency increase due to more round-trips

---

## Recommended Next Steps

### Immediate (Unblock Development)

```python
# In infemeral/config.py, add:
class CryptoSettings(BaseSettings):
    disable_cloaking: bool = Field(default=False, description="Disable cloaking for testing")

# In infemeral/client.py, modify generate():
if not crypto_settings.disable_cloaking:
    cloaked = cloak(hidden, self.cloaking_ctx)
else:
    cloaked = hidden  # Identity
```

### Medium-Term (Architectural Decision)

Evaluate trade-offs between:
1. **Homomorphic encryption**: Strong privacy, high latency
2. **Split architecture**: Moderate privacy, low latency
3. **Accept current design**: Document that cloaking provides obfuscation, not cryptographic security

### Long-Term (Research)

Investigate if there's a class of neural network architectures that ARE equivariant to orthogonal transformations (e.g., certain geometric deep learning models).

---

## Session Context

This issue was discovered while debugging what was initially thought to be a KV cache problem. The KV cache issues were fixed (see commits), but the garbled output persisted. Systematic testing with cloaking disabled revealed the true root cause.

**Related Issues**:
- `agent/issues/kv-cache-not-returned-2026-01-16_22-32-31.md` (now resolved)

**KV Cache Fixes Applied** (separate from cloaking):
- `past_key_value` parameter name (singular, not plural)
- Direct cache list assignment instead of `update()`
- Eager attention implementation
- 4D causal attention mask
