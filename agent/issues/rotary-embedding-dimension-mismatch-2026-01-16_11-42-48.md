# Debugging Snapshot: Rotary Embedding Dimension Mismatch

**Issue ID**: rotary-embedding-dimension-mismatch
**Timestamp**: 2026-01-16_11-42-48
**Status**: RESOLVED

---

## Error Summary

```
RuntimeError: The size of tensor a (32) must match the size of tensor b (128) at non-singleton dimension 3
```

**Location**: `transformers/models/llama/modeling_llama.py:138` in `apply_rotary_pos_emb`

**Stack Trace Path**:
```
handler() → forward_transformer() → layer() → LlamaDecoderLayer.forward()
→ LlamaAttention.forward() → apply_rotary_pos_emb()
```

---

## Environment

| Component | Version/Value |
|-----------|---------------|
| transformers | 4.57.6 |
| Model | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 |
| Quantization | AWQ 4-bit (GEMM backend) |
| hidden_size | 4096 |
| num_attention_heads | 32 |
| head_dim | 128 |
| Platform | RunPod Serverless |

---

## Files Modified During Session

1. `infemeral/server.py` - Main debugging target
2. `infemeral/config.py` - Updated weights_path → weights_dir
3. `infemeral/model_prep.py` - Rewrote model download/preparation
4. `tests/test_config.py` - Updated for new config fields

---

## Dead Ends (Attempted Solutions That Failed)

### 1. Letting Layer Compute Own Position Embeddings
- **Hypothesis**: Pass `position_embeddings=None` and let the layer compute internally
- **Result**: Failed - transformers 4.46+ requires position_embeddings parameter

### 2. Slicing Rotary Embeddings to Match Query
- **Hypothesis**: Query had head_dim=32, so slice cos/sin to 32
- **Result**: Failed - Query actually had correct head_dim=128; problem was elsewhere

### 3. Checking Model Config Mismatch
- **Hypothesis**: Wrong model loaded (smaller model with different head_dim)
- **Result**: Failed - Config confirmed correct: hidden_size=4096, num_heads=32, head_dim=128

### 4. Inspecting AWQ Layer Attributes
- **Hypothesis**: AWQ WQLinear_GEMM was outputting wrong dimensions
- **Result**: Failed - q_proj output was correct [1, 1, 4096]

---

## Root Cause Analysis

**Category**: Tensor Shape Mutation (Squeeze Operation)

### The Problem

The `LlamaDecoderLayer` output was being squeezed from 3D to 2D when `seq_len=1`:

| Layer | Input Shape | Output Shape |
|-------|-------------|--------------|
| Layer 0 | `[1, 1, 4096]` | `[1, 4096]` ← squeezed! |
| Layer 1 | `[1, 4096]` | CRASH |

When Layer 1 received a 2D tensor `[1, 4096]`, the internal reshape logic:
```python
hidden_shape = (*input_shape, -1, self.head_dim)  # (1, -1, 128)
query_states = self.q_proj(hidden_states).view(hidden_shape)  # [1, 32, 128]
query_states = query_states.transpose(1, 2)  # [1, 128, 32] - WRONG!
```

Expected 4D: `[batch, num_heads, seq_len, head_dim]` = `[1, 32, 1, 128]`
Actual 3D: `[1, 128, 32]`

The rotary embeddings `cos.unsqueeze(1)` = `[1, 1, 1, 128]` couldn't broadcast to a 3D tensor.

---

## Solution

**Fix Applied**: Restore sequence dimension after each layer if squeezed

```python
# In forward_transformer(), after layer() call:
hidden_states = layer_outputs[0]

# CRITICAL FIX: Ensure hidden_states maintains 3D shape [batch, seq_len, hidden_dim]
if hidden_states.dim() == 2:
    hidden_states = hidden_states.unsqueeze(1)  # [batch, 1, hidden_dim]
```

**Why It Works**: The unsqueeze restores the expected `[batch, seq_len, hidden_dim]` shape, allowing subsequent layers to correctly reshape query/key/value tensors to 4D.

---

## Debugging Timeline

1. **Initial Error**: 32 vs 128 dimension mismatch at apply_rotary_pos_emb
2. **Added Debug Prints**: Verified model config (correct head_dim=128)
3. **Traced q_proj Output**: Confirmed correct [1, 1, 4096] shape
4. **Simulated Reshape**: Showed Layer 0 correct, Layer 1 wrong
5. **Key Discovery**: Layer 1 input was 2D [1, 4096] instead of 3D [1, 1, 4096]
6. **Root Cause**: Layer output squeeze when seq_len=1
7. **Fix Applied**: unsqueeze(1) after each layer if dim==2

---

## Verification Steps

1. Deploy updated `server.py` to RunPod
2. Send test cloaked embedding through handler
3. Verify all 32 layers process without dimension errors
4. Confirm output shape matches input shape (minus embedding operations)

---

## Lessons Learned

1. **Tensor shape mutations** can occur silently in optimized inference code paths
2. **Single-token sequences** (seq_len=1) are edge cases where squeeze optimizations trigger
3. **Debug prints at layer boundaries** are essential for tracing shape mutations
4. **The error location** (apply_rotary_pos_emb) was 3 levels removed from the actual bug

---

## Related Issues

- [transformers #32582](https://github.com/huggingface/transformers/issues/32582) - apply_rotary_pos_emb tensor mismatch
- [transformers #35376](https://github.com/huggingface/transformers/pull/35376) - RoPE embedding fixes
- [llm-compressor #1762](https://github.com/vllm-project/llm-compressor/issues/1762) - AWQ dimension mismatch
