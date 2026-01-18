# Issue: KV Cache Not Returned by Transformer Layers

**Created**: 2026-01-16 22:32:31
**Status**: Open
**Category**: API Compatibility / Logic Error
**Severity**: High (causes garbled output)

---

## Problem Statement

The transformer forward pass completes successfully, but the **KV cache is not being captured or returned** by the decoder layers. This results in:
1. Each token generation having no context from previous tokens
2. Garbled/incoherent output: `'Hello��dup_CLASSESbeckdoor'`

### Evidence from Logs

```
2026-01-17 00:21:35,236 - __main__ - INFO - Layer 25 input - cache is None
2026-01-17 00:21:35,236 - __main__ - INFO - Layer 25: calling with hidden_states=torch.Size([1, 2, 4096]), cache=<class 'NoneType'>
...
2026-01-17 00:21:35,246 - __main__ - INFO - Layer 31 input - cache is None
```

All 32 layers show `cache is None` throughout the forward pass, meaning no KV cache is being accumulated or passed between layers.

### Key Deprecation Warning

```
FutureWarning: `past_key_value` is deprecated and will be removed in version 4.58
for `LlamaDecoderLayer.forward`. Use `past_key_values` instead.
```

This warning strongly suggests the API has changed from singular `past_key_value` to plural `past_key_values`.

---

## Root Cause Analysis

### Current Behavior

In `server.py:forward_transformer()`:
```python
layer_out = layer(
    hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=cache,  # <-- Singular, deprecated
    use_cache=True,
    position_embeddings=layer_position_embeddings,
)
```

The layer returns a **tensor directly** instead of a `(hidden_states, cache)` tuple:
```
Layer 0 output: type=<class 'torch.Tensor'>, len=1
```

### Hypothesis

The transformers library (version 4.x+) changed the KV cache API:
1. Parameter renamed from `past_key_value` (singular) to `past_key_values` (plural)
2. Cache format changed from tuple-per-layer to a unified `Cache` object
3. When using deprecated parameter, the layer may not return the cache

---

## Files Modified During Debugging Session

| File | Changes |
|------|---------|
| `infemeral/server.py` | Added DynamicCache conversion, fixed tensor vs tuple handling, fixed unsqueeze dimension, added debug logging |
| `infemeral/client.py` | Added `decrypt_bytes` import, added response decryption, fixed dtype conversion in `de_embed` |

---

## Dead Ends (Attempted Solutions That Failed)

1. **Converting tuple cache to DynamicCache before layer loop**
   - Result: Cache still None - layer doesn't return cache with deprecated parameter

2. **Checking `layer_out[1]` for cache**
   - Result: Layer returns tensor directly, not tuple, so indexing failed

3. **Using `cache.update()` to populate DynamicCache**
   - Result: Works for loading saved cache, but doesn't solve the issue of layers not returning new cache

---

## Proposed Solutions (Not Yet Attempted)

### Standard Troubleshooting

1. **Change parameter name from `past_key_value` to `past_key_values`**
   ```python
   layer_out = layer(
       hidden_states,
       past_key_values=cache,  # <-- Plural
       use_cache=True,
       ...
   )
   ```

2. **Use the model's native `generate()` or `forward()` method** instead of calling layers individually
   - The model-level forward handles cache management internally

3. **Check transformers version and consult migration guide**
   ```bash
   pip show transformers  # Check version
   ```
   - Review changelog for cache API changes between versions

### Novel/Outside-the-Box Avenues

1. **Monkey-patch the layer's forward to capture cache**
   - Wrap each layer's forward method to intercept and store KV tensors
   - Use PyTorch hooks (`register_forward_hook`) to capture attention K/V tensors

2. **Use `model.model.forward()` with `output_hidden_states=True` and `use_cache=True`**
   - Instead of iterating layers manually, call the transformer stack as a unit
   - This may require refactoring how cloaked embeddings are injected

---

## Environment Context

- **Transformers version**: Unknown (need to verify on RunPod)
- **Model**: Llama-based (32 layers, hidden_size=4096)
- **Platform**: RunPod serverless (CUDA)

---

## Next Steps

1. Verify transformers version on RunPod: `pip show transformers`
2. Try changing `past_key_value` → `past_key_values` parameter name
3. If that fails, investigate using `register_forward_hook` to capture KV tensors
4. Consider using model-level forward instead of layer-by-layer iteration

---

## Related Links

- [Transformers Cache Documentation](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.Cache)
- [DynamicCache API](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache)
