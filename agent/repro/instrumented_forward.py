#!/usr/bin/env python3
"""Instrumented forward pass to isolate GPU kernel time from CPU overhead.

This script patches the forward_transformer function with CUDA timing
to identify exactly where time is being spent.

Run ON THE SERVER POD:
    cd /workspace/infemeral-src
    python agent/repro/instrumented_forward.py
"""

import gc
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/infemeral-src")

import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)


class CUDATimer:
    """Context manager for GPU kernel timing."""

    def __init__(self, name: str, results: dict):
        self.name = name
        self.results = results
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        torch.cuda.synchronize()
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        elapsed = self.start.elapsed_time(self.end)
        if self.name not in self.results:
            self.results[self.name] = []
        self.results[self.name].append(elapsed)


def instrumented_forward_transformer(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    past_key_values: tuple | None = None,
) -> tuple[torch.Tensor, tuple, dict]:
    """Forward pass with detailed CUDA timing instrumentation.

    Returns:
        (hidden_states, new_kv_cache, timing_breakdown)
    """
    from transformers.cache_utils import DynamicCache
    from infemeral.config import server_settings

    timings: dict = {}

    # Suppress excessive logging
    logging.getLogger("infemeral.server").setLevel(logging.WARNING)

    # Get transformer layers
    if hasattr(model, "model"):
        transformer = model.model
        layers = transformer.layers
        norm = transformer.norm
    else:
        raise ValueError("Unsupported model architecture")

    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Time cache initialization
    with CUDATimer("cache_init", timings):
        cache = DynamicCache()
        past_len = 0
        if past_key_values is not None:
            for layer_idx, (k, v) in enumerate(past_key_values):
                # Skip .contiguous() if already contiguous
                if not k.is_contiguous():
                    k = k.contiguous()
                if not v.is_contiguous():
                    v = v.contiguous()
                cache.key_cache.append(k)
                cache.value_cache.append(v)
            past_len = past_key_values[0][0].shape[2]

    # Time position IDs and attention mask
    with CUDATimer("prep_masks", timings):
        if past_len > 0:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=device
            ).unsqueeze(0)
        else:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        total_len = past_len + seq_len
        causal_mask = torch.triu(
            torch.ones(seq_len, total_len, dtype=dtype, device=device) * float("-inf"),
            diagonal=past_len + 1,
        )
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    # Time each layer
    layer_times = []
    for i, layer in enumerate(layers):
        with CUDATimer(f"layer_{i}", timings):
            layer_attn = layer.self_attn
            layer_rotary_emb = getattr(layer_attn, "rotary_emb", None)
            if layer_rotary_emb is None:
                layer_rotary_emb = getattr(transformer, "rotary_emb", None)

            cos, sin = layer_rotary_emb(hidden_states, position_ids)
            layer_position_embeddings = (cos, sin)

            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
                position_embeddings=layer_position_embeddings,
            )

            if isinstance(layer_out, torch.Tensor):
                hidden_states = layer_out
            else:
                hidden_states = layer_out[0]
                if len(layer_out) > 1 and layer_out[1] is not None:
                    cache = layer_out[1]

            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

    # Time final norm
    with CUDATimer("final_norm", timings):
        hidden_states = norm(hidden_states)

    # Time cache extraction
    with CUDATimer("cache_extract", timings):
        new_key_values = []
        if hasattr(cache, 'key_cache') and len(cache.key_cache) > 0:
            for layer_idx in range(len(cache.key_cache)):
                k = cache.key_cache[layer_idx]
                v = cache.value_cache[layer_idx]
                new_key_values.append((k, v))

    return hidden_states, tuple(new_key_values) if new_key_values else (), timings


def run_instrumented_benchmark():
    """Run benchmark with instrumented forward pass."""
    from infemeral.server import load_model

    print("=" * 70)
    print("INSTRUMENTED FORWARD PASS BENCHMARK")
    print("=" * 70)

    print("\nLoading model...")
    model = load_model(device="cuda")
    config = model.config

    print(f"Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # Test inputs
    batch_size = 1
    hidden_size = config.hidden_size

    print("\n" + "-" * 70)
    print("BENCHMARK: Single token with growing KV cache")
    print("-" * 70)

    # First pass: prompt (simulate 32 tokens)
    print("\nPhase 1: Initial prompt (32 tokens)...")
    prompt_input = torch.randn(batch_size, 32, hidden_size, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        output, kv_cache, timing1 = instrumented_forward_transformer(model, prompt_input, None)

    # Print timing breakdown
    total_gpu = sum(sum(v) for v in timing1.values())
    print(f"\nPrompt phase GPU time: {total_gpu:.2f} ms")
    print(f"  Cache init: {sum(timing1.get('cache_init', [0])):.2f} ms")
    print(f"  Prep masks: {sum(timing1.get('prep_masks', [0])):.2f} ms")
    layer_total = sum(sum(timing1.get(f'layer_{i}', [0])) for i in range(config.num_hidden_layers))
    print(f"  All layers: {layer_total:.2f} ms ({layer_total/config.num_hidden_layers:.2f} ms/layer)")
    print(f"  Final norm: {sum(timing1.get('final_norm', [0])):.2f} ms")
    print(f"  Cache extract: {sum(timing1.get('cache_extract', [0])):.2f} ms")

    # Single token passes (autoregressive)
    print("\nPhase 2: Single token generation (5 iterations)...")
    print(f"{'Run':<6} {'GPU Total':>12} {'Layers':>12} {'Cache':>12} {'Overhead':>12}")
    print("-" * 56)

    prev_kv = kv_cache
    for i in range(5):
        gc.collect()
        torch.cuda.empty_cache()

        single_token = torch.randn(batch_size, 1, hidden_size, dtype=torch.float16, device="cuda")

        # Wall clock
        wall_start = time.perf_counter()
        with torch.no_grad():
            output, new_kv, timing = instrumented_forward_transformer(model, single_token, prev_kv)
        wall_ms = (time.perf_counter() - wall_start) * 1000

        gpu_total = sum(sum(v) for v in timing.values())
        layer_total = sum(sum(timing.get(f'layer_{j}', [0])) for j in range(config.num_hidden_layers))
        cache_time = sum(timing.get('cache_init', [0])) + sum(timing.get('cache_extract', [0]))
        overhead = wall_ms - gpu_total

        print(f"{i+1:<6} {gpu_total:>12.2f} {layer_total:>12.2f} {cache_time:>12.2f} {overhead:>12.2f}")

        prev_kv = new_kv

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("""
Key metrics to examine:

1. GPU Total vs Wall Clock:
   - If GPU Total << Wall Clock, significant CPU overhead exists
   - Overhead sources: logging, Python GIL, gRPC processing

2. Layer time vs expected:
   - AWQ INT4 on RTX 4090 should process ~50-100 tok/s for batch_size=1
   - Each layer should take ~0.3-0.6ms for single token
   - If layer time is much less, GPU may be starved (batch too small)

3. Cache overhead:
   - Cache init/extract should be <1ms total
   - If high, consider keeping cache in DynamicCache format

4. Overhead trend:
   - If overhead increases with KV cache size, memory copy is the issue
   - If constant, it's fixed CPU overhead (logging, gRPC)
""")


if __name__ == "__main__":
    run_instrumented_benchmark()
