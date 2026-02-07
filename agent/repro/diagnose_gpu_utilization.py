#!/usr/bin/env python3
"""Diagnostic script to identify GPU underutilization root cause.

This script instruments the server-side code to measure:
1. Actual GPU kernel execution time (via CUDA events)
2. Tensor device placement verification
3. CPU vs GPU time breakdown
4. VRAM growth across iterations

Run this script ON THE SERVER POD to diagnose the issue.

Usage:
    # SSH to pod then:
    python /workspace/infemeral-src/agent/repro/diagnose_gpu_utilization.py
"""

import gc
import sys
import time
from pathlib import Path

# Ensure infemeral is importable
sys.path.insert(0, "/workspace/infemeral-src")

import torch

# Check CUDA availability first
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. This script must run on a GPU pod.")
    sys.exit(1)

print("=" * 70)
print("GPU UNDERUTILIZATION DIAGNOSTIC")
print("=" * 70)

# GPU info
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

# Initial memory state
torch.cuda.reset_peak_memory_stats()
initial_mem = torch.cuda.memory_allocated() / 1024**2
print(f"Initial VRAM usage: {initial_mem:.1f} MB")


def measure_gpu_kernel_time(func, *args, **kwargs):
    """Measure actual GPU kernel execution time using CUDA events."""
    # Ensure all prior work is done
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    result = func(*args, **kwargs)
    end.record()

    # Wait for GPU work to complete
    torch.cuda.synchronize()

    gpu_time_ms = start.elapsed_time(end)
    return result, gpu_time_ms


def verify_tensor_devices(model):
    """Verify all model parameters are on GPU."""
    cpu_params = []
    gpu_params = []

    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            cpu_params.append(name)
        else:
            gpu_params.append(name)

    print(f"\nModel device placement:")
    print(f"  GPU parameters: {len(gpu_params)}")
    print(f"  CPU parameters: {len(cpu_params)}")

    if cpu_params:
        print(f"\n  WARNING: {len(cpu_params)} parameters on CPU!")
        for name in cpu_params[:5]:  # Show first 5
            print(f"    - {name}")
        if len(cpu_params) > 5:
            print(f"    ... and {len(cpu_params) - 5} more")
        return False
    return True


def run_diagnostic():
    """Main diagnostic routine."""
    from infemeral.server import load_model, forward_transformer

    print("\n" + "-" * 70)
    print("STEP 1: Loading model")
    print("-" * 70)

    load_start = time.perf_counter()
    model = load_model(device="cuda")
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    post_load_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"VRAM after load: {post_load_mem:.1f} MB")

    # Verify device placement
    print("\n" + "-" * 70)
    print("STEP 2: Verifying tensor device placement")
    print("-" * 70)

    all_on_gpu = verify_tensor_devices(model)

    # Check model config
    if hasattr(model, 'config'):
        print(f"\nModel config:")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Num layers: {model.config.num_hidden_layers}")
        print(f"  Num heads: {model.config.num_attention_heads}")
        print(f"  Vocab size: {model.config.vocab_size}")

    # Create test input
    print("\n" + "-" * 70)
    print("STEP 3: Timing forward passes")
    print("-" * 70)

    batch_size = 1
    seq_len = 32  # Reasonable prompt length
    hidden_size = model.config.hidden_size

    # Create input on GPU
    test_input = torch.randn(
        batch_size, seq_len, hidden_size,
        dtype=torch.float16,
        device="cuda"
    )

    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test input device: {test_input.device}")
    print(f"Test input dtype: {test_input.dtype}")

    # Warmup
    print("\nWarmup pass...")
    with torch.no_grad():
        output, kv_cache = forward_transformer(model, test_input, past_key_values=None)
    torch.cuda.synchronize()

    warmup_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"VRAM after warmup: {warmup_mem:.1f} MB")

    # Timed passes - measure both wall clock and GPU kernel time
    print("\nTimed passes (3 iterations):")
    print(f"{'Run':<6} {'Wall (ms)':>12} {'GPU (ms)':>12} {'VRAM (MB)':>12} {'KV Len':>10}")
    print("-" * 58)

    results = []
    prev_kv = kv_cache

    for i in range(3):
        # Clear caches
        gc.collect()
        torch.cuda.empty_cache()

        # Single token input (autoregressive mode)
        single_token_input = torch.randn(
            batch_size, 1, hidden_size,
            dtype=torch.float16,
            device="cuda"
        )

        # Wall clock timing
        wall_start = time.perf_counter()

        # GPU kernel timing
        torch.cuda.synchronize()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)

        gpu_start.record()
        with torch.no_grad():
            output, new_kv = forward_transformer(model, single_token_input, past_key_values=prev_kv)
        gpu_end.record()

        torch.cuda.synchronize()
        wall_time_ms = (time.perf_counter() - wall_start) * 1000
        gpu_time_ms = gpu_start.elapsed_time(gpu_end)

        vram_mb = torch.cuda.memory_allocated() / 1024**2
        kv_len = new_kv[0][0].shape[2] if new_kv else 0

        print(f"{i+1:<6} {wall_time_ms:>12.2f} {gpu_time_ms:>12.2f} {vram_mb:>12.1f} {kv_len:>10}")

        results.append({
            "wall_ms": wall_time_ms,
            "gpu_ms": gpu_time_ms,
            "vram_mb": vram_mb,
            "kv_len": kv_len,
        })

        prev_kv = new_kv

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    avg_wall = sum(r["wall_ms"] for r in results) / len(results)
    avg_gpu = sum(r["gpu_ms"] for r in results) / len(results)
    cpu_overhead = avg_wall - avg_gpu
    gpu_fraction = (avg_gpu / avg_wall) * 100 if avg_wall > 0 else 0

    print(f"\nTiming breakdown:")
    print(f"  Average wall clock:  {avg_wall:.2f} ms")
    print(f"  Average GPU kernel:  {avg_gpu:.2f} ms")
    print(f"  CPU/Overhead:        {cpu_overhead:.2f} ms ({100 - gpu_fraction:.1f}%)")
    print(f"  GPU fraction:        {gpu_fraction:.1f}%")

    # Memory analysis
    vram_growth = results[-1]["vram_mb"] - results[0]["vram_mb"]
    print(f"\nMemory analysis:")
    print(f"  VRAM growth over {len(results)} runs: {vram_growth:.1f} MB")
    if vram_growth > 100:
        print("  WARNING: Significant VRAM growth detected - possible memory leak!")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    issues = []

    if not all_on_gpu:
        issues.append("CRITICAL: Model parameters on CPU - GPU cannot be utilized")

    if gpu_fraction < 20:
        issues.append(f"SEVERE: Only {gpu_fraction:.1f}% of time spent on GPU kernels")
        issues.append("  -> Likely cause: CPU-bound operations or excessive synchronization")

    if avg_gpu < 1.0:
        issues.append(f"SEVERE: GPU kernel time {avg_gpu:.2f}ms is suspiciously low")
        issues.append("  -> Likely cause: Batch size 1 does not saturate GPU")

    if cpu_overhead > 500:
        issues.append(f"HIGH: {cpu_overhead:.1f}ms CPU overhead per token")
        issues.append("  -> Likely causes: gRPC serialization, KV cache management, Python GIL")

    if vram_growth > 100:
        issues.append(f"MODERATE: {vram_growth:.1f} MB VRAM growth - possible memory leak")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\nNo obvious issues detected. GPU utilization may be inherent to batch size 1.")

    print("\n" + "=" * 70)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 70)

    if gpu_fraction < 50:
        print("""
1. The forward pass completes too quickly for batch_size=1 to saturate GPU
2. Consider:
   - Continuous batching (batch multiple requests)
   - Speculative decoding
   - Profile with torch.profiler for detailed kernel breakdown

3. If CPU overhead is high, investigate:
   - gRPC serialization/deserialization
   - KV cache disk I/O (ensure kv_cache_mode='memory')
   - Encryption/decryption overhead
""")

    return results


if __name__ == "__main__":
    try:
        results = run_diagnostic()
        print("\nDiagnostic complete. Results saved.")
    except Exception as e:
        import traceback
        print(f"\nDiagnostic failed: {e}")
        traceback.print_exc()
        sys.exit(1)
