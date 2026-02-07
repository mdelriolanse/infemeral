# GPU Utilization Benchmark Report

**Date**: 2026-01-31
**Pod**: 203.57.40.175:10271
**GPU**: NVIDIA RTX 4090 (24GB GDDR6X)
**Model**: AWQ INT4 8B Transformer

---

## Executive Summary

**CRITICAL: GPU is severely underutilized at 1.9% average utilization.**

The RTX 4090 is essentially idle during inference. This indicates a fundamental architectural issue - the compute is not happening on the GPU as expected.

---

## Benchmark Configuration

- **Tokens per run**: 150
- **Total runs**: 3
- **Prompts**: Varied (relativity, Python guide, AI breakthroughs)
- **Temperature**: 0.7

---

## Results

### Throughput (DEGRADING)

| Run | Tokens | Time | Throughput | Degradation |
|-----|--------|------|------------|-------------|
| 1 | 150 | 146.22s | 1.03 tok/s | baseline |
| 2 | 150 | 371.19s | 0.40 tok/s | -61% |
| 3 | 150 | 853.67s | 0.18 tok/s | -83% |

### GPU Metrics

| Metric | Average | Max | Min | Expected (AWQ) | Status |
|--------|---------|-----|-----|----------------|--------|
| SM Utilization | 1.9% | 25% | 0% | 40-80% | CRITICAL |
| Memory Utilization | 0.2% | 5% | 0% | 30-70% | CRITICAL |
| Power Draw | 48.1W | 76W | 11W | 150-350W | CRITICAL |
| Temperature | 36.2C | 38C | 34C | 55-80C | Cold (idle) |
| SM Clock | 2520 MHz | 2775 MHz | 210 MHz | 2100-2520 MHz | OK |
| Memory Clock | 10251 MHz | - | 405 MHz | ~10500 MHz | OK |
| VRAM Used | ~8.6 GB | - | 5.9 GB | 5-8 GB | OK |

---

## Analysis

### Root Cause Hypothesis

The GPU showing 1.9% average utilization while generating tokens means:

1. **The transformer forward pass is NOT running on GPU** - or runs so briefly it doesn't register
2. **CPU-bound operations dominate** - Python GIL, serialization, or I/O blocking
3. **Memory leak or resource exhaustion** - explains the 83% degradation across runs

### Expected vs Actual

For an AWQ INT4 8B model on RTX 4090:
- **Expected**: 40-80% GPU utilization, 150-350W power draw
- **Actual**: 1.9% GPU utilization, 48W power draw

This is a **20-40x underutilization**.

### Throughput Degradation

The severe degradation (1.03 -> 0.40 -> 0.18 tok/s) suggests:
- Memory leak in the inference pipeline
- KV cache growing unbounded
- Resource contention accumulating

---

## Recommendations

### Immediate Investigation

1. **Profile the server-side forward pass**
   - Add CUDA event timing around `model.forward()`
   - Check if weights are actually on GPU (`model.to('cuda')`)

2. **Check for CPU bottlenecks**
   - Profile Python code for GIL contention
   - Check gRPC serialization overhead

3. **Investigate memory leak**
   - Monitor VRAM growth across runs
   - Check for tensor accumulation without cleanup

### Potential Fixes

1. **Ensure model is on GPU**: Verify `model.parameters()` show `cuda:0`
2. **Use torch.cuda.synchronize()**: Ensure GPU work completes
3. **Check batch dimension**: Single-sample batches waste GPU parallelism
4. **Review KV cache management**: May be causing memory exhaustion

---

## Raw Data

- GPU metrics: `benchmark_results/gpu_full_benchmark_20260131_200921.csv`
- Total samples: 957 seconds (~16 minutes of monitoring)

---

## Conclusion

The RTX 4090 is operating at **<2% of expected capacity** during inference. This represents a critical performance issue that needs immediate investigation. The throughput degradation across runs (83% slower by run 3) indicates additional memory or resource management problems.
