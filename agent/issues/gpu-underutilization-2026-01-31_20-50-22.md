# Debugging Snapshot: GPU Severe Underutilization

**Created**: 2026-01-31_20-50-22
**Status**: OPEN - Root Cause Not Yet Identified
**Severity**: CRITICAL

---

## Issue Summary

The RTX 4090 GPU shows **1.9% average utilization** during transformer inference, when expected utilization is 40-80% for AWQ INT4 models. Additionally, throughput degrades 83% across consecutive runs (1.03 → 0.18 tok/s).

---

## Evidence Collected

### Error Symptoms

| Metric | Observed | Expected | Gap |
|--------|----------|----------|-----|
| GPU SM Utilization | 1.9% avg, 25% max | 40-80% | **20-40x under** |
| Memory Utilization | 0.2% avg | 30-70% | **150-350x under** |
| Power Draw | 48W avg (76W max) | 150-350W | **3-7x under** |
| Temperature | 36°C | 55-80°C | Cold (idle) |
| Throughput Run 1 | 1.03 tok/s | - | baseline |
| Throughput Run 2 | 0.40 tok/s | - | -61% |
| Throughput Run 3 | 0.18 tok/s | - | -83% |

### Raw Data Files
- `benchmark_results/gpu_full_benchmark_20260131_200921.csv` (957 samples)
- `benchmark_results/gpu_utilization_report_20260131.md`

### Key Observations

1. **GPU is essentially idle during inference** - SM utilization peaks at 25%, averages 1.9%
2. **No thermal or power throttling** - Temp 36°C (threshold 83°C), power 48W (TDP 450W)
3. **Clocks are at boost speed** - SM 2520 MHz, Memory 10251 MHz when active
4. **Progressive throughput degradation** - Suggests memory leak or resource exhaustion
5. **VRAM usage is reasonable** - ~8.6 GB for AWQ INT4 8B model

---

## Error Categorization

**Primary**: Performance Anomaly - GPU Underutilization
**Secondary**: Resource Leak - Progressive Degradation

### Likely Root Causes (Ranked)

1. **CPU-Bound Bottleneck** (HIGH probability)
   - Python GIL contention during inference
   - gRPC serialization/deserialization overhead
   - KV cache operations running on CPU

2. **Architecture Mismatch** (MEDIUM probability)
   - Model forward pass may not be fully on GPU
   - Tensor transfers CPU↔GPU on every token
   - Infemeral's split-model architecture may cause excessive synchronization

3. **Memory Leak** (HIGH probability for degradation)
   - KV cache growing unbounded
   - Tensor accumulation without cleanup
   - Python reference cycles preventing garbage collection

4. **I/O Blocking** (LOW probability)
   - Disk-based KV cache (unlikely - mode appears to be memory)
   - Network latency (unlikely - running locally on pod)

---

## Dead Ends (Already Attempted)

| Attempt | Result | Why It Failed |
|---------|--------|---------------|
| Basic nvidia-smi monitoring | Captured data | Confirmed underutilization but didn't explain cause |
| Multiple SSH-based monitoring approaches | SSH connection instability | Required inline Python monitoring instead |
| Background process monitoring via SSH | Process didn't persist | nohup/screen didn't maintain across sessions |

---

## Files Modified This Session

| File | Change |
|------|--------|
| `scripts/pod_health_check.sh` | NEW - Pod connectivity verification |
| `scripts/gpu_monitor.sh` | NEW - GPU monitoring script |
| `scripts/benchmark_gpu_utilization.sh` | NEW - Benchmark orchestrator |
| `scripts/analyze_gpu_logs.py` | NEW - Log parser and analyzer |
| `benchmark_results/gpu_utilization_report_20260131.md` | NEW - Analysis report |

---

## Untried Troubleshooting Steps

### Standard Approaches

1. **Profile Server-Side Forward Pass**
   ```python
   # In server.py forward_transformer()
   import torch
   torch.cuda.synchronize()
   start = torch.cuda.Event(enable_timing=True)
   end = torch.cuda.Event(enable_timing=True)
   start.record()
   # ... forward pass ...
   end.record()
   torch.cuda.synchronize()
   print(f"Forward pass: {start.elapsed_time(end):.2f}ms")
   ```

2. **Verify Model Device Placement**
   ```python
   # Check where model parameters actually reside
   for name, param in model.named_parameters():
       print(f"{name}: {param.device}")
   ```

3. **Add torch.cuda.synchronize() Barriers**
   - Insert sync points to force GPU work completion
   - May reveal if GPU work is being queued but not executed

### Novel Avenues

1. **Hypothesis: Infemeral Split-Model Architecture**

   Infemeral appears to split the model between client and server. The client handles embedding/de-embedding, the server handles transformer layers. If the architecture requires:
   - Client embeds on GPU → transfers to server
   - Server processes → transfers back to client
   - Client de-embeds on GPU

   This could explain why GPU shows low utilization - most time is spent on data transfer, not compute. **Investigation**: Trace the data flow and measure transfer times vs compute times.

2. **Hypothesis: Batch Size = 1 Pathology**

   Single-token autoregressive generation with batch size 1 is notoriously inefficient on GPUs designed for parallelism. The RTX 4090 has 16,384 CUDA cores - generating 1 token at a time may simply not provide enough work to utilize them.

   **Investigation**: Check if continuous batching or speculative decoding is feasible. Compare with known benchmarks for single-batch inference on 4090.

---

## Recommended Next Steps

### Immediate (This Session)

1. **Add CUDA timing to server.py** - Measure actual GPU kernel time
2. **Print tensor devices** - Verify model is on GPU
3. **Monitor VRAM growth** - Check for memory leak across runs

### Short-Term

4. **Profile with torch.profiler** - Full CPU/GPU trace
5. **Compare with vLLM/TGI baseline** - Sanity check expected performance
6. **Review Infemeral architecture** - Understand client-server split implications

### Long-Term

7. **Consider continuous batching** - If single-token is fundamental limit
8. **Evaluate tensor parallelism** - If memory-bound
9. **Optimize KV cache management** - Address degradation issue

---

## Environment Details

| Component | Value |
|-----------|-------|
| Pod IP | 203.57.40.175:10271 |
| GPU | NVIDIA RTX 4090 (24GB GDDR6X) |
| Driver | 570.195.03 |
| VRAM Total | 24564 MiB |
| Model | AWQ INT4 8B Transformer |
| VRAM Used | ~8.6 GB during inference |
| Python | /mnt/.venv/bin/python |
| Workspace | /workspace/infemeral-src |

---

## Resolution Status

**NOT RESOLVED** - Root cause not yet identified.

The GPU underutilization appears to be architectural rather than a simple configuration issue. The Infemeral client-server split model may inherently cause this pattern, or there may be a bug causing compute to run on CPU instead of GPU.

**Next debugging session should focus on**: Adding CUDA profiling to server.py to measure actual GPU kernel execution time.
