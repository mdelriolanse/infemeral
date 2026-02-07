# GPU Hardware Utilization Benchmark Plan

**Created**: 2026-01-31_19-37-09
**Feature Request**: Benchmark hardware utilization during inference to analyze GPU efficiency
**Goal**: Identify if the GPU is being underutilized during transformer inference on the remote RunPod pod

---

## Clarification Questions (Resolved)

| Question | Answer |
|----------|--------|
| **Pod Availability** | ✅ Yes - Pod accessible at `203.57.40.175:10271` |
| **GPU Model** | **NVIDIA RTX 4090** (24GB GDDR6X, 450W TDP, 1 TB/s bandwidth) |
| **Benchmark Duration** | **150 tokens** (sustained load test) |
| **NVIDIA Tooling** | To be verified (nvidia-smi expected on RunPod) |

---

## RTX 4090 Performance Baseline

### Hardware Specifications
| Spec | Value |
|------|-------|
| CUDA Cores | 16,384 |
| VRAM | 24 GB GDDR6X |
| Memory Bandwidth | 1,008 GB/s |
| TDP | 450W |
| Base Clock | 2.23 GHz |
| Boost Clock | 2.52 GHz |

### Expected Metrics for AWQ INT4 8B Model
| Metric | Expected Range | Notes |
|--------|----------------|-------|
| GPU Utilization | 40-80% | Lower due to memory-bound nature of quantized models |
| Memory Used | 5-8 GB | AWQ INT4 compression reduces footprint |
| Power Draw | 150-350W | Varies with utilization |
| Temperature | 55-80°C | RunPod cooling dependent |
| SM Clock | 2.1-2.5 GHz | Should stay near boost if not throttling |

### Why RTX 4090 May Show Lower Utilization
1. **Memory-bound workload**: AWQ INT4 models are bandwidth-limited, not compute-limited
2. **Single-request inference**: No batching means GPU cores underutilized between layers
3. **Sequential token generation**: Each token requires full forward pass, causing GPU idle between requests
4. **Consumer GPU architecture**: Unlike datacenter GPUs (A100/H100), 4090 lacks HBM and tensor core optimizations for inference

---

## Dependency Mapping

### Prerequisites (Must Happen First)
1. **SSH Connectivity**: Verify SSH access to pod using `~/.ssh/runpod` key
2. **Server Running**: Ensure gRPC server is active on pod (`python -m infemeral.server --mode grpc`)
3. **NVIDIA Tooling**: Confirm `nvidia-smi` is available on the pod

### Blockers/External Dependencies
| Dependency | Type | Impact | Mitigation |
|------------|------|--------|------------|
| RunPod pod availability | External | Critical | Check pod status before starting |
| SSH key authentication | External | Critical | Verify key exists at `~/.ssh/runpod` |
| NVIDIA drivers on pod | External | High | Fall back to PyTorch CUDA metrics if missing |
| gRPC server running | External | Critical | Start server via SSH before benchmarking |
| Network latency (SSH + gRPC) | External | Medium | Account for network overhead in metrics |

---

## Phase 1: MVP/Foundational - Basic GPU Monitoring

### Objective
Establish baseline GPU utilization visibility during inference using standard NVIDIA tools.

### Tasks

#### 1.1 Verify SSH Connectivity and Pod Setup
- [ ] Test SSH connection to pod: `ssh -p 10271 root@203.57.40.175`
- [ ] Verify NVIDIA tools available: `nvidia-smi --query`
- [ ] Check current GPU state: `nvidia-smi dmon -s pucvmet -d 1`
- [ ] Confirm gRPC server is running or start it

**Script**: Create `scripts/pod_health_check.sh`
```bash
#!/bin/bash
POD_IP="203.57.40.175"
POD_PORT="10271"
SSH_KEY="$HOME/.ssh/runpod"

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$POD_PORT" root@"$POD_IP" "
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv
ps aux | grep -E 'python.*server|grpc' | grep -v grep
"
```

#### 1.2 Create GPU Monitoring Script (Remote)
- [ ] Write a monitoring script that runs on the pod alongside inference
- [ ] Capture: GPU utilization %, memory usage, SM activity, power draw
- [ ] Output to timestamped log file

**Script**: Create `scripts/gpu_monitor.sh` (to be run on pod)
```bash
#!/bin/bash
# Run this on the pod to monitor GPU during inference
OUTPUT_FILE="/tmp/gpu_metrics_$(date +%Y%m%d_%H%M%S).csv"

nvidia-smi --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv -l 1 > "$OUTPUT_FILE" &
MONITOR_PID=$!

echo "GPU monitoring started (PID: $MONITOR_PID)"
echo "Output: $OUTPUT_FILE"
echo "Run 'kill $MONITOR_PID' to stop"
```

#### 1.3 Create Benchmark Orchestrator (Local)
- [ ] Script that:
  1. Starts GPU monitoring on pod via SSH
  2. Runs inference from local machine
  3. Stops monitoring and retrieves logs
  4. Parses and analyzes results

**Script**: Create `scripts/benchmark_gpu_utilization.sh`

#### 1.4 Baseline Metrics Collection
- [ ] Run 3 inference sessions generating 150 tokens each
- [ ] Collect GPU metrics for each session
- [ ] Document baseline utilization patterns for RTX 4090
- [ ] Compare against expected 40-80% GPU utilization range

### Impacted Files
- `scripts/benchmark_gpu_utilization.sh` (new)
- `scripts/gpu_monitor.sh` (new)
- `scripts/pod_health_check.sh` (new)

### Success Criteria
- [ ] Can SSH to pod and run nvidia-smi
- [ ] GPU monitoring captures utilization data during inference
- [ ] Have baseline utilization numbers for 150-token sustained generation
- [ ] Identify if GPU utilization stays above 40% threshold for RTX 4090

---

## Phase 2: Scaling/Refining - Detailed Profiling

### Objective
Add fine-grained profiling to understand GPU utilization patterns throughout the inference pipeline.

### Tasks

#### 2.1 Enhanced nvidia-smi Monitoring
- [ ] Use `nvidia-smi dmon` for more granular metrics (1-second intervals)
- [ ] Capture SM utilization, memory bandwidth, PCIe throughput
- [ ] Add tensor core utilization if available (datacenter GPUs)

**Command**:
```bash
nvidia-smi dmon -s pucvmet -d 1 -o DT > /tmp/gpu_dmon.log
```

Metrics captured:
- `pwr`: Power usage (W)
- `util`: GPU utilization (%)
- `mem`: Memory utilization (%)
- `enc/dec`: Encoder/Decoder utilization
- `mclk/pclk`: Memory/GPU clock speeds
- `pviol/tviol`: Power/Thermal violations

#### 2.2 PyTorch CUDA Profiling Integration
- [ ] Add CUDA event timing to `forward_transformer()` in `server.py`
- [ ] Measure kernel launch times vs actual execution
- [ ] Profile memory allocation patterns

**Code Addition** (for debugging, not production):
```python
# In server.py forward_transformer()
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    # existing forward pass
    pass
prof.export_chrome_trace("/tmp/trace.json")
```

#### 2.3 Identify GPU Idle Periods
- [ ] Correlate low GPU utilization with specific pipeline stages
- [ ] Common suspects:
  - gRPC serialization/deserialization overhead
  - KV cache disk I/O (if using disk mode)
  - Python GIL contention
  - CPU-bound operations between GPU kernels

#### 2.4 Memory Bandwidth Analysis
- [ ] Check if memory bandwidth is the bottleneck (not compute)
- [ ] AWQ INT4 models are often memory-bound
- [ ] Use `nvidia-smi --query-gpu=memory.bandwidth` or `dcgm-exporter`

### Impacted Files
- `infemeral/server.py` (optional profiling hooks)
- `scripts/benchmark_gpu_utilization.sh` (enhanced metrics)
- `scripts/analyze_gpu_logs.py` (new - parse and visualize)

### Success Criteria
- [ ] Can identify which pipeline stages have low GPU utilization
- [ ] Understand if bottleneck is compute-bound or memory-bound
- [ ] Have per-layer timing breakdown

---

## Phase 3: Optimization - Remediation & Continuous Monitoring

### Objective
Create permanent monitoring infrastructure and address identified bottlenecks.

### Tasks

#### 3.1 Continuous Monitoring Setup
- [ ] Create systemd service or tmux session for persistent GPU monitoring
- [ ] Log rotation to prevent disk fill
- [ ] Optional: Push metrics to Prometheus/Grafana

#### 3.2 Automated Benchmark Suite
- [ ] Integrate GPU utilization checks into existing `benchmark_client.py`
- [ ] Add server-side metrics endpoint (optional gRPC extension)
- [ ] Create baseline regression tests

#### 3.3 Bottleneck Remediation (Based on Findings)

**If GPU utilization is low (<50%):**
- Check if batching is possible (unlikely with single-request model)
- Verify CUDA graphs aren't possible (dynamic shapes)
- Consider torch.compile() for static optimization

**If memory-bound (high memory util, low GPU util):**
- Expected for AWQ INT4 models
- Consider tensor parallelism if multi-GPU available
- Check for unnecessary memory copies

**If KV cache I/O is bottleneck:**
- Ensure `kv_cache_mode="memory"` (not disk)
- Profile disk operations if using hybrid mode

**If network is bottleneck:**
- Profile gRPC overhead
- Consider local testing to isolate

### Impacted Files
- `scripts/benchmark_client.py` (server-side metrics integration)
- `infemeral/server.py` (metrics endpoint, optional)
- New monitoring scripts and dashboards

### Success Criteria
- [ ] GPU utilization improved by measurable amount (or confirmed as expected)
- [ ] Have automated benchmark that tracks GPU utilization over time
- [ ] Documentation of expected vs actual utilization

---

## Risk Assessment

| Risk | Impact | Mitigation Strategy |
|------|--------|---------------------|
| Pod not running/accessible | Critical | Check pod status first; have backup pod ready |
| nvidia-smi not available | High | Use PyTorch CUDA metrics as fallback |
| Network latency skews results | Medium | Run server-local tests to isolate GPU performance |
| AWQ model naturally memory-bound | Medium | Document as expected behavior if confirmed |
| SSH connection drops during monitoring | Low | Use tmux/screen on remote; auto-restart scripts |
| Log files fill disk | Low | Use log rotation; limit capture duration |
| GPU throttling due to thermal | Medium | Monitor temperature; check cooling |

---

## Suggested Tests

### Unit Tests
1. **Test GPU metrics parsing**: Mock nvidia-smi output and verify parsing logic
2. **Test benchmark script idempotency**: Run multiple times, verify consistent results

### Integration Tests
1. **Test full benchmark cycle**: SSH → monitor → inference → collect → analyze
2. **Test with different prompt lengths**: Verify scaling behavior

### Performance Validation (RTX 4090 + 150 Tokens)
1. **Baseline Utilization Test**:
   - Generate 150 tokens with complex prompt
   - Expected GPU utilization: >40% average during transformer forward pass
   - Flag if GPU utilization <25% sustained (indicates severe underutilization)

2. **Memory Bandwidth Test**:
   - Monitor memory utilization alongside GPU compute utilization
   - If memory util >60% while GPU util <40%, model is memory-bound (expected for AWQ on RTX 4090)
   - RTX 4090 has 1 TB/s bandwidth - should not be the primary bottleneck

3. **Thermal/Power Throttling Test**:
   - Monitor temperature and power draw throughout 150-token generation
   - Flag if temp >83°C or power >420W (approaching 450W TDP)
   - Check if SM clock drops below 2100 MHz (indicates throttling)

4. **Latency Correlation Test**:
   - Correlate low GPU utilization periods with high network/CPU latency
   - Identify if GPU is idle waiting for gRPC data transfer

---

## Implementation Notes

### SSH Command Templates

**Start GPU monitoring in background**:
```bash
ssh -i ~/.ssh/runpod -p 10271 root@203.57.40.175 "
nohup nvidia-smi dmon -s pucvmet -d 1 -o DT > /tmp/gpu_metrics.log 2>&1 &
echo \$!
"
```

**Run inference (existing script)**:
```bash
./run_inference.sh "Explain the theory of relativity in simple terms" 150
```

**Retrieve logs**:
```bash
scp -i ~/.ssh/runpod -P 10271 root@203.57.40.175:/tmp/gpu_metrics.log ./
```

**Stop monitoring**:
```bash
ssh -i ~/.ssh/runpod -p 10271 root@203.57.40.175 "pkill nvidia-smi"
```

### Key Metrics to Capture (RTX 4090 Specific)

| Metric | nvidia-smi Flag | Expected Range (RTX 4090) | Red Flag |
|--------|-----------------|---------------------------|----------|
| GPU Utilization | `utilization.gpu` | 40-80% during forward | <25% sustained |
| Memory Utilization | `utilization.memory` | 30-70% | <15% or >90% |
| Memory Used | `memory.used` | 5-8 GB for AWQ 8B model | >12 GB or unexpected growth |
| Power Draw | `power.draw` | 150-350W | >420W (near 450W TDP) = throttle |
| Temperature | `temperature.gpu` | 55-80°C | >83°C = thermal throttling |
| SM Clock | `clocks.sm` | 2100-2520 MHz | <1800 MHz = thermal/power throttle |
| Memory Clock | `clocks.mem` | ~10500 MHz | Significant drop = throttling |

---

## Task List

### Phase 1 Tasks
1. [ ] Verify SSH connectivity to pod
2. [ ] Confirm nvidia-smi available on pod
3. [ ] Create `scripts/pod_health_check.sh`
4. [ ] Create `scripts/gpu_monitor.sh`
5. [ ] Create `scripts/benchmark_gpu_utilization.sh`
6. [ ] Run baseline benchmark (3 runs)
7. [ ] Document baseline metrics

### Phase 2 Tasks
8. [ ] Add enhanced dmon monitoring
9. [ ] Create log parsing script
10. [ ] Correlate GPU idle periods with pipeline stages
11. [ ] Analyze memory bandwidth patterns
12. [ ] Document bottleneck findings

### Phase 3 Tasks
13. [ ] Implement recommended optimizations
14. [ ] Integrate GPU metrics into benchmark_client.py
15. [ ] Create regression test baseline
16. [ ] Document final GPU utilization expectations

---

## Files Impacted

### New Files
- `scripts/pod_health_check.sh` - Verify pod health and GPU availability
- `scripts/gpu_monitor.sh` - GPU monitoring script (runs on pod)
- `scripts/benchmark_gpu_utilization.sh` - Orchestrate benchmark from local
- `scripts/analyze_gpu_logs.py` - Parse and visualize GPU metrics
- `docs/gpu-utilization-baseline.md` - Document expected utilization

### Modified Files
- `scripts/benchmark_client.py` - Add server-side GPU metrics integration
- `infemeral/server.py` - (Optional) Add profiling hooks

---

## Expected Outcomes

After completing this plan, we will have:

1. **Baseline Metrics**: Documented GPU utilization during normal inference
2. **Bottleneck Identification**: Clear understanding of where GPU is underutilized and why
3. **Monitoring Scripts**: Reusable tools for future GPU performance analysis
4. **Optimization Roadmap**: Specific recommendations if utilization is below expectations
5. **Regression Tests**: Automated checks to catch future GPU utilization degradation

---

## Next Steps (Ready to Execute)

1. ~~User confirms pod is accessible~~ ✅ Confirmed
2. Run Phase 1 health checks and baseline collection
   - SSH to pod and verify nvidia-smi
   - Start GPU monitoring
   - Run 150-token inference from local machine
   - Collect and analyze GPU metrics
3. Analyze results and proceed to Phase 2 if needed
4. Document findings and implement optimizations

**Benchmark Configuration:**
- Pod: `203.57.40.175:10271`
- GPU: RTX 4090 (24GB)
- Token count: 150 tokens
- Expected duration: ~30-60 seconds for full generation
