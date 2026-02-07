#!/bin/bash
# GPU Utilization Benchmark Orchestrator
# Coordinates GPU monitoring on pod, runs inference, and retrieves metrics.
#
# Usage:
#   ./scripts/benchmark_gpu_utilization.sh [prompt] [max_tokens] [runs]
#
# Examples:
#   ./scripts/benchmark_gpu_utilization.sh "Explain quantum computing" 150 3
#   ./scripts/benchmark_gpu_utilization.sh  # Uses defaults

set -euo pipefail

# Configuration
POD_IP="203.57.40.175"
POD_PORT="10271"
SSH_KEY="${HOME}/.ssh/runpod"
SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -p $POD_PORT root@$POD_IP"
SCP_CMD="scp -i $SSH_KEY -o StrictHostKeyChecking=no -P $POD_PORT"
REMOTE_WORKSPACE="/workspace/infemeral-src"
REMOTE_VENV="/mnt/.venv/bin/python"

# Default parameters
PROMPT="${1:-Explain the theory of relativity in simple terms. Be thorough and detailed.}"
MAX_TOKENS="${2:-150}"
RUNS="${3:-3}"

# Local output directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

echo "=============================================================="
echo "GPU UTILIZATION BENCHMARK"
echo "=============================================================="
echo "Pod:        ${POD_IP}:${POD_PORT}"
echo "GPU:        RTX 4090 (24GB)"
echo "Prompt:     ${PROMPT:0:50}..."
echo "Max Tokens: ${MAX_TOKENS}"
echo "Runs:       ${RUNS}"
echo "Output:     ${OUTPUT_DIR}"
echo "=============================================================="
echo ""

# Step 1: Check pod connectivity
echo "[1/6] Checking pod connectivity..."
if ! $SSH_CMD "echo 'Connected'" >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to pod"
    exit 1
fi
echo "Pod connectivity: OK"

# Step 2: Check/start gRPC server
echo ""
echo "[2/6] Checking gRPC server..."
SERVER_RUNNING=$($SSH_CMD "ps aux | grep -E 'python.*server' | grep -v grep" 2>/dev/null || true)
if [ -z "$SERVER_RUNNING" ]; then
    echo "Server not running. Starting gRPC server..."
    $SSH_CMD "cd $REMOTE_WORKSPACE && nohup $REMOTE_VENV -m infemeral.server --mode grpc > /tmp/server.log 2>&1 &"
    echo "Waiting for server to start..."
    sleep 5

    SERVER_RUNNING=$($SSH_CMD "ps aux | grep -E 'python.*server' | grep -v grep" 2>/dev/null || true)
    if [ -z "$SERVER_RUNNING" ]; then
        echo "ERROR: Failed to start gRPC server"
        echo "Check server log: $SSH_CMD 'cat /tmp/server.log'"
        exit 1
    fi
fi
echo "gRPC server: Running"

# Step 3: Get baseline GPU state
echo ""
echo "[3/6] Getting baseline GPU state..."
$SSH_CMD "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm --format=csv"

# Step 4: Start GPU monitoring on pod
echo ""
echo "[4/6] Starting GPU monitoring on pod..."
REMOTE_LOG_FILE="/tmp/gpu_benchmark_${TIMESTAMP}.csv"

# Kill any existing nvidia-smi dmon processes
$SSH_CMD "pkill -f 'nvidia-smi dmon' 2>/dev/null || true"

# Start monitoring
$SSH_CMD "nohup nvidia-smi dmon -s pucvmet -d 1 -o DT > $REMOTE_LOG_FILE 2>&1 &"
MONITOR_PID=$($SSH_CMD "pgrep -f 'nvidia-smi dmon'")
echo "GPU monitoring started (PID: $MONITOR_PID)"
echo "Remote log: $REMOTE_LOG_FILE"

# Give monitoring a moment to start
sleep 2

# Step 5: Run inference benchmarks
echo ""
echo "[5/6] Running inference benchmarks..."
ALL_RESULTS=""
for i in $(seq 1 $RUNS); do
    echo ""
    echo "--- Run $i of $RUNS ---"
    START_TIME=$(date +%s.%N)

    # Run inference on the pod
    RESULT=$($SSH_CMD "cd $REMOTE_WORKSPACE && $REMOTE_VENV -c \"
from infemeral.client import Client
import time

client = Client(
    weights_path='/workspace/weights/client_weights.safetensors',
    server_url='localhost:50051',
    device='cuda'
)

start = time.time()
result, metrics = client.generate(
    '''$PROMPT''',
    max_new_tokens=$MAX_TOKENS,
    temperature=0.7,
    return_metrics=True
)
end = time.time()

print(f'Tokens generated: {metrics.total_tokens}')
print(f'Time: {end - start:.2f}s')
print(f'Tokens/sec: {metrics.tokens_per_sec:.2f}')
client.close()
\"" 2>&1)

    END_TIME=$(date +%s.%N)
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)

    echo "$RESULT"
    echo "Wall-clock time: ${DURATION}s"
    ALL_RESULTS="${ALL_RESULTS}Run $i:\n${RESULT}\nWall time: ${DURATION}s\n\n"
done

# Step 6: Stop monitoring and retrieve logs
echo ""
echo "[6/6] Stopping monitoring and retrieving logs..."
$SSH_CMD "pkill -f 'nvidia-smi dmon' 2>/dev/null || true"
sleep 1

# Retrieve the log file
LOCAL_LOG_FILE="${OUTPUT_DIR}/gpu_metrics_${TIMESTAMP}.csv"
$SCP_CMD "root@$POD_IP:$REMOTE_LOG_FILE" "$LOCAL_LOG_FILE"
echo "Log file saved: $LOCAL_LOG_FILE"

# Get final GPU state
echo ""
echo "Final GPU state:"
$SSH_CMD "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm --format=csv"

# Summary
echo ""
echo "=============================================================="
echo "BENCHMARK COMPLETE"
echo "=============================================================="
echo "GPU metrics log: $LOCAL_LOG_FILE"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_gpu_logs.py $LOCAL_LOG_FILE"
echo ""

# Quick inline analysis
if command -v python3 &>/dev/null && [ -f "$LOCAL_LOG_FILE" ]; then
    echo "Quick Analysis:"
    python3 -c "
import sys

# Parse nvidia-smi dmon output
lines = open('$LOCAL_LOG_FILE').readlines()
gpu_utils = []
mem_utils = []
powers = []
temps = []

for line in lines:
    if line.startswith('#') or not line.strip():
        continue
    parts = line.split()
    if len(parts) >= 10:
        try:
            gpu_utils.append(int(parts[2]))  # SM utilization
            mem_utils.append(int(parts[3]))  # Memory utilization
            powers.append(float(parts[4]))   # Power
            temps.append(int(parts[5]))      # Temperature
        except (ValueError, IndexError):
            continue

if gpu_utils:
    print(f'  Samples collected: {len(gpu_utils)}')
    print(f'  GPU Utilization:   avg={sum(gpu_utils)/len(gpu_utils):.1f}%, max={max(gpu_utils)}%, min={min(gpu_utils)}%')
    print(f'  Memory Utilization: avg={sum(mem_utils)/len(mem_utils):.1f}%, max={max(mem_utils)}%, min={min(mem_utils)}%')
    print(f'  Power Draw:        avg={sum(powers)/len(powers):.1f}W, max={max(powers):.1f}W')
    print(f'  Temperature:       avg={sum(temps)/len(temps):.1f}C, max={max(temps)}C')

    # Check thresholds (RTX 4090 specific)
    avg_gpu = sum(gpu_utils)/len(gpu_utils)
    if avg_gpu < 25:
        print(f'  WARNING: GPU utilization below 25% - severe underutilization')
    elif avg_gpu < 40:
        print(f'  NOTE: GPU utilization below 40% - may be memory-bound (expected for AWQ)')
    else:
        print(f'  GPU utilization in expected range for AWQ INT4 model')
else:
    print('  No valid GPU metrics found in log')
" 2>/dev/null || echo "  (Python analysis unavailable)"
fi
