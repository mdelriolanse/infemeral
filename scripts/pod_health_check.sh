#!/bin/bash
# Pod Health Check Script
# Verifies SSH connectivity to RunPod pod and checks GPU availability.
#
# Usage: ./scripts/pod_health_check.sh

set -euo pipefail

POD_IP="203.57.40.175"
POD_PORT="10271"
SSH_KEY="${HOME}/.ssh/runpod"

echo "=== RunPod Health Check ==="
echo "Pod: ${POD_IP}:${POD_PORT}"
echo ""

# Test SSH connectivity
echo "[1/4] Testing SSH connectivity..."
if ! ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$POD_PORT" root@"$POD_IP" "echo 'SSH connection successful'" 2>/dev/null; then
    echo "ERROR: Cannot connect to pod via SSH"
    exit 1
fi

# Check GPU availability
echo "[2/4] Checking GPU availability..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$POD_PORT" root@"$POD_IP" "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv"

# Check nvidia-smi monitoring capability
echo ""
echo "[3/4] Checking nvidia-smi dmon capability..."
if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$POD_PORT" root@"$POD_IP" "nvidia-smi dmon -c 1 -s pucvmet" 2>/dev/null; then
    echo "nvidia-smi dmon available"
else
    echo "WARNING: nvidia-smi dmon not available, falling back to basic monitoring"
fi

# Check gRPC server status
echo ""
echo "[4/4] Checking gRPC server status..."
SERVER_STATUS=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$POD_PORT" root@"$POD_IP" "ps aux | grep -E 'python.*server' | grep -v grep" 2>/dev/null || true)

if [ -n "$SERVER_STATUS" ]; then
    echo "Server process found:"
    echo "$SERVER_STATUS"
else
    echo "WARNING: No gRPC server process found"
    echo "Start the server with: python -m infemeral.server --mode grpc"
fi

echo ""
echo "=== Health Check Complete ==="
