#!/bin/bash
# GPU Monitor Script (runs on pod)
# Captures GPU utilization metrics during inference using nvidia-smi dmon.
#
# Usage: ./gpu_monitor.sh [output_file]
# Stop: kill $(cat /tmp/gpu_monitor.pid)

OUTPUT_FILE="${1:-/tmp/gpu_metrics_$(date +%Y%m%d_%H%M%S).csv}"
PID_FILE="/tmp/gpu_monitor.pid"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "GPU monitoring already running (PID: $OLD_PID)"
        echo "Stop with: kill $OLD_PID"
        exit 1
    fi
fi

echo "Starting GPU monitoring..."
echo "Output: $OUTPUT_FILE"

# Use nvidia-smi dmon for detailed metrics at 1-second intervals
# pwr: power, gtemp: GPU temp, mtemp: memory temp
# sm: SM utilization, mem: memory utilization
# enc/dec: encoder/decoder utilization
# mclk/pclk: memory/GPU clock speeds
# pviol/tviol: power/thermal violations
nohup nvidia-smi dmon -s pucvmet -d 1 -o DT > "$OUTPUT_FILE" 2>&1 &
MONITOR_PID=$!

echo "$MONITOR_PID" > "$PID_FILE"
echo "GPU monitoring started (PID: $MONITOR_PID)"
echo "To stop: kill $MONITOR_PID"
echo "To check: tail -f $OUTPUT_FILE"
