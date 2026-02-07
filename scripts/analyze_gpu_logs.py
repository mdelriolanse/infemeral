#!/usr/bin/env python3
"""Analyze GPU utilization logs from nvidia-smi dmon.

Parses nvidia-smi dmon output and provides detailed analysis of GPU
utilization patterns during inference, with RTX 4090 specific thresholds.

Usage:
    python scripts/analyze_gpu_logs.py <log_file> [--output json|text] [--plot]

Examples:
    python scripts/analyze_gpu_logs.py benchmark_results/gpu_metrics_20260131.csv
    python scripts/analyze_gpu_logs.py gpu_log.csv --output json
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class GPUMetrics:
    """Aggregated GPU metrics from a benchmark session."""

    sample_count: int
    duration_seconds: float

    # GPU Utilization (%)
    gpu_util_avg: float
    gpu_util_max: int
    gpu_util_min: int
    gpu_util_p50: float
    gpu_util_p95: float

    # Memory Utilization (%)
    mem_util_avg: float
    mem_util_max: int
    mem_util_min: int

    # Power (W)
    power_avg: float
    power_max: float
    power_min: float

    # Temperature (C)
    temp_avg: float
    temp_max: int
    temp_min: int

    # Clocks (MHz)
    sm_clock_avg: float
    sm_clock_max: int
    sm_clock_min: int
    mem_clock_avg: float

    # Analysis flags
    is_memory_bound: bool
    is_underutilized: bool
    is_thermal_throttling: bool
    is_power_throttling: bool


def percentile(data: list[float], p: int) -> float:
    """Calculate percentile from data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def parse_dmon_log(filepath: str) -> tuple[list[dict], list[str]]:
    """Parse nvidia-smi dmon output file.

    nvidia-smi dmon -s pucvmet -d 1 -o DT produces output like:
    # gpu   pwr gtemp mtemp    sm   mem   enc   dec  mclk  pclk pviol tviol
    #Date       Time        gpu   pwr gtemp mtemp    sm   mem   enc   dec  mclk  pclk pviol tviol
    2026/01/31  10:30:15    0   150    65    55    45    30     0     0 10501  2520     0     0

    Returns:
        Tuple of (records, errors)
    """
    records = []
    errors = []

    with open(filepath) as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # Skip headers and empty lines
        if not line or line.startswith("#"):
            continue

        parts = line.split()

        # Expected format with timestamp: date time gpu pwr gtemp mtemp sm mem enc dec mclk pclk pviol tviol
        # Without timestamp: gpu pwr gtemp mtemp sm mem enc dec mclk pclk pviol tviol
        try:
            if len(parts) >= 14:
                # With timestamp (DT option)
                timestamp = f"{parts[0]} {parts[1]}"
                gpu_idx = int(parts[2])
                power = float(parts[3])
                gpu_temp = int(parts[4])
                mem_temp = int(parts[5]) if parts[5] != "-" else 0
                sm_util = int(parts[6])
                mem_util = int(parts[7])
                enc_util = int(parts[8])
                dec_util = int(parts[9])
                mem_clock = int(parts[10])
                sm_clock = int(parts[11])
                power_violation = int(parts[12])
                thermal_violation = int(parts[13])
            elif len(parts) >= 12:
                # Without timestamp
                timestamp = None
                gpu_idx = int(parts[0])
                power = float(parts[1])
                gpu_temp = int(parts[2])
                mem_temp = int(parts[3]) if parts[3] != "-" else 0
                sm_util = int(parts[4])
                mem_util = int(parts[5])
                enc_util = int(parts[6])
                dec_util = int(parts[7])
                mem_clock = int(parts[8])
                sm_clock = int(parts[9])
                power_violation = int(parts[10])
                thermal_violation = int(parts[11])
            else:
                errors.append(f"Line {line_num}: Unexpected format ({len(parts)} fields)")
                continue

            records.append(
                {
                    "timestamp": timestamp,
                    "gpu_idx": gpu_idx,
                    "power_w": power,
                    "gpu_temp_c": gpu_temp,
                    "mem_temp_c": mem_temp,
                    "sm_util_pct": sm_util,
                    "mem_util_pct": mem_util,
                    "enc_util_pct": enc_util,
                    "dec_util_pct": dec_util,
                    "mem_clock_mhz": mem_clock,
                    "sm_clock_mhz": sm_clock,
                    "power_violation": power_violation,
                    "thermal_violation": thermal_violation,
                }
            )
        except (ValueError, IndexError) as e:
            errors.append(f"Line {line_num}: Parse error - {e}")

    return records, errors


def analyze_metrics(records: list[dict]) -> GPUMetrics:
    """Analyze parsed GPU metrics and compute aggregates."""
    if not records:
        raise ValueError("No records to analyze")

    # Extract arrays
    gpu_utils = [r["sm_util_pct"] for r in records]
    mem_utils = [r["mem_util_pct"] for r in records]
    powers = [r["power_w"] for r in records]
    temps = [r["gpu_temp_c"] for r in records]
    sm_clocks = [r["sm_clock_mhz"] for r in records]
    mem_clocks = [r["mem_clock_mhz"] for r in records]

    # Duration (1 sample per second)
    duration = len(records)

    # RTX 4090 specific thresholds
    # Memory-bound: high memory utilization (>50%) with low GPU utilization (<40%)
    avg_gpu = sum(gpu_utils) / len(gpu_utils)
    avg_mem = sum(mem_utils) / len(mem_utils)
    is_memory_bound = avg_mem > 50 and avg_gpu < 40

    # Underutilized: sustained GPU usage below 25%
    is_underutilized = avg_gpu < 25

    # Thermal throttling: temp > 83C or SM clock < 1800 MHz
    max_temp = max(temps)
    min_sm_clock = min(sm_clocks)
    is_thermal_throttling = max_temp > 83 or min_sm_clock < 1800

    # Power throttling: power > 420W (approaching 450W TDP)
    max_power = max(powers)
    is_power_throttling = max_power > 420

    return GPUMetrics(
        sample_count=len(records),
        duration_seconds=float(duration),
        gpu_util_avg=avg_gpu,
        gpu_util_max=max(gpu_utils),
        gpu_util_min=min(gpu_utils),
        gpu_util_p50=percentile(gpu_utils, 50),
        gpu_util_p95=percentile(gpu_utils, 95),
        mem_util_avg=avg_mem,
        mem_util_max=max(mem_utils),
        mem_util_min=min(mem_utils),
        power_avg=sum(powers) / len(powers),
        power_max=max_power,
        power_min=min(powers),
        temp_avg=sum(temps) / len(temps),
        temp_max=max_temp,
        temp_min=min(temps),
        sm_clock_avg=sum(sm_clocks) / len(sm_clocks),
        sm_clock_max=max(sm_clocks),
        sm_clock_min=min_sm_clock,
        mem_clock_avg=sum(mem_clocks) / len(mem_clocks),
        is_memory_bound=is_memory_bound,
        is_underutilized=is_underutilized,
        is_thermal_throttling=is_thermal_throttling,
        is_power_throttling=is_power_throttling,
    )


def print_text_report(metrics: GPUMetrics, filepath: str) -> None:
    """Print human-readable analysis report."""
    print("=" * 60)
    print("GPU UTILIZATION ANALYSIS REPORT")
    print("=" * 60)
    print(f"Log file: {filepath}")
    print(f"Samples:  {metrics.sample_count}")
    print(f"Duration: {metrics.duration_seconds:.0f} seconds")
    print()

    print("--- GPU Utilization ---")
    print(f"  Average:    {metrics.gpu_util_avg:.1f}%")
    print(f"  Max:        {metrics.gpu_util_max}%")
    print(f"  Min:        {metrics.gpu_util_min}%")
    print(f"  p50:        {metrics.gpu_util_p50:.1f}%")
    print(f"  p95:        {metrics.gpu_util_p95:.1f}%")
    print()

    print("--- Memory Utilization ---")
    print(f"  Average:    {metrics.mem_util_avg:.1f}%")
    print(f"  Max:        {metrics.mem_util_max}%")
    print(f"  Min:        {metrics.mem_util_min}%")
    print()

    print("--- Power Draw ---")
    print(f"  Average:    {metrics.power_avg:.1f}W")
    print(f"  Max:        {metrics.power_max:.1f}W")
    print(f"  Min:        {metrics.power_min:.1f}W")
    print()

    print("--- Temperature ---")
    print(f"  Average:    {metrics.temp_avg:.1f}°C")
    print(f"  Max:        {metrics.temp_max}°C")
    print(f"  Min:        {metrics.temp_min}°C")
    print()

    print("--- SM Clock ---")
    print(f"  Average:    {metrics.sm_clock_avg:.0f} MHz")
    print(f"  Max:        {metrics.sm_clock_max} MHz")
    print(f"  Min:        {metrics.sm_clock_min} MHz")
    print()

    print("--- Analysis (RTX 4090 Thresholds) ---")
    if metrics.is_underutilized:
        print("  [!] UNDERUTILIZED: GPU utilization below 25%")
        print("      This indicates severe underutilization of compute resources.")
    elif metrics.gpu_util_avg < 40:
        print("  [i] LOW UTILIZATION: GPU utilization below 40%")
        print("      This is common for memory-bound AWQ INT4 models.")
    else:
        print("  [✓] GPU utilization in expected range (40-80%)")

    if metrics.is_memory_bound:
        print("  [i] MEMORY-BOUND: High memory util with low GPU util")
        print("      This is expected for quantized transformer models.")

    if metrics.is_thermal_throttling:
        print(f"  [!] THERMAL THROTTLING: Max temp {metrics.temp_max}°C, min SM clock {metrics.sm_clock_min} MHz")
    else:
        print("  [✓] No thermal throttling detected")

    if metrics.is_power_throttling:
        print(f"  [!] POWER THROTTLING: Max power {metrics.power_max:.1f}W (near 450W TDP)")
    else:
        print("  [✓] No power throttling detected")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GPU utilization logs from nvidia-smi dmon"
    )
    parser.add_argument(
        "log_file",
        help="Path to nvidia-smi dmon log file",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--save",
        help="Save analysis to file (auto-determines format from extension)",
    )

    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"ERROR: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    # Parse the log file
    records, errors = parse_dmon_log(args.log_file)

    if errors:
        print(f"WARNING: {len(errors)} parse errors:", file=sys.stderr)
        for err in errors[:5]:
            print(f"  {err}", file=sys.stderr)
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more", file=sys.stderr)

    if not records:
        print("ERROR: No valid records found in log file", file=sys.stderr)
        sys.exit(1)

    # Analyze metrics
    metrics = analyze_metrics(records)

    # Output results
    if args.output == "json":
        result = {
            "log_file": args.log_file,
            "analysis_time": datetime.now().isoformat(),
            "metrics": asdict(metrics),
        }
        print(json.dumps(result, indent=2))
    else:
        print_text_report(metrics, args.log_file)

    # Save if requested
    if args.save:
        save_path = Path(args.save)
        result = {
            "log_file": args.log_file,
            "analysis_time": datetime.now().isoformat(),
            "metrics": asdict(metrics),
        }
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nAnalysis saved to: {save_path}")

    # Exit with error if severe issues detected
    if metrics.is_underutilized:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
