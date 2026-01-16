#!/usr/bin/env python3
"""Benchmark script for client-side inference performance.

Measures per-token timing breakdown, memory usage, and throughput.
Compares CPU vs GPU performance when CUDA is available.

Usage:
    python scripts/benchmark_client.py --weights ./weights/client_weights.safetensors
    python scripts/benchmark_client.py --warmup 2 --runs 5 --tokens 20
"""

import argparse
import json
import statistics
import sys
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from infemeral.client import Client, GenerationMetrics, TokenTiming


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    device: str
    prompt_tokens: int
    generated_tokens: int
    total_time_ms: float
    tokens_per_sec: float
    peak_gpu_memory_mb: float
    peak_cpu_memory_mb: float

    # Per-phase latency percentiles (ms)
    embed_p50: float
    embed_p95: float
    embed_p99: float
    cloak_p50: float
    cloak_p95: float
    cloak_p99: float
    network_p50: float
    network_p95: float
    network_p99: float
    uncloak_p50: float
    uncloak_p95: float
    uncloak_p99: float
    de_embed_p50: float
    de_embed_p95: float
    de_embed_p99: float
    sample_p50: float
    sample_p95: float
    sample_p99: float
    total_p50: float
    total_p95: float
    total_p99: float


def percentile(data: list[float], p: int) -> float:
    """Calculate percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def compute_phase_percentiles(
    timings: list[TokenTiming], phase: str
) -> tuple[float, float, float]:
    """Compute p50, p95, p99 for a phase."""
    values = [getattr(t, f"{phase}_ms") for t in timings]
    return percentile(values, 50), percentile(values, 95), percentile(values, 99)


def run_benchmark(
    client: Client,
    prompt: str,
    max_tokens: int,
) -> tuple[GenerationMetrics, float]:
    """Run a single benchmark and return metrics + CPU memory usage."""
    tracemalloc.start()

    _, metrics = client.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        return_metrics=True,
    )

    _, peak_cpu_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return metrics, peak_cpu_memory / (1024 * 1024)


def benchmark_device(
    weights_path: str,
    device: str,
    prompt: str,
    max_tokens: int,
    warmup_runs: int,
    measurement_runs: int,
    server_url: str,
) -> BenchmarkResult | None:
    """Run benchmark on a specific device."""
    if device == "cuda" and not torch.cuda.is_available():
        print(f"Skipping {device}: CUDA not available")
        return None

    print(f"\n{'='*60}")
    print(f"Benchmarking on {device.upper()}")
    print(f"{'='*60}")

    client = Client(
        weights_path=weights_path,
        server_url=server_url,
        device=device,
    )

    all_timings: list[TokenTiming] = []
    all_metrics: list[GenerationMetrics] = []
    cpu_memory_peaks: list[float] = []

    try:
        # Warmup runs
        print(f"Running {warmup_runs} warmup iterations...")
        for i in range(warmup_runs):
            try:
                run_benchmark(client, prompt, max_tokens)
                print(f"  Warmup {i+1}/{warmup_runs} complete")
            except Exception as e:
                print(f"  Warmup {i+1} failed: {e}")
                return None

        # Measurement runs
        print(f"Running {measurement_runs} measurement iterations...")
        for i in range(measurement_runs):
            try:
                metrics, cpu_peak = run_benchmark(client, prompt, max_tokens)
                all_metrics.append(metrics)
                all_timings.extend(metrics.timings)
                cpu_memory_peaks.append(cpu_peak)
                print(
                    f"  Run {i+1}/{measurement_runs}: "
                    f"{metrics.tokens_per_sec:.1f} tok/s, "
                    f"{metrics.total_tokens} tokens"
                )
            except Exception as e:
                print(f"  Run {i+1} failed: {e}")
                traceback_str = "".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                print(f"  Traceback:\n{traceback_str}")

        if not all_metrics:
            print("No successful runs")
            return None

        # Aggregate results
        avg_tokens_per_sec = statistics.mean(m.tokens_per_sec for m in all_metrics)
        total_tokens = sum(m.total_tokens for m in all_metrics)
        total_time_ms = sum(
            sum(t.total_ms for t in m.timings) for m in all_metrics
        )
        prompt_tokens = all_metrics[0].prompt_tokens

        gpu_peak = max((m.peak_memory_mb for m in all_metrics), default=0.0)
        cpu_peak = max(cpu_memory_peaks) if cpu_memory_peaks else 0.0

        # Compute percentiles for each phase
        phases = ["embed", "cloak", "network", "uncloak", "de_embed", "sample", "total"]
        percentiles_dict: dict[str, tuple[float, float, float]] = {}
        for phase in phases:
            percentiles_dict[phase] = compute_phase_percentiles(all_timings, phase)

        return BenchmarkResult(
            device=device,
            prompt_tokens=prompt_tokens,
            generated_tokens=total_tokens,
            total_time_ms=total_time_ms,
            tokens_per_sec=avg_tokens_per_sec,
            peak_gpu_memory_mb=gpu_peak,
            peak_cpu_memory_mb=cpu_peak,
            embed_p50=percentiles_dict["embed"][0],
            embed_p95=percentiles_dict["embed"][1],
            embed_p99=percentiles_dict["embed"][2],
            cloak_p50=percentiles_dict["cloak"][0],
            cloak_p95=percentiles_dict["cloak"][1],
            cloak_p99=percentiles_dict["cloak"][2],
            network_p50=percentiles_dict["network"][0],
            network_p95=percentiles_dict["network"][1],
            network_p99=percentiles_dict["network"][2],
            uncloak_p50=percentiles_dict["uncloak"][0],
            uncloak_p95=percentiles_dict["uncloak"][1],
            uncloak_p99=percentiles_dict["uncloak"][2],
            de_embed_p50=percentiles_dict["de_embed"][0],
            de_embed_p95=percentiles_dict["de_embed"][1],
            de_embed_p99=percentiles_dict["de_embed"][2],
            sample_p50=percentiles_dict["sample"][0],
            sample_p95=percentiles_dict["sample"][1],
            sample_p99=percentiles_dict["sample"][2],
            total_p50=percentiles_dict["total"][0],
            total_p95=percentiles_dict["total"][1],
            total_p99=percentiles_dict["total"][2],
        )

    finally:
        client.close()


def print_result(result: BenchmarkResult) -> None:
    """Print benchmark result in a formatted table."""
    print(f"\n--- {result.device.upper()} Results ---")
    print(f"Prompt tokens:     {result.prompt_tokens}")
    print(f"Generated tokens:  {result.generated_tokens}")
    print(f"Tokens/sec:        {result.tokens_per_sec:.2f}")
    print(f"Total time:        {result.total_time_ms:.1f} ms")

    if result.peak_gpu_memory_mb > 0:
        print(f"Peak GPU memory:   {result.peak_gpu_memory_mb:.1f} MB")
    print(f"Peak CPU memory:   {result.peak_cpu_memory_mb:.1f} MB")

    print(f"\n{'Phase':<12} {'p50':>10} {'p95':>10} {'p99':>10}")
    print("-" * 44)
    for phase in ["embed", "cloak", "network", "uncloak", "de_embed", "sample", "total"]:
        p50 = getattr(result, f"{phase}_p50")
        p95 = getattr(result, f"{phase}_p95")
        p99 = getattr(result, f"{phase}_p99")
        print(f"{phase:<12} {p50:>10.2f} {p95:>10.2f} {p99:>10.2f}")


def print_comparison(cpu_result: BenchmarkResult | None, gpu_result: BenchmarkResult | None) -> None:
    """Print side-by-side comparison of CPU vs GPU results."""
    if not cpu_result or not gpu_result:
        return

    print("\n" + "=" * 60)
    print("CPU vs GPU COMPARISON")
    print("=" * 60)

    speedup = gpu_result.tokens_per_sec / cpu_result.tokens_per_sec
    print(f"Throughput speedup: {speedup:.1f}x")

    print(f"\n{'Phase':<12} {'CPU p50':>10} {'GPU p50':>10} {'Speedup':>10}")
    print("-" * 44)
    for phase in ["embed", "cloak", "network", "uncloak", "de_embed", "sample", "total"]:
        cpu_p50 = getattr(cpu_result, f"{phase}_p50")
        gpu_p50 = getattr(gpu_result, f"{phase}_p50")
        phase_speedup = cpu_p50 / gpu_p50 if gpu_p50 > 0 else 0
        print(f"{phase:<12} {cpu_p50:>10.2f} {gpu_p50:>10.2f} {phase_speedup:>9.1f}x")


def save_results(
    results: dict[str, BenchmarkResult | None],
    output_path: str,
) -> None:
    """Save results to JSON file."""
    data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "results": {},
    }

    if torch.cuda.is_available():
        data["cuda_device"] = torch.cuda.get_device_name(0)

    for device, result in results.items():
        if result:
            data["results"][device] = asdict(result)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


# Baseline thresholds for regression detection
BASELINE_CPU_MS = 25.0
BASELINE_GPU_MS = 5.0


def check_regression(result: BenchmarkResult) -> bool:
    """Check if results indicate a performance regression."""
    threshold = BASELINE_GPU_MS if result.device == "cuda" else BASELINE_CPU_MS
    if result.total_p50 > threshold * 1.2:
        print(f"\nWARNING: Performance regression detected!")
        print(f"  Expected p50 latency: <= {threshold * 1.2:.1f} ms")
        print(f"  Actual p50 latency:   {result.total_p50:.1f} ms")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Benchmark Infemeral client")
    parser.add_argument(
        "--weights",
        default="./weights/client_weights.safetensors",
        help="Path to client weights",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC server URL",
    )
    parser.add_argument(
        "--prompt",
        default="The quick brown fox",
        help="Prompt to use for benchmarking",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=20,
        help="Number of tokens to generate per run",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of measurement runs",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "both"],
        default="both",
        help="Device(s) to benchmark",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Exit with error if regression detected",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("INFEMERAL CLIENT BENCHMARK")
    print("=" * 60)
    print(f"Prompt: {args.prompt!r}")
    print(f"Tokens per run: {args.tokens}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Measurement runs: {args.runs}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results: dict[str, BenchmarkResult | None] = {}

    devices = []
    if args.device in ("cpu", "both"):
        devices.append("cpu")
    if args.device in ("cuda", "both"):
        devices.append("cuda")

    for device in devices:
        result = benchmark_device(
            weights_path=args.weights,
            device=device,
            prompt=args.prompt,
            max_tokens=args.tokens,
            warmup_runs=args.warmup,
            measurement_runs=args.runs,
            server_url=args.server,
        )
        results[device] = result
        if result:
            print_result(result)

    if len(results) == 2 and results.get("cpu") and results.get("cuda"):
        print_comparison(results["cpu"], results["cuda"])

    save_results(results, args.output)

    if args.check_regression:
        has_regression = False
        for result in results.values():
            if result and check_regression(result):
                has_regression = True
        if has_regression:
            sys.exit(1)


if __name__ == "__main__":
    main()
