#!/usr/bin/env python3
"""Profile the end-to-end inference pipeline to identify bottlenecks.

This script runs locally and profiles:
1. Client-side embedding time
2. gRPC serialization/encryption time
3. Network round-trip time
4. gRPC deserialization/decryption time
5. Client-side de-embedding time
6. Sampling time

Usage:
    python agent/repro/profile_inference_pipeline.py --server <pod_ip>:<port>
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure infemeral is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from infemeral.client import Client
from infemeral.crypto import encrypt_bytes, decrypt_bytes
from infemeral.tensors import serialize_tensor, deserialize_tensor


def profile_serialization_overhead():
    """Measure serialization/encryption overhead independent of network."""
    print("\n" + "=" * 60)
    print("SERIALIZATION/ENCRYPTION OVERHEAD")
    print("=" * 60)

    # Typical hidden state size for Llama 8B
    hidden_size = 4096
    seq_lengths = [1, 16, 64, 256]

    for seq_len in seq_lengths:
        tensor = torch.randn(1, seq_len, hidden_size, dtype=torch.float16)
        key = b"0" * 32  # 256-bit key

        # Serialize
        t0 = time.perf_counter()
        for _ in range(10):
            data, shape, dtype = serialize_tensor(tensor)
        serialize_time = (time.perf_counter() - t0) / 10 * 1000

        # Encrypt
        t0 = time.perf_counter()
        for _ in range(10):
            encrypted, nonce = encrypt_bytes(data, key)
        encrypt_time = (time.perf_counter() - t0) / 10 * 1000

        # Decrypt
        t0 = time.perf_counter()
        for _ in range(10):
            decrypted = decrypt_bytes(encrypted, key, nonce)
        decrypt_time = (time.perf_counter() - t0) / 10 * 1000

        # Deserialize
        t0 = time.perf_counter()
        for _ in range(10):
            tensor_back = deserialize_tensor(decrypted, shape, dtype, device="cpu")
        deserialize_time = (time.perf_counter() - t0) / 10 * 1000

        total = serialize_time + encrypt_time + decrypt_time + deserialize_time
        data_size_kb = len(data) / 1024

        print(f"\nSeq length {seq_len} ({data_size_kb:.1f} KB):")
        print(f"  Serialize:   {serialize_time:>7.2f} ms")
        print(f"  Encrypt:     {encrypt_time:>7.2f} ms")
        print(f"  Decrypt:     {decrypt_time:>7.2f} ms")
        print(f"  Deserialize: {deserialize_time:>7.2f} ms")
        print(f"  Total:       {total:>7.2f} ms")


def profile_end_to_end(server_url: str, weights_path: str, device: str, num_runs: int = 5):
    """Profile end-to-end inference with detailed timing breakdown."""
    print("\n" + "=" * 60)
    print("END-TO-END INFERENCE PROFILE")
    print("=" * 60)

    print(f"\nConnecting to {server_url}...")
    client = Client(weights_path=weights_path, server_url=server_url, device=device)

    prompt = "The quick brown fox"
    tokens_to_generate = 10

    print(f"Prompt: {prompt!r}")
    print(f"Device: {device}")
    print(f"Tokens to generate: {tokens_to_generate}")

    # Warmup
    print("\nWarmup run...")
    try:
        _, metrics = client.generate(prompt, max_new_tokens=5, return_metrics=True)
        print(f"  Warmup complete: {metrics.tokens_per_sec:.2f} tok/s")
    except Exception as e:
        print(f"  Warmup failed: {e}")
        client.close()
        return None

    # Measurement runs
    print(f"\nMeasurement runs ({num_runs}):")
    all_results = []

    for run in range(num_runs):
        # Create new session for each run to avoid KV cache effects
        client.session_id = f"profile_run_{run}"

        result, metrics = client.generate(
            prompt,
            max_new_tokens=tokens_to_generate,
            return_metrics=True,
        )

        # Aggregate per-token timing
        if metrics.timings:
            avg_embed = sum(t.embed_ms for t in metrics.timings) / len(metrics.timings)
            avg_network = sum(t.network_ms for t in metrics.timings) / len(metrics.timings)
            avg_de_embed = sum(t.de_embed_ms for t in metrics.timings) / len(metrics.timings)
            avg_sample = sum(t.sample_ms for t in metrics.timings) / len(metrics.timings)
            avg_total = sum(t.total_ms for t in metrics.timings) / len(metrics.timings)
        else:
            avg_embed = avg_network = avg_de_embed = avg_sample = avg_total = 0

        all_results.append({
            "tokens_per_sec": metrics.tokens_per_sec,
            "avg_embed_ms": avg_embed,
            "avg_network_ms": avg_network,
            "avg_de_embed_ms": avg_de_embed,
            "avg_sample_ms": avg_sample,
            "avg_total_ms": avg_total,
        })

        print(f"  Run {run+1}: {metrics.tokens_per_sec:.2f} tok/s, {avg_network:.1f}ms network, {avg_total:.1f}ms total")

    client.close()

    # Analysis
    print("\n" + "-" * 60)
    print("TIMING BREAKDOWN (averages across all runs)")
    print("-" * 60)

    avg_tok_s = sum(r["tokens_per_sec"] for r in all_results) / len(all_results)
    avg_embed = sum(r["avg_embed_ms"] for r in all_results) / len(all_results)
    avg_network = sum(r["avg_network_ms"] for r in all_results) / len(all_results)
    avg_de_embed = sum(r["avg_de_embed_ms"] for r in all_results) / len(all_results)
    avg_sample = sum(r["avg_sample_ms"] for r in all_results) / len(all_results)
    avg_total = sum(r["avg_total_ms"] for r in all_results) / len(all_results)

    # Calculate percentages
    total_time = avg_embed + avg_network + avg_de_embed + avg_sample

    print(f"\n{'Phase':<15} {'Time (ms)':>12} {'% of Total':>12}")
    print("-" * 42)
    print(f"{'Embed':<15} {avg_embed:>12.2f} {(avg_embed/total_time*100):>11.1f}%")
    print(f"{'Network':<15} {avg_network:>12.2f} {(avg_network/total_time*100):>11.1f}%")
    print(f"{'De-embed':<15} {avg_de_embed:>12.2f} {(avg_de_embed/total_time*100):>11.1f}%")
    print(f"{'Sample':<15} {avg_sample:>12.2f} {(avg_sample/total_time*100):>11.1f}%")
    print(f"{'-'*42}")
    print(f"{'TOTAL':<15} {total_time:>12.2f} {'100.0%':>12}")
    print(f"\nThroughput: {avg_tok_s:.2f} tok/s")

    # Diagnosis
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)

    network_pct = (avg_network / total_time) * 100

    if network_pct > 90:
        print(f"\nNETWORK is the dominant bottleneck ({network_pct:.1f}% of time)")
        print("\nThis includes:")
        print("  1. gRPC serialization on client")
        print("  2. Encryption on client")
        print("  3. Network latency to server")
        print("  4. Server-side processing (forward pass + KV cache)")
        print("  5. Network latency back to client")
        print("  6. Decryption on client")
        print("  7. gRPC deserialization on client")
        print("\nTo further diagnose, run diagnose_gpu_utilization.py ON THE SERVER")
        print("to measure actual GPU kernel time vs CPU overhead.")
    elif avg_embed > avg_network:
        print(f"\nEMBEDDING is the bottleneck ({avg_embed:.1f}ms > {avg_network:.1f}ms network)")
        print("This is unusual - check client-side GPU utilization.")
    else:
        print(f"\nNo single dominant bottleneck. Profile server-side for more detail.")

    # Performance classification
    print("\n" + "-" * 60)
    print("THROUGHPUT DEGRADATION CHECK")
    print("-" * 60)

    throughputs = [r["tokens_per_sec"] for r in all_results]
    if len(throughputs) >= 2:
        degradation = (throughputs[0] - throughputs[-1]) / throughputs[0] * 100
        print(f"\nRun 1: {throughputs[0]:.2f} tok/s")
        print(f"Run {len(throughputs)}: {throughputs[-1]:.2f} tok/s")
        print(f"Degradation: {degradation:.1f}%")

        if degradation > 20:
            print("\nWARNING: Significant throughput degradation detected!")
            print("Possible causes:")
            print("  - Memory leak (VRAM or system memory)")
            print("  - KV cache growing unbounded")
            print("  - Python GC pressure")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Profile inference pipeline")
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC server URL (e.g., 203.57.40.175:10271)",
    )
    parser.add_argument(
        "--weights",
        default="./weights/client_weights.safetensors",
        help="Path to client weights",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Client device (cuda/cpu)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of measurement runs",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only run local serialization tests (no server needed)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("INFEMERAL INFERENCE PIPELINE PROFILER")
    print("=" * 60)

    # Always run local serialization profile
    profile_serialization_overhead()

    if not args.local_only:
        profile_end_to_end(
            server_url=args.server,
            weights_path=args.weights,
            device=args.device,
            num_runs=args.runs,
        )

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
