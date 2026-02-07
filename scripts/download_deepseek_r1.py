#!/usr/bin/env python3
"""Download DeepSeek-R1-32B-GPTQ-INT4 to network volume.

Usage:
    python scripts/download_deepseek_r1.py [--output-dir /workspace/weights/deepseek-r1-32b]
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4"
DEFAULT_OUTPUT_DIR = "/workspace/weights/deepseek-r1-32b"


def main():
    parser = argparse.ArgumentParser(description="Download DeepSeek-R1-32B model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Downloading {MODEL_ID} to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )

    print(f"\nDownload complete! Files in {output_dir}:")
    total_size = 0
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_gb = f.stat().st_size / 1e9
            total_size += size_gb
            print(f"  {f.name}: {size_gb:.2f} GB")
        else:
            print(f"  {f.name}/")

    print(f"\nTotal size: {total_size:.2f} GB")
    print("\nNext steps:")
    print("  1. Extract client weights:")
    print('     python -m infemeral.model_prep \\')
    print(f'         --model-id "{MODEL_ID}" \\')
    print('         --output-dir /workspace/weights/deepseek-r1-32b-client \\')
    print('         --client-only')
    print("\n  2. Set environment variables:")
    print(f"     export INFEMERAL_SERVER_WEIGHTS_DIR={output_dir}")
    print(f"     export INFEMERAL_SERVER_MODEL_ID={MODEL_ID}")
    print("     export INFEMERAL_SERVER_MAX_CONTEXT_LENGTH=4096  # RTX 4090")


if __name__ == "__main__":
    main()
