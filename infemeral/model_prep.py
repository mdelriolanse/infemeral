"""One-time model preparation: download model and extract client weights.

This script:
1. Downloads the AWQ model from HuggingFace to a local directory
2. Extracts embedding layers for client (embed_tokens + lm_head)
3. Server loads the full model directly via from_pretrained()

Usage:
    python -m infemeral.model_prep --output-dir /workspace/weights
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def download_model(
    model_id: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    output_dir: str = "/workspace/weights",
) -> Path:
    """Download model from HuggingFace to local directory.

    Args:
        model_id: HuggingFace model ID
        output_dir: Base directory for weights

    Returns:
        Path to downloaded model directory
    """
    output_path = Path(output_dir)
    model_dir = output_path / "model"

    print(f"Downloading {model_id} to {model_dir}...")

    # Download all model files
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )

    print(f"Model downloaded to {model_dir}")
    return model_dir


def extract_client_weights(
    model_id: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    model_dir: str | Path | None = None,
    output_dir: str = "/workspace/weights",
    device: str = "cuda",
) -> Path:
    """Extract client weights (embed_tokens + lm_head) from model.

    Args:
        model_id: HuggingFace model ID (used if model_dir not provided)
        model_dir: Local path to model (if already downloaded)
        output_dir: Directory to save client weights
        device: Device for loading model

    Returns:
        Path to client weights file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine where to load from
    load_from = str(model_dir) if model_dir else model_id

    print(f"Loading model from {load_from} to extract client weights...")

    model = AutoModelForCausalLM.from_pretrained(
        load_from,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Extract client components (embedding + lm_head)
    print("Extracting client components (embed_tokens + lm_head)...")
    client_state_dict = {}
    metadata = {}

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        # Llama-style architecture
        # Convert to float16 for smaller file size (sufficient precision for embeddings)
        embed_weight = model.model.embed_tokens.weight.data.clone().cpu().half()
        client_state_dict["embed_tokens.weight"] = embed_weight
        print(f"  embed_tokens.weight: {embed_weight.shape} (float16)")
    else:
        raise ValueError("Could not find embed_tokens in model")

    if hasattr(model, "lm_head"):
        lm_head_weight = model.lm_head.weight.data.clone().cpu().half()

        # Check for tied embeddings (many models share embed_tokens and lm_head)
        # Compare in float32 for accuracy, then save in float16
        is_tied = torch.equal(
            model.model.embed_tokens.weight.data,
            model.lm_head.weight.data,
        )

        if is_tied:
            print("  Detected TIED EMBEDDINGS - skipping duplicate lm_head.weight")
            print(f"  Memory saved: {lm_head_weight.numel() * 2 / 1e6:.1f} MB")
            metadata["tied_embeddings"] = "true"
        else:
            client_state_dict["lm_head.weight"] = lm_head_weight
            print(f"  lm_head.weight: {lm_head_weight.shape} (float16)")
            metadata["tied_embeddings"] = "false"
    else:
        raise ValueError("Could not find lm_head in model")

    # Save client weights with metadata
    metadata["dtype"] = "float16"
    client_path = output_path / "client_weights.safetensors"
    save_file(client_state_dict, client_path, metadata=metadata)
    print(f"Saved client weights to {client_path}")
    if metadata.get("tied_embeddings") == "true":
        print(f"  Metadata: tied_embeddings=true, dtype=float16")

    # Save tokenizer for client use
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    tokenizer.save_pretrained(output_path / "tokenizer")

    # Print summary
    config = AutoConfig.from_pretrained(load_from)
    client_size = sum(p.numel() * 2 for p in client_state_dict.values()) / 1e9
    print(f"\nSummary:")
    print(f"  Model: {model_id}")
    print(f"  Hidden dim: {config.hidden_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Client weights size: {client_size:.2f} GB")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return client_path


def tensorize_model(
    model_dir: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> Path:
    """Convert model to Tensorizer format for fast loading.

    Tensorizer enables ~5GB/s streaming from storage to GPU,
    significantly reducing cold start time.

    Args:
        model_dir: Path to model directory
        output_path: Path for tensorized output

    Returns:
        Path to tensorized weights
    """
    try:
        from tensorizer import TensorSerializer
    except ImportError:
        print("Tensorizer not installed. Skipping tensorization.")
        print("Install with: pip install tensorizer")
        return Path(model_dir)

    print(f"Loading model from {model_dir} for tensorization...")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    output = Path(output_path)
    print(f"Tensorizing to {output}...")

    serializer = TensorSerializer(str(output))
    serializer.write_module(model)
    serializer.close()

    print(f"Tensorized weights saved to {output}")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output


def prepare_model(
    model_id: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    output_dir: str = "/workspace/weights",
    tensorize: bool = False,
    device: str = "cuda",
) -> dict:
    """Full model preparation: download, extract client weights, optionally tensorize.

    Args:
        model_id: HuggingFace model ID
        output_dir: Base directory for all weights
        tensorize: Whether to create tensorized weights
        device: Device for model loading

    Returns:
        Dict with paths to all generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {}

    # Step 1: Download model
    model_dir = download_model(model_id, output_dir)
    result["model_dir"] = model_dir

    # Step 2: Extract client weights
    client_path = extract_client_weights(
        model_id=model_id,
        model_dir=model_dir,
        output_dir=output_dir,
        device=device,
    )
    result["client_weights"] = client_path

    # Step 3: Optionally tensorize for fast server loading
    if tensorize:
        tensor_path = output_path / "model.tensors"
        tensorize_model(model_dir, tensor_path, device)
        result["tensorized"] = tensor_path

    print("\n" + "=" * 60)
    print("MODEL PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Model directory: {result['model_dir']}")
    print(f"Client weights:  {result['client_weights']}")
    if "tensorized" in result:
        print(f"Tensorized:      {result['tensorized']}")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare model for Infemeral")
    parser.add_argument(
        "--model-id",
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/weights",
        help="Output directory for weights",
    )
    parser.add_argument(
        "--tensorize",
        action="store_true",
        help="Also create tensorized server weights for fast loading",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for model loading (AWQ models require 'cuda')",
    )
    parser.add_argument(
        "--client-only",
        action="store_true",
        help="Only extract client weights (assumes model already downloaded)",
    )

    args = parser.parse_args()

    if args.client_only:
        # Just extract client weights from existing model
        model_dir = Path(args.output_dir) / "model"
        if not model_dir.exists():
            print(f"Error: Model directory {model_dir} not found.")
            print("Run without --client-only first to download the model.")
            return
        extract_client_weights(
            model_id=args.model_id,
            model_dir=model_dir,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        # Full preparation
        prepare_model(
            model_id=args.model_id,
            output_dir=args.output_dir,
            tensorize=args.tensorize,
            device=args.device,
        )


if __name__ == "__main__":
    main()
