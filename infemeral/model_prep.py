"""One-time model preparation: split model into client/server components.

This script:
1. Downloads the base model (LLaMA-Pro-8B-AWQ)
2. Extracts embedding layers for client (embed_tokens + lm_head)
3. Tensorizes transformer layers for server (fast GPU loading)

Usage:
    python -m infemeral.model_prep --output-dir ./weights
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AwqConfig


def split_model(
    model_id: str = "TheBloke/LLaMA-Pro-8B-AWQ",
    output_dir: str = "./weights",
    device: str = "cpu",
) -> tuple[Path, Path]:
    """Split a causal LM into client and server components.

    Client gets: embed_tokens, lm_head (semantic entry/exit points)
    Server gets: transformer layers (blind computation)

    Args:
        model_id: HuggingFace model ID
        output_dir: Directory to save weights
        device: Device for loading model

    Returns:
        Tuple of (client_weights_path, server_weights_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {model_id}...")
    config = AutoConfig.from_pretrained(model_id)

    # Use GEMM backend for AWQ to avoid IPEX version conflicts
    quantization_config = AwqConfig(version="GEMM")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )

    # Also save tokenizer for client use
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_path / "tokenizer")

    # Extract client components (embedding + lm_head)
    print("Extracting client components...")
    client_state_dict = {}

    # Handle different model architectures
    if hasattr(model, "model"):
        # Llama-style: model.model.embed_tokens, model.lm_head
        client_state_dict["embed_tokens.weight"] = (
            model.model.embed_tokens.weight.data.clone().cpu()
        )
        if hasattr(model, "lm_head"):
            client_state_dict["lm_head.weight"] = model.lm_head.weight.data.clone().cpu()
    else:
        # GPT-style: model.wte, model.lm_head
        if hasattr(model, "transformer"):
            client_state_dict["embed_tokens.weight"] = (
                model.transformer.wte.weight.data.clone().cpu()
            )
        if hasattr(model, "lm_head"):
            client_state_dict["lm_head.weight"] = model.lm_head.weight.data.clone().cpu()

    client_path = output_path / "client_weights.safetensors"
    save_file(client_state_dict, client_path)
    print(f"Saved client weights to {client_path}")

    # Extract server components (transformer layers + norms)
    print("Extracting server components...")
    server_state_dict = {}

    for name, param in model.named_parameters():
        # Skip embedding and lm_head (those go to client)
        if "embed_tokens" in name or "lm_head" in name or "wte" in name:
            continue
        server_state_dict[name] = param.data.clone().cpu()

    server_path = output_path / "server_weights.safetensors"
    save_file(server_state_dict, server_path)
    print(f"Saved server weights to {server_path}")

    # Save config for both client and server
    config.save_pretrained(output_path)
    print(f"Saved config to {output_path}")

    # Print summary
    client_size = sum(p.numel() * 2 for p in client_state_dict.values()) / 1e9
    server_size = sum(p.numel() * 2 for p in server_state_dict.values()) / 1e9
    print(f"\nSummary:")
    print(f"  Client weights: {client_size:.2f} GB")
    print(f"  Server weights: {server_size:.2f} GB")
    print(f"  Hidden dim: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")

    return client_path, server_path


def tensorize_server_weights(
    server_weights_path: str,
    output_path: str,
    model_id: str = "TheBloke/LLaMA-Pro-8B-AWQ",
) -> Path:
    """Convert server weights to Tensorizer format for fast loading.

    Tensorizer enables ~5GB/s streaming from storage to GPU,
    which significantly reduces cold start time on RunPod.

    Args:
        server_weights_path: Path to server safetensors file
        output_path: Path for tensorized output
        model_id: Model ID for architecture

    Returns:
        Path to tensorized weights
    """
    try:
        from tensorizer import TensorSerializer
    except ImportError:
        print("Tensorizer not installed. Skipping tensorization.")
        print("Install with: pip install tensorizer")
        return Path(server_weights_path)

    from safetensors.torch import load_file

    print(f"Loading server weights from {server_weights_path}...")
    state_dict = load_file(server_weights_path)

    # Load model architecture without weights
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    # Load server weights (skip missing embed/lm_head)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Tensorize
    output = Path(output_path)
    print(f"Tensorizing to {output}...")

    serializer = TensorSerializer(str(output))
    serializer.write_module(model)
    serializer.close()

    print(f"Tensorized weights saved to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Prepare model for Infemeral")
    parser.add_argument(
        "--model-id",
        default="TheBloke/LLaMA-Pro-8B-AWQ",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default="./weights",
        help="Output directory for weights",
    )
    parser.add_argument(
        "--tensorize",
        action="store_true",
        help="Also create tensorized server weights",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model loading",
    )

    args = parser.parse_args()

    # Split model
    client_path, server_path = split_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Optionally tensorize
    if args.tensorize:
        tensorize_server_weights(
            server_weights_path=str(server_path),
            output_path=str(Path(args.output_dir) / "server_weights.tensors"),
            model_id=args.model_id,
        )


if __name__ == "__main__":
    main()
