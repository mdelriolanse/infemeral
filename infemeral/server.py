"""Server-side inference: transformer blocks on cloaked hidden states.

The server processes cloaked embeddings through transformer layers,
never seeing raw tokens or being able to reconstruct user intent.

Uses vLLM with AWQ quantization for efficient 4-bit inference.
"""

import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from infemeral.config import server_settings
from infemeral.crypto import decrypt_bytes, encrypt_bytes
from infemeral.tensors import (
    deserialize_tensor,
    pack_kv_cache,
    serialize_tensor,
    unpack_kv_cache,
)

# Global model instance (persists across serverless invocations)
_model = None
_config = None
_vllm_engine = None


def load_model_vllm(model_id: str = "TheBloke/LLaMA-Pro-8B-AWQ"):
    """Load model using vLLM with AWQ 4-bit quantization.

    This is the preferred method for AWQ models as vLLM handles
    the quantized weights natively with optimized kernels.
    """
    global _vllm_engine

    if _vllm_engine is not None:
        return _vllm_engine

    try:
        from vllm import LLM

        print(f"Loading {model_id} with vLLM (AWQ quantization)...")
        _vllm_engine = LLM(
            model=model_id,
            quantization="awq",
            dtype="float16",
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )
        print("vLLM engine loaded successfully")
        return _vllm_engine

    except ImportError:
        print("vLLM not available, falling back to transformers")
        return None


def load_model(weights_path: str, device: str | None = None) -> torch.nn.Module:
    """Load transformer model from weights.

    Tries vLLM first for AWQ models, falls back to transformers.
    """
    global _model, _config

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _model is not None:
        return _model

    # For AWQ models, prefer vLLM
    if "AWQ" in server_settings.weights_path or "awq" in server_settings.weights_path:
        vllm_engine = load_model_vllm()
        if vllm_engine is not None:
            # vLLM handles the model internally
            # Return a wrapper that exposes the model for our forward_transformer
            _model = vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
            _config = AutoConfig.from_pretrained(
                "TheBloke/LLaMA-Pro-8B-AWQ", trust_remote_code=True
            )
            return _model

    print(f"Loading model from {weights_path}...")

    # Try tensorizer first for fast loading
    if weights_path.endswith(".tensors"):
        try:
            from tensorizer import TensorDeserializer

            print("Using Tensorizer for fast model loading...")
            deserializer = TensorDeserializer(weights_path, device=device)
            _config = AutoConfig.from_pretrained(
                Path(weights_path).parent, trust_remote_code=True
            )
            _model = AutoModelForCausalLM.from_config(
                _config, torch_dtype=torch.float16
            )
            deserializer.load_into_module(_model)
            deserializer.close()
        except ImportError:
            print("Tensorizer not available, falling back to safetensors")
            weights_path = weights_path.replace(".tensors", ".safetensors")

    # Fallback to safetensors
    if _model is None:
        config_path = Path(weights_path).parent
        _config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # Load model without weights first
        _model = AutoModelForCausalLM.from_config(_config, torch_dtype=torch.float16)

        # Load server weights (transformer layers only)
        state_dict = load_file(weights_path)
        missing, unexpected = _model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    _model = _model.to(device)
    _model.eval()
    print(f"Model loaded. Hidden size: {_config.hidden_size}")

    return _model


def get_kv_cache_path(session_id: str) -> Path:
    """Get path for session's KV cache."""
    cache_dir = Path(server_settings.kv_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{session_id}.bin"


def load_kv_cache(
    session_id: str,
    session_key: bytes,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load encrypted KV cache from Network Volume.

    Returns None if no cache exists for this session.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_path = get_kv_cache_path(session_id)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            # Read nonce (first 12 bytes)
            nonce = f.read(12)
            ciphertext = f.read()

        # Decrypt
        plaintext = decrypt_bytes(ciphertext, session_key, nonce)

        # Unpack KV tensors
        keys, values = unpack_kv_cache(plaintext, device)
        return keys, values

    except Exception as e:
        print(f"Failed to load KV cache: {e}")
        return None


def save_kv_cache(
    session_id: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    session_key: bytes,
) -> None:
    """Save encrypted KV cache to Network Volume."""
    cache_path = get_kv_cache_path(session_id)

    # Pack and encrypt
    plaintext = pack_kv_cache(keys, values)
    ciphertext, nonce = encrypt_bytes(plaintext, session_key)

    # Write nonce + ciphertext
    with open(cache_path, "wb") as f:
        f.write(nonce)
        f.write(ciphertext)


def delete_kv_cache(session_id: str) -> None:
    """Delete KV cache for a session."""
    cache_path = get_kv_cache_path(session_id)
    if cache_path.exists():
        cache_path.unlink()


def forward_transformer(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    past_key_values: tuple | None = None,
) -> tuple[torch.Tensor, tuple]:
    """Forward pass through transformer layers only.

    This bypasses the embedding layer and directly feeds
    hidden states into the transformer blocks.
    """
    # Get the transformer layers
    if hasattr(model, "model"):
        # Llama-style architecture
        transformer = model.model
        layers = transformer.layers
        norm = transformer.norm
    else:
        raise ValueError("Unsupported model architecture")

    # Prepare attention mask and position IDs
    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device

    # Create causal attention mask
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # Position IDs
    if past_key_values is not None:
        past_len = past_key_values[0][0].shape[2]
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=device
        ).unsqueeze(0)
    else:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # Forward through each transformer layer
    new_key_values = []

    for i, layer in enumerate(layers):
        past_kv = past_key_values[i] if past_key_values else None

        layer_outputs = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_kv,
            use_cache=True,
        )

        hidden_states = layer_outputs[0]
        new_key_values.append(layer_outputs[1])

    # Final layer norm
    hidden_states = norm(hidden_states)

    return hidden_states, tuple(new_key_values)


def handler(event: dict) -> dict:
    """RunPod serverless handler.

    Receives cloaked embeddings, runs transformer inference,
    returns transformed hidden states.

    Input event:
        {
            "input": {
                "cloaked_embedding": bytes (base64),
                "encrypted_session_key": bytes,
                "nonce": bytes,
                "shape": [batch, seq_len, hidden_dim],
                "dtype": str,
                "session_id": str,
            }
        }

    Output:
        {
            "output": bytes (base64),
            "shape": [batch, seq_len, hidden_dim],
            "dtype": str,
        }
    """
    try:
        inp = event.get("input", event)

        # Load model on first request (cold start)
        model = load_model(server_settings.weights_path)
        device = next(model.parameters()).device

        # Decrypt input tensor
        session_key = inp["encrypted_session_key"]
        if isinstance(session_key, str):
            import base64
            session_key = base64.b64decode(session_key)

        nonce = inp["nonce"]
        if isinstance(nonce, str):
            import base64
            nonce = base64.b64decode(nonce)

        cloaked_data = inp["cloaked_embedding"]
        if isinstance(cloaked_data, str):
            import base64
            cloaked_data = base64.b64decode(cloaked_data)

        plaintext = decrypt_bytes(cloaked_data, session_key, nonce)

        # Deserialize tensor
        cloaked = deserialize_tensor(
            plaintext,
            inp["shape"],
            inp["dtype"],
            device=str(device),
        )

        session_id = inp["session_id"]

        # Load KV cache if exists
        kv_cache = load_kv_cache(session_id, session_key, str(device))
        past_key_values = None
        if kv_cache is not None:
            # TODO: Convert to proper past_key_values format
            pass

        # Forward through transformer
        with torch.no_grad():
            output, new_kv = forward_transformer(model, cloaked, past_key_values)

        # Save updated KV cache
        # TODO: Implement tide-windowing for context > 2048
        # save_kv_cache(session_id, new_kv[0], new_kv[1], session_key)

        # Serialize output
        output_data, output_shape, output_dtype = serialize_tensor(output)

        # Encrypt output
        encrypted_output, output_nonce = encrypt_bytes(output_data, session_key)

        # Memory wipe
        del cloaked, output, kv_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import base64
        return {
            "output": base64.b64encode(encrypted_output).decode(),
            "nonce": base64.b64encode(output_nonce).decode(),
            "shape": output_shape,
            "dtype": output_dtype,
            "tokens_processed": inp["shape"][1],
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# RunPod entry point
if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
