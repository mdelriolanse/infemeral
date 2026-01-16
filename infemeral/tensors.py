"""Tensor serialization utilities for gRPC transport."""

import struct
from typing import Literal

import numpy as np
import torch

# Supported dtypes for serialization
DTYPE_MAP: dict[str, np.dtype] = {
    "float16": np.float16,
    "float32": np.float32,
    "bfloat16": np.float16,  # bfloat16 serialized as float16
}

TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def serialize_tensor(tensor: torch.Tensor) -> tuple[bytes, list[int], str]:
    """Serialize a PyTorch tensor to bytes.

    Args:
        tensor: Input tensor

    Returns:
        Tuple of (data_bytes, shape, dtype_str)
    """
    # Convert bfloat16 to float16 for serialization (bfloat16 not in numpy)
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float16)
        dtype_str = "bfloat16"  # Remember original dtype
    else:
        dtype_str = str(tensor.dtype).split(".")[-1]  # "torch.float16" -> "float16"

    data = tensor.detach().cpu().numpy().tobytes()
    shape = list(tensor.shape)

    return data, shape, dtype_str


def deserialize_tensor(
    data: bytes,
    shape: list[int],
    dtype_str: str,
    device: str = "cuda",
) -> torch.Tensor:
    """Deserialize bytes to a PyTorch tensor.

    Args:
        data: Raw bytes
        shape: Tensor shape
        dtype_str: Dtype string (e.g., "float16")
        device: Target device

    Returns:
        Reconstructed tensor
    """
    np_dtype = DTYPE_MAP.get(dtype_str, np.float32)
    arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    tensor = torch.from_numpy(arr.copy())

    # Convert back to bfloat16 if that was the original dtype
    if dtype_str == "bfloat16":
        tensor = tensor.to(torch.bfloat16)
    else:
        tensor = tensor.to(TORCH_DTYPE_MAP.get(dtype_str, torch.float32))

    return tensor.to(device)


def pack_kv_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
) -> bytes:
    """Pack KV cache tensors into a single byte buffer.

    Format:
        - 4 bytes: key data length (uint32)
        - 4 bytes: num dimensions (uint32)
        - 8 bytes * ndim: shape (int64 each)
        - key_len bytes: key data
        - value data (same shape, immediately follows)

    Args:
        keys: Key tensor
        values: Value tensor (must have same shape)

    Returns:
        Packed bytes
    """
    assert keys.shape == values.shape, "Key and value shapes must match"

    key_data = keys.detach().cpu().to(torch.float16).numpy().tobytes()
    val_data = values.detach().cpu().to(torch.float16).numpy().tobytes()

    shape = keys.shape
    header = struct.pack("<I", len(key_data))  # key data length
    header += struct.pack("<I", len(shape))  # num dimensions
    for dim in shape:
        header += struct.pack("<q", dim)  # shape values

    return header + key_data + val_data


def pack_kv_cache_v2(
    kv_tuples: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> bytes:
    """Pack per-layer KV cache into versioned byte buffer.

    Format:
        - 1 byte: version (0x02)
        - 4 bytes: num_layers (uint32)
        For each layer:
            - 4 bytes: key data length (uint32)
            - 4 bytes: num dimensions (uint32)
            - 8 bytes * ndim: shape (int64 each)
            - key_len bytes: key data
            - 4 bytes: value data length (uint32)
            - 4 bytes: num dimensions (uint32)
            - 8 bytes * ndim: shape (int64 each)
            - val_len bytes: value data

    Args:
        kv_tuples: Tuple of (key, value) pairs for each layer

    Returns:
        Packed bytes with version header
    """
    # Version header
    buffer = struct.pack("<B", 0x02)  # Version 2
    buffer += struct.pack("<I", len(kv_tuples))  # Number of layers

    for keys, values in kv_tuples:
        # Pack keys
        key_data = keys.detach().cpu().to(torch.float16).numpy().tobytes()
        key_shape = keys.shape
        buffer += struct.pack("<I", len(key_data))
        buffer += struct.pack("<I", len(key_shape))
        for dim in key_shape:
            buffer += struct.pack("<q", dim)
        buffer += key_data

        # Pack values
        val_data = values.detach().cpu().to(torch.float16).numpy().tobytes()
        val_shape = values.shape
        buffer += struct.pack("<I", len(val_data))
        buffer += struct.pack("<I", len(val_shape))
        for dim in val_shape:
            buffer += struct.pack("<q", dim)
        buffer += val_data

    return buffer


def unpack_kv_cache_v2(
    data: bytes,
    device: str = "cuda",
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Unpack per-layer KV cache from versioned byte buffer.

    Args:
        data: Packed bytes from pack_kv_cache_v2
        device: Target device

    Returns:
        Tuple of (key, value) pairs for each layer
    """
    pos = 0

    # Read version
    version = struct.unpack("<B", data[pos : pos + 1])[0]
    pos += 1
    if version != 0x02:
        raise ValueError(f"Expected version 0x02, got {version:#04x}")

    # Read number of layers
    num_layers = struct.unpack("<I", data[pos : pos + 4])[0]
    pos += 4

    layers = []
    for _ in range(num_layers):
        # Unpack key
        key_len = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        key_ndim = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        key_shape = []
        for _ in range(key_ndim):
            key_shape.append(struct.unpack("<q", data[pos : pos + 8])[0])
            pos += 8
        key_data = data[pos : pos + key_len]
        pos += key_len
        keys = torch.from_numpy(
            np.frombuffer(key_data, dtype=np.float16).reshape(key_shape).copy()
        ).to(device)

        # Unpack value
        val_len = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        val_ndim = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        val_shape = []
        for _ in range(val_ndim):
            val_shape.append(struct.unpack("<q", data[pos : pos + 8])[0])
            pos += 8
        val_data = data[pos : pos + val_len]
        pos += val_len
        values = torch.from_numpy(
            np.frombuffer(val_data, dtype=np.float16).reshape(val_shape).copy()
        ).to(device)

        layers.append((keys, values))

    return tuple(layers)


def unpack_kv_cache(
    data: bytes,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor] | tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Unpack KV cache from byte buffer with version detection.

    Supports both v1 (flat) and v2 (per-layer) formats for backward compatibility.

    Args:
        data: Packed bytes from pack_kv_cache or pack_kv_cache_v2
        device: Target device

    Returns:
        V1: Tuple of (keys, values) tensors
        V2: Tuple of (key, value) pairs for each layer
    """
    # Check if this is v2 format (starts with version byte 0x02)
    if len(data) > 0 and data[0] == 0x02:
        return unpack_kv_cache_v2(data, device)

    # V1 format (legacy)
    pos = 0

    # Read header
    key_len = struct.unpack("<I", data[pos : pos + 4])[0]
    pos += 4
    ndim = struct.unpack("<I", data[pos : pos + 4])[0]
    pos += 4

    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack("<q", data[pos : pos + 8])[0])
        pos += 8

    # Read key and value data
    key_data = data[pos : pos + key_len]
    pos += key_len
    val_data = data[pos : pos + key_len]  # Same length as keys

    # Reconstruct tensors
    keys = torch.from_numpy(
        np.frombuffer(key_data, dtype=np.float16).reshape(shape).copy()
    ).to(device)
    values = torch.from_numpy(
        np.frombuffer(val_data, dtype=np.float16).reshape(shape).copy()
    ).to(device)

    return keys, values
