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


def unpack_kv_cache(
    data: bytes,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack KV cache from byte buffer.

    Args:
        data: Packed bytes from pack_kv_cache
        device: Target device

    Returns:
        Tuple of (keys, values) tensors
    """
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
