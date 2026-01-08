"""Tests for tensor serialization utilities."""

import numpy as np
import pytest
import torch

from infemeral.tensors import (
    deserialize_tensor,
    pack_kv_cache,
    serialize_tensor,
    unpack_kv_cache,
)


class TestTensorSerialization:
    """Tests for basic tensor serialization."""

    def test_serialize_deserialize_roundtrip(self):
        """Serialize and deserialize should recover original tensor."""
        tensor = torch.randn(2, 10, 4096)

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        torch.testing.assert_close(tensor, recovered)

    def test_float16_roundtrip(self):
        """Test float16 tensor serialization."""
        tensor = torch.randn(1, 5, 4096, dtype=torch.float16)

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        assert recovered.dtype == torch.float16
        torch.testing.assert_close(tensor, recovered)

    def test_float32_roundtrip(self):
        """Test float32 tensor serialization."""
        tensor = torch.randn(1, 5, 4096, dtype=torch.float32)

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        assert recovered.dtype == torch.float32
        torch.testing.assert_close(tensor, recovered)

    def test_bfloat16_roundtrip(self):
        """Test bfloat16 tensor serialization (converts to float16)."""
        tensor = torch.randn(1, 5, 4096, dtype=torch.bfloat16)

        data, shape, dtype = serialize_tensor(tensor)

        # dtype should indicate original was bfloat16
        assert dtype == "bfloat16"

        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        # Should be converted back to bfloat16
        assert recovered.dtype == torch.bfloat16

        # Values should be close (some precision loss from bf16->f16->bf16)
        torch.testing.assert_close(
            tensor.to(torch.float32),
            recovered.to(torch.float32),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_shape_preserved(self):
        """Tensor shape should be exactly preserved."""
        shapes = [
            (1, 1, 4096),
            (4, 128, 4096),
            (1, 2048, 4096),
            (8, 1, 4096),
        ]

        for shape in shapes:
            tensor = torch.randn(*shape)
            data, recovered_shape, dtype = serialize_tensor(tensor)
            assert tuple(recovered_shape) == shape

    def test_empty_tensor(self):
        """Handle empty tensors gracefully."""
        tensor = torch.randn(1, 0, 4096)

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        assert recovered.shape == tensor.shape

    def test_large_tensor(self):
        """Test serialization of large tensors."""
        # ~32MB tensor
        tensor = torch.randn(1, 1024, 4096, dtype=torch.float16)

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        torch.testing.assert_close(tensor, recovered)

    def test_data_is_bytes(self):
        """Serialized data should be bytes."""
        tensor = torch.randn(1, 10, 4096)
        data, _, _ = serialize_tensor(tensor)

        assert isinstance(data, bytes)

    def test_data_size_correct(self):
        """Serialized data size should match tensor size."""
        tensor = torch.randn(1, 10, 4096, dtype=torch.float16)
        data, _, _ = serialize_tensor(tensor)

        expected_size = tensor.numel() * 2  # float16 = 2 bytes
        assert len(data) == expected_size


class TestKVCacheSerialization:
    """Tests for KV cache packing/unpacking."""

    def test_kv_cache_roundtrip(self):
        """Pack and unpack should recover original KV tensors."""
        # Typical KV cache shape: [batch, num_heads, seq_len, head_dim]
        keys = torch.randn(1, 32, 128, 128)
        values = torch.randn(1, 32, 128, 128)

        packed = pack_kv_cache(keys, values)
        unpacked_keys, unpacked_values = unpack_kv_cache(packed, device="cpu")

        # Note: pack_kv_cache converts to float16
        torch.testing.assert_close(
            keys.to(torch.float16), unpacked_keys, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            values.to(torch.float16), unpacked_values, rtol=1e-3, atol=1e-3
        )

    def test_kv_cache_shape_preserved(self):
        """KV cache shape should be exactly preserved."""
        keys = torch.randn(2, 8, 256, 64)
        values = torch.randn(2, 8, 256, 64)

        packed = pack_kv_cache(keys, values)
        unpacked_keys, unpacked_values = unpack_kv_cache(packed, device="cpu")

        assert unpacked_keys.shape == keys.shape
        assert unpacked_values.shape == values.shape

    def test_kv_cache_different_shapes(self):
        """Test various KV cache shapes."""
        shapes = [
            (1, 32, 1, 128),  # Single token
            (1, 32, 2048, 128),  # Full context
            (4, 8, 64, 256),  # Different config
        ]

        for shape in shapes:
            keys = torch.randn(*shape)
            values = torch.randn(*shape)

            packed = pack_kv_cache(keys, values)
            unpacked_keys, unpacked_values = unpack_kv_cache(packed, device="cpu")

            assert unpacked_keys.shape == shape
            assert unpacked_values.shape == shape

    def test_kv_cache_mismatched_shapes_raises(self):
        """Mismatched key/value shapes should raise."""
        keys = torch.randn(1, 32, 128, 128)
        values = torch.randn(1, 32, 64, 128)  # Different seq_len

        with pytest.raises(AssertionError):
            pack_kv_cache(keys, values)

    def test_packed_format_structure(self):
        """Verify packed format has correct structure."""
        keys = torch.randn(1, 32, 128, 128)
        values = torch.randn(1, 32, 128, 128)

        packed = pack_kv_cache(keys, values)

        # Should be bytes
        assert isinstance(packed, bytes)

        # Should start with header (key_len + ndim + shape)
        import struct

        key_len = struct.unpack("<I", packed[0:4])[0]
        ndim = struct.unpack("<I", packed[4:8])[0]

        assert ndim == 4  # 4-dimensional tensor
        assert key_len == keys.numel() * 2  # float16

    def test_empty_kv_cache(self):
        """Handle empty KV cache (zero sequence length)."""
        keys = torch.randn(1, 32, 0, 128)
        values = torch.randn(1, 32, 0, 128)

        packed = pack_kv_cache(keys, values)
        unpacked_keys, unpacked_values = unpack_kv_cache(packed, device="cpu")

        assert unpacked_keys.shape == keys.shape
        assert unpacked_values.shape == values.shape


class TestSerializationEdgeCases:
    """Edge case tests for serialization."""

    def test_nan_values(self):
        """NaN values should be preserved."""
        tensor = torch.tensor([[[1.0, float("nan"), 3.0]]])

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        assert torch.isnan(recovered[0, 0, 1])
        assert recovered[0, 0, 0] == 1.0
        assert recovered[0, 0, 2] == 3.0

    def test_inf_values(self):
        """Infinity values should be preserved."""
        tensor = torch.tensor([[[float("inf"), float("-inf"), 0.0]]])

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        assert torch.isinf(recovered[0, 0, 0]) and recovered[0, 0, 0] > 0
        assert torch.isinf(recovered[0, 0, 1]) and recovered[0, 0, 1] < 0
        assert recovered[0, 0, 2] == 0.0

    def test_very_small_values(self):
        """Very small values should be handled (within dtype precision)."""
        tensor = torch.tensor([[[1e-7, 1e-10, 1e-15]]], dtype=torch.float32)

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        # float32 can handle these
        torch.testing.assert_close(tensor, recovered, rtol=1e-5, atol=1e-10)

    def test_very_large_values(self):
        """Very large values should be handled (within dtype precision)."""
        tensor = torch.tensor([[[1e30, 1e35, 1e38]]], dtype=torch.float32)

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        torch.testing.assert_close(tensor, recovered, rtol=1e-5, atol=1e30)

    def test_negative_values(self):
        """Negative values should be preserved."""
        tensor = torch.tensor([[[-1.0, -100.0, -0.001]]])

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        torch.testing.assert_close(tensor, recovered)

    def test_zeros(self):
        """Zero tensor should serialize correctly."""
        tensor = torch.zeros(1, 10, 4096)

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        assert torch.all(recovered == 0)


class TestSerializationPerformance:
    """Performance-related tests."""

    def test_serialization_deterministic(self):
        """Same tensor should produce same serialization."""
        tensor = torch.randn(1, 10, 4096)
        tensor_copy = tensor.clone()

        data1, _, _ = serialize_tensor(tensor)
        data2, _, _ = serialize_tensor(tensor_copy)

        assert data1 == data2

    def test_deserialization_creates_copy(self):
        """Deserialized tensor should be independent of source data."""
        tensor = torch.randn(1, 10, 4096)
        data, shape, dtype = serialize_tensor(tensor)

        recovered1 = deserialize_tensor(data, shape, dtype, device="cpu")
        recovered2 = deserialize_tensor(data, shape, dtype, device="cpu")

        # Modify one
        recovered1[0, 0, 0] = 999.0

        # Other should be unaffected
        assert recovered2[0, 0, 0] != 999.0
