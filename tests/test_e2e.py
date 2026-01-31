"""End-to-end integration tests.

These tests verify the complete flow from client to server and back,
ensuring all components work together correctly.
"""

import base64
from unittest import mock

import pytest
import torch

from infemeral.crypto import (
    decrypt_bytes,
    encrypt_bytes,
    generate_session_key,
)
from infemeral.tensors import deserialize_tensor, serialize_tensor


class TestFullInferenceFlow:
    """Tests for complete inference flow."""

    @pytest.fixture
    def session_key(self):
        """Generate session key."""
        return generate_session_key()

    def test_full_flow_preserves_shape(self, session_key):
        """Full flow should preserve tensor shape."""
        # Original hidden states (from client embedding)
        hidden = torch.randn(1, 10, 4096, dtype=torch.float16)

        # Step 1: Client - Serialize
        data, shape, dtype = serialize_tensor(hidden)

        # Step 2: Client - Encrypt
        ciphertext, nonce = encrypt_bytes(data, session_key)

        # Step 3: Server - Decrypt
        plaintext = decrypt_bytes(ciphertext, session_key, nonce)

        # Step 4: Server - Deserialize
        server_input = deserialize_tensor(plaintext, shape, dtype, device="cpu")

        # Step 5: Server - Transform (identity for test)
        server_output = server_input  # In reality: forward_transformer()

        # Step 6: Server - Serialize
        out_data, out_shape, out_dtype = serialize_tensor(server_output)

        # Step 7: Server - Encrypt
        out_ciphertext, out_nonce = encrypt_bytes(out_data, session_key)

        # Step 8: Client - Decrypt
        out_plaintext = decrypt_bytes(out_ciphertext, session_key, out_nonce)

        # Step 9: Client - Deserialize
        client_output = deserialize_tensor(out_plaintext, out_shape, out_dtype, "cpu")

        # Verify shape preserved
        assert client_output.shape == hidden.shape

        # Verify values preserved (identity transform)
        torch.testing.assert_close(client_output, hidden)

    def test_encryption_protects_data(self, session_key):
        """Encryption should protect tensor data."""
        tensor = torch.randn(1, 10, 4096)
        data, shape, dtype = serialize_tensor(tensor)

        # Encrypt
        ciphertext, nonce = encrypt_bytes(data, session_key)

        # Ciphertext should not reveal tensor values
        assert ciphertext != data

        # Wrong key should fail
        wrong_key = generate_session_key()
        with pytest.raises(Exception):
            decrypt_bytes(ciphertext, wrong_key, nonce)


class TestSecurityInvariants:
    """Tests verifying security properties of the system."""

    def test_different_sessions_isolated(self):
        """Different sessions should have different encryption keys."""
        hidden = torch.randn(1, 10, 4096)
        data, shape, dtype = serialize_tensor(hidden)

        key1 = generate_session_key()
        key2 = generate_session_key()

        # Encrypt with key1
        ciphertext1, nonce1 = encrypt_bytes(data, key1)

        # Cannot decrypt with key2
        with pytest.raises(Exception):
            decrypt_bytes(ciphertext1, key2, nonce1)

    def test_session_keys_unique(self):
        """Each session should have unique keys."""
        keys = [generate_session_key() for _ in range(100)]

        # All keys should be unique
        assert len(set(keys)) == 100


class TestDataIntegrity:
    """Tests for data integrity through the pipeline."""

    def test_serialization_preserves_values(self):
        """Serialization should exactly preserve tensor values."""
        tensor = torch.randn(1, 10, 4096, dtype=torch.float32)

        data, shape, dtype = serialize_tensor(tensor)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        torch.testing.assert_close(tensor, recovered)

    def test_encryption_preserves_values(self):
        """Encryption should exactly preserve data."""
        key = generate_session_key()
        original = b"test data " * 1000

        ciphertext, nonce = encrypt_bytes(original, key)
        recovered = decrypt_bytes(ciphertext, key, nonce)

        assert recovered == original

    def test_full_roundtrip_exact(self):
        """Full serialize -> encrypt -> decrypt -> deserialize should be exact."""
        key = generate_session_key()
        tensor = torch.randn(1, 10, 4096, dtype=torch.float16)

        # Forward
        data, shape, dtype = serialize_tensor(tensor)
        ciphertext, nonce = encrypt_bytes(data, key)

        # Backward
        plaintext = decrypt_bytes(ciphertext, key, nonce)
        recovered = deserialize_tensor(plaintext, shape, dtype, device="cpu")

        torch.testing.assert_close(tensor, recovered)


class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_mismatched_shapes_detected(self):
        """Mismatched shapes should be detected."""
        tensor = torch.randn(1, 10, 4096)
        data, shape, dtype = serialize_tensor(tensor)

        # Try to deserialize with wrong shape
        wrong_shape = [1, 20, 4096]  # Wrong seq_len

        with pytest.raises(Exception):
            deserialize_tensor(data, wrong_shape, dtype, device="cpu")

    def test_corrupted_ciphertext_detected(self):
        """Corrupted ciphertext should be detected (AES-GCM auth)."""
        key = generate_session_key()
        data = b"test data"

        ciphertext, nonce = encrypt_bytes(data, key)

        # Corrupt the ciphertext
        corrupted = bytearray(ciphertext)
        corrupted[0] ^= 0xFF
        corrupted = bytes(corrupted)

        with pytest.raises(Exception):
            decrypt_bytes(corrupted, key, nonce)


class TestPerformanceCharacteristics:
    """Tests for performance-related characteristics."""

    def test_serialization_size_predictable(self):
        """Serialized size should be predictable from tensor shape."""
        shapes = [
            (1, 10, 4096),
            (1, 100, 4096),
            (4, 50, 4096),
        ]

        for shape in shapes:
            tensor = torch.randn(*shape, dtype=torch.float16)
            data, _, _ = serialize_tensor(tensor)

            expected_size = tensor.numel() * 2  # float16 = 2 bytes
            assert len(data) == expected_size

    def test_batch_handling(self):
        """System should handle batched tensors."""
        key = generate_session_key()

        # Single item
        single = torch.randn(1, 100, 4096, dtype=torch.float16)
        single_data, single_shape, single_dtype = serialize_tensor(single)
        single_cipher, single_nonce = encrypt_bytes(single_data, key)

        # Batch
        batch = torch.randn(8, 100, 4096, dtype=torch.float16)
        batch_data, batch_shape, batch_dtype = serialize_tensor(batch)
        batch_cipher, batch_nonce = encrypt_bytes(batch_data, key)

        # Both should round-trip correctly
        single_recovered = deserialize_tensor(
            decrypt_bytes(single_cipher, key, single_nonce),
            single_shape, single_dtype, device="cpu"
        )
        batch_recovered = deserialize_tensor(
            decrypt_bytes(batch_cipher, key, batch_nonce),
            batch_shape, batch_dtype, device="cpu"
        )

        torch.testing.assert_close(single, single_recovered)
        torch.testing.assert_close(batch, batch_recovered)


class TestEdgeCases:
    """Edge case tests for the complete system."""

    def test_single_token_inference(self):
        """System should handle single token inference."""
        key = generate_session_key()

        hidden = torch.randn(1, 1, 4096, dtype=torch.float16)  # Single token

        # Full flow
        data, shape, dtype = serialize_tensor(hidden)
        cipher, nonce = encrypt_bytes(data, key)

        plain = decrypt_bytes(cipher, key, nonce)
        recovered = deserialize_tensor(plain, shape, dtype, device="cpu")

        torch.testing.assert_close(hidden, recovered)

    def test_long_sequence_inference(self):
        """System should handle long sequences."""
        key = generate_session_key()

        hidden = torch.randn(1, 2048, 4096, dtype=torch.float16)  # Full context

        # Full flow
        data, shape, dtype = serialize_tensor(hidden)
        cipher, nonce = encrypt_bytes(data, key)

        plain = decrypt_bytes(cipher, key, nonce)
        recovered = deserialize_tensor(plain, shape, dtype, device="cpu")

        torch.testing.assert_close(hidden, recovered)

    def test_float16_precision(self):
        """Float16 serialization should maintain precision."""
        key = generate_session_key()

        hidden = torch.randn(1, 10, 4096, dtype=torch.float16)

        data, shape, dtype = serialize_tensor(hidden)
        cipher, nonce = encrypt_bytes(data, key)

        plain = decrypt_bytes(cipher, key, nonce)
        recovered = deserialize_tensor(plain, shape, dtype, device="cpu")

        # Should be exact for float16
        torch.testing.assert_close(hidden, recovered)

    def test_float32_precision(self):
        """Float32 serialization should maintain precision."""
        key = generate_session_key()

        hidden = torch.randn(1, 10, 4096, dtype=torch.float32)

        data, shape, dtype = serialize_tensor(hidden)
        cipher, nonce = encrypt_bytes(data, key)

        plain = decrypt_bytes(cipher, key, nonce)
        recovered = deserialize_tensor(plain, shape, dtype, device="cpu")

        # Should be exact for float32
        torch.testing.assert_close(hidden, recovered)
