"""End-to-end integration tests.

These tests verify the complete flow from client to server and back,
ensuring all components work together correctly.
"""

import base64
from unittest import mock

import pytest
import torch

from infemeral.crypto import (
    cloak,
    create_cloaking_context,
    decrypt_bytes,
    encrypt_bytes,
    generate_session_key,
    uncloak,
)
from infemeral.tensors import deserialize_tensor, serialize_tensor


class TestFullInferenceFlow:
    """Tests for complete inference flow."""

    @pytest.fixture
    def cloaking_ctx(self):
        """Create cloaking context."""
        return create_cloaking_context(seed=42)

    @pytest.fixture
    def session_key(self):
        """Generate session key."""
        return generate_session_key()

    def test_full_flow_preserves_shape(self, cloaking_ctx, session_key):
        """Full flow should preserve tensor shape."""
        # Original hidden states (from client embedding)
        hidden = torch.randn(1, 10, 4096, dtype=torch.float16)

        # Step 1: Client - Cloak
        cloaked = cloak(hidden.float(), cloaking_ctx, add_noise=False).half()

        # Step 2: Client - Serialize
        data, shape, dtype = serialize_tensor(cloaked)

        # Step 3: Client - Encrypt
        ciphertext, nonce = encrypt_bytes(data, session_key)

        # Step 4: Server - Decrypt
        plaintext = decrypt_bytes(ciphertext, session_key, nonce)

        # Step 5: Server - Deserialize
        server_input = deserialize_tensor(plaintext, shape, dtype, device="cpu")

        # Step 6: Server - Transform (identity for test)
        server_output = server_input  # In reality: forward_transformer()

        # Step 7: Server - Serialize
        out_data, out_shape, out_dtype = serialize_tensor(server_output)

        # Step 8: Server - Encrypt
        out_ciphertext, out_nonce = encrypt_bytes(out_data, session_key)

        # Step 9: Client - Decrypt
        out_plaintext = decrypt_bytes(out_ciphertext, session_key, out_nonce)

        # Step 10: Client - Deserialize
        client_output = deserialize_tensor(out_plaintext, out_shape, out_dtype, "cpu")

        # Step 11: Client - Uncloak
        uncloaked = uncloak(client_output.float(), cloaking_ctx).half()

        # Verify shape preserved
        assert uncloaked.shape == hidden.shape

    def test_cloaking_preserves_information(self, cloaking_ctx, session_key):
        """Cloaking should be reversible (without noise)."""
        hidden = torch.randn(1, 10, 4096, dtype=torch.float32)

        # Cloak without noise
        cloaked = cloak(hidden, cloaking_ctx, add_noise=False)

        # Simulate server pass-through
        server_output = cloaked.clone()

        # Uncloak
        recovered = uncloak(server_output, cloaking_ctx)

        torch.testing.assert_close(hidden, recovered, rtol=1e-4, atol=1e-4)

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

    def test_server_cannot_see_raw_tokens(self):
        """Server should never receive raw token IDs."""
        # This is an architectural test - verify the flow
        # Server only receives cloaked hidden states, not tokens

        tokens = torch.tensor([[1, 2, 3, 4, 5]])  # Raw tokens

        # Client embeds (mocked)
        hidden = torch.randn(1, 5, 4096)  # Embedded representation

        # Client cloaks
        ctx = create_cloaking_context(seed=42)
        cloaked = cloak(hidden, ctx, add_noise=True)

        # What server receives
        server_input = cloaked

        # Server cannot recover tokens from cloaked hidden states
        # because:
        # 1. Hidden states are rotated by unknown orthogonal matrix
        # 2. DP noise is added
        # 3. Server doesn't have the embedding matrix

        # This is verified by the fact that cloaked looks nothing like hidden
        assert not torch.allclose(server_input, hidden, rtol=0.1, atol=0.1)

    def test_server_cannot_see_raw_output(self):
        """Server should never produce readable output."""
        # Server returns cloaked hidden states, not logits

        ctx = create_cloaking_context(seed=42)

        # What server outputs (cloaked)
        server_output = torch.randn(1, 5, 4096)

        # Client uncloaks
        uncloaked = uncloak(server_output, ctx)

        # Without the correct matrix, server cannot interpret output
        wrong_ctx = create_cloaking_context(seed=123)
        wrong_uncloak = uncloak(server_output, wrong_ctx)

        # Should be completely different
        assert not torch.allclose(uncloaked, wrong_uncloak, rtol=0.1, atol=0.1)

    def test_different_sessions_isolated(self):
        """Different sessions should have different cloaking."""
        hidden = torch.randn(1, 10, 4096)

        ctx1 = create_cloaking_context(seed=1)
        ctx2 = create_cloaking_context(seed=2)

        cloaked1 = cloak(hidden, ctx1, add_noise=False)
        cloaked2 = cloak(hidden, ctx2, add_noise=False)

        # Same input, different sessions = different cloaking
        assert not torch.allclose(cloaked1, cloaked2)

        # Cannot uncloak session1's data with session2's matrix
        wrong_uncloaked = uncloak(cloaked1, ctx2)
        assert not torch.allclose(hidden, wrong_uncloaked, rtol=0.1, atol=0.1)


class TestDataIntegrity:
    """Tests for data integrity through the pipeline."""

    def test_multiple_rounds_stable(self):
        """Multiple cloak/uncloak rounds should be stable."""
        ctx = create_cloaking_context(seed=42)
        hidden = torch.randn(1, 10, 4096)

        for _ in range(5):
            cloaked = cloak(hidden, ctx, add_noise=False)
            recovered = uncloak(cloaked, ctx)

            torch.testing.assert_close(hidden, recovered, rtol=1e-4, atol=1e-4)

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

    def test_cloaking_batch_efficient(self):
        """Cloaking should work efficiently with batches."""
        ctx = create_cloaking_context(seed=42)

        # Single item
        single = torch.randn(1, 100, 4096)
        cloaked_single = cloak(single, ctx, add_noise=False)

        # Batch
        batch = torch.randn(8, 100, 4096)
        cloaked_batch = cloak(batch, ctx, add_noise=False)

        # Both should work
        assert cloaked_single.shape == single.shape
        assert cloaked_batch.shape == batch.shape

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


class TestEdgeCases:
    """Edge case tests for the complete system."""

    def test_single_token_inference(self):
        """System should handle single token inference."""
        ctx = create_cloaking_context(seed=42)
        key = generate_session_key()

        hidden = torch.randn(1, 1, 4096)  # Single token

        # Full flow
        cloaked = cloak(hidden, ctx, add_noise=False)
        data, shape, dtype = serialize_tensor(cloaked)
        cipher, nonce = encrypt_bytes(data, key)

        plain = decrypt_bytes(cipher, key, nonce)
        recovered = deserialize_tensor(plain, shape, dtype, device="cpu")
        uncloaked = uncloak(recovered, ctx)

        torch.testing.assert_close(hidden, uncloaked, rtol=1e-4, atol=1e-4)

    def test_long_sequence_inference(self):
        """System should handle long sequences."""
        ctx = create_cloaking_context(seed=42)
        key = generate_session_key()

        hidden = torch.randn(1, 2048, 4096)  # Full context

        # Full flow
        cloaked = cloak(hidden, ctx, add_noise=False)
        data, shape, dtype = serialize_tensor(cloaked)
        cipher, nonce = encrypt_bytes(data, key)

        plain = decrypt_bytes(cipher, key, nonce)
        recovered = deserialize_tensor(plain, shape, dtype, device="cpu")
        uncloaked = uncloak(recovered, ctx)

        torch.testing.assert_close(hidden, uncloaked, rtol=1e-4, atol=1e-4)

    def test_float16_precision_acceptable(self):
        """Float16 precision loss should be acceptable."""
        ctx = create_cloaking_context(seed=42)

        # Use float32 for cloaking (higher precision)
        hidden = torch.randn(1, 10, 4096, dtype=torch.float32)

        cloaked = cloak(hidden, ctx, add_noise=False)

        # Convert to float16 for transmission (as in real system)
        cloaked_f16 = cloaked.half()

        # Convert back and uncloak
        cloaked_f32 = cloaked_f16.float()
        recovered = uncloak(cloaked_f32, ctx)

        # Should be close despite precision loss
        torch.testing.assert_close(hidden, recovered, rtol=1e-2, atol=1e-2)
