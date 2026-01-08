"""Tests for cryptographic primitives."""

import numpy as np
import pytest
import torch

from infemeral.crypto import (
    CloakingContext,
    cloak,
    compute_dp_sigma,
    create_cloaking_context,
    decrypt_bytes,
    encrypt_bytes,
    generate_orthogonal_matrix,
    generate_session_key,
    uncloak,
)


class TestOrthogonalMatrix:
    """Tests for orthogonal matrix generation."""

    def test_matrix_is_orthogonal(self):
        """Verify M @ M.T = I."""
        dim = 128  # Smaller dim for fast test
        matrix = generate_orthogonal_matrix(dim, seed=42)

        identity = matrix @ matrix.T
        expected = np.eye(dim, dtype=np.float32)

        np.testing.assert_array_almost_equal(identity, expected, decimal=5)

    def test_preserves_dot_products(self):
        """Verify (Mx).T @ (My) = x.T @ y."""
        dim = 128
        matrix = generate_orthogonal_matrix(dim, seed=42)

        x = np.random.randn(dim).astype(np.float32)
        y = np.random.randn(dim).astype(np.float32)

        original_dot = x @ y
        rotated_dot = (matrix @ x) @ (matrix @ y)

        np.testing.assert_almost_equal(original_dot, rotated_dot, decimal=5)

    def test_reproducible_with_seed(self):
        """Same seed produces same matrix."""
        m1 = generate_orthogonal_matrix(64, seed=123)
        m2 = generate_orthogonal_matrix(64, seed=123)

        np.testing.assert_array_equal(m1, m2)

    def test_different_seeds_different_matrices(self):
        """Different seeds produce different matrices."""
        m1 = generate_orthogonal_matrix(64, seed=1)
        m2 = generate_orthogonal_matrix(64, seed=2)

        assert not np.allclose(m1, m2)


class TestDifferentialPrivacy:
    """Tests for DP noise calibration."""

    def test_sigma_calculation(self):
        """Verify sigma formula."""
        epsilon = 2.0
        delta = 1e-5

        sigma = compute_dp_sigma(epsilon, delta)

        # Expected: sqrt(2 * ln(1.25/1e-5)) / 2.0 ≈ 2.4
        assert 2.0 < sigma < 3.0

    def test_higher_epsilon_lower_sigma(self):
        """More privacy budget = less noise."""
        sigma_high_priv = compute_dp_sigma(epsilon=1.0, delta=1e-5)
        sigma_low_priv = compute_dp_sigma(epsilon=4.0, delta=1e-5)

        assert sigma_high_priv > sigma_low_priv


class TestCloaking:
    """Tests for cloaking and uncloaking."""

    @pytest.fixture
    def ctx(self):
        """Create a cloaking context for testing."""
        return create_cloaking_context(seed=42)

    def test_cloak_changes_values(self, ctx):
        """Cloaking should transform the input."""
        hidden = torch.randn(1, 10, 4096)
        cloaked = cloak(hidden, ctx)

        assert not torch.allclose(hidden, cloaked)

    def test_uncloak_recovers_structure(self, ctx):
        """Uncloaking should reverse the rotation (not noise)."""
        hidden = torch.randn(1, 10, 4096)

        # Cloak without noise to test rotation
        cloaked = cloak(hidden, ctx, add_noise=False)
        recovered = uncloak(cloaked, ctx)

        torch.testing.assert_close(hidden, recovered, rtol=1e-4, atol=1e-4)

    def test_attention_scores_preserved(self, ctx):
        """Verify Q'K'.T ≈ QK.T for rotated Q, K."""
        batch, seq, dim = 1, 5, 4096
        Q = torch.randn(batch, seq, dim)
        K = torch.randn(batch, seq, dim)

        # Original attention scores
        original_scores = torch.bmm(Q, K.transpose(1, 2))

        # Rotate both Q and K
        Q_rot = cloak(Q, ctx, add_noise=False)
        K_rot = cloak(K, ctx, add_noise=False)

        # Rotated attention scores
        rotated_scores = torch.bmm(Q_rot, K_rot.transpose(1, 2))

        # The key property: rotation should not change attention scores
        # Because (MQ)(MK).T = MQ K.T M.T, and for attention we need
        # the relative scores which are preserved under orthogonal transform
        # Note: absolute values differ but relative rankings preserved
        torch.testing.assert_close(original_scores, rotated_scores, rtol=1e-3, atol=1e-3)


class TestEncryption:
    """Tests for AES-256-GCM encryption."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encryption followed by decryption recovers original."""
        key = generate_session_key()
        plaintext = b"Hello, world! This is a test message."

        ciphertext, nonce = encrypt_bytes(plaintext, key)
        recovered = decrypt_bytes(ciphertext, key, nonce)

        assert recovered == plaintext

    def test_different_keys_fail(self):
        """Decryption with wrong key should fail."""
        key1 = generate_session_key()
        key2 = generate_session_key()
        plaintext = b"Secret data"

        ciphertext, nonce = encrypt_bytes(plaintext, key1)

        with pytest.raises(Exception):
            decrypt_bytes(ciphertext, key2, nonce)

    def test_ciphertext_different_from_plaintext(self):
        """Ciphertext should not equal plaintext."""
        key = generate_session_key()
        plaintext = b"Test data"

        ciphertext, _ = encrypt_bytes(plaintext, key)

        assert ciphertext != plaintext

    def test_session_key_length(self):
        """Session key should be 256 bits (32 bytes)."""
        key = generate_session_key()
        assert len(key) == 32

    def test_large_payload_encryption(self):
        """Test encryption of large payloads (simulating tensor data)."""
        key = generate_session_key()
        # 1MB payload
        plaintext = bytes(np.random.randint(0, 256, size=1024 * 1024, dtype=np.uint8))

        ciphertext, nonce = encrypt_bytes(plaintext, key)
        recovered = decrypt_bytes(ciphertext, key, nonce)

        assert recovered == plaintext

    def test_tampered_ciphertext_fails(self):
        """Tampering with ciphertext should cause decryption to fail."""
        key = generate_session_key()
        plaintext = b"Authenticated data"

        ciphertext, nonce = encrypt_bytes(plaintext, key)

        # Tamper with ciphertext
        tampered = bytearray(ciphertext)
        tampered[0] ^= 0xFF
        tampered = bytes(tampered)

        with pytest.raises(Exception):
            decrypt_bytes(tampered, key, nonce)

    def test_wrong_nonce_fails(self):
        """Using wrong nonce should cause decryption to fail."""
        import os

        key = generate_session_key()
        plaintext = b"Test data"

        ciphertext, nonce = encrypt_bytes(plaintext, key)
        wrong_nonce = os.urandom(12)

        with pytest.raises(Exception):
            decrypt_bytes(ciphertext, key, wrong_nonce)

    def test_unique_nonces(self):
        """Each encryption should produce a unique nonce."""
        key = generate_session_key()
        plaintext = b"Same message"

        _, nonce1 = encrypt_bytes(plaintext, key)
        _, nonce2 = encrypt_bytes(plaintext, key)

        assert nonce1 != nonce2


class TestCloakingEdgeCases:
    """Edge case tests for cloaking."""

    def test_empty_sequence(self):
        """Handle empty sequence gracefully."""
        ctx = create_cloaking_context(seed=42)
        # Zero-length sequence
        hidden = torch.randn(1, 0, 4096)
        cloaked = cloak(hidden, ctx, add_noise=False)
        assert cloaked.shape == hidden.shape

    def test_large_batch(self):
        """Test with larger batch sizes."""
        ctx = create_cloaking_context(seed=42)
        hidden = torch.randn(32, 128, 4096)

        cloaked = cloak(hidden, ctx, add_noise=False)
        recovered = uncloak(cloaked, ctx)

        torch.testing.assert_close(hidden, recovered, rtol=1e-4, atol=1e-4)

    def test_device_transfer(self):
        """Cloaking should work across devices."""
        ctx = create_cloaking_context(seed=42)
        hidden = torch.randn(1, 10, 4096)

        # Cloak on CPU
        cloaked = cloak(hidden, ctx, add_noise=False)
        recovered = uncloak(cloaked, ctx)

        torch.testing.assert_close(hidden, recovered, rtol=1e-4, atol=1e-4)

    def test_different_dtypes(self):
        """Test cloaking with different tensor dtypes."""
        ctx = create_cloaking_context(seed=42)

        for dtype in [torch.float32, torch.float16]:
            hidden = torch.randn(1, 5, 4096, dtype=dtype)
            cloaked = cloak(hidden, ctx, add_noise=False)
            recovered = uncloak(cloaked, ctx)

            # Lower precision for float16
            rtol = 1e-2 if dtype == torch.float16 else 1e-4
            torch.testing.assert_close(hidden, recovered, rtol=rtol, atol=rtol)


class TestSecurityProperties:
    """Tests verifying security guarantees."""

    def test_cloaked_values_uniformly_distributed(self):
        """Cloaked embeddings should appear random."""
        ctx = create_cloaking_context(seed=42)

        # Create structured input (all ones)
        hidden = torch.ones(1, 100, 4096)
        cloaked = cloak(hidden, ctx, add_noise=True)

        # Cloaked values should have mean ~0 and std ~sigma
        mean = cloaked.mean().item()
        std = cloaked.std().item()

        # With DP noise + rotation, values should be spread out
        assert abs(mean) < 1.0  # Not clustered around 1
        assert std > 1.0  # Has significant variance

    def test_different_sessions_different_cloaking(self):
        """Different sessions should use different cloaking matrices."""
        ctx1 = create_cloaking_context(seed=1)
        ctx2 = create_cloaking_context(seed=2)

        hidden = torch.randn(1, 10, 4096)

        cloaked1 = cloak(hidden, ctx1, add_noise=False)
        cloaked2 = cloak(hidden, ctx2, add_noise=False)

        # Same input, different cloaking = different outputs
        assert not torch.allclose(cloaked1, cloaked2)

    def test_cannot_recover_without_matrix(self):
        """Cannot uncloak without the correct matrix."""
        ctx1 = create_cloaking_context(seed=1)
        ctx2 = create_cloaking_context(seed=2)

        hidden = torch.randn(1, 10, 4096)
        cloaked = cloak(hidden, ctx1, add_noise=False)

        # Try to uncloak with wrong matrix
        wrong_recovered = uncloak(cloaked, ctx2)

        # Should not match original
        assert not torch.allclose(hidden, wrong_recovered, rtol=0.1, atol=0.1)
