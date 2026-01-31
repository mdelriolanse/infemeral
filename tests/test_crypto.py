"""Tests for cryptographic primitives."""

import numpy as np
import pytest

from infemeral.crypto import (
    decrypt_bytes,
    encrypt_bytes,
    generate_session_key,
)


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
