"""Cryptographic primitives for zero-trust inference.

This module implements AES-256-GCM envelope encryption for:
1. Tensor data in transit between client and server
2. KV cache storage at rest
"""

import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def generate_session_key() -> bytes:
    """Generate a random 256-bit AES key for session encryption."""
    return os.urandom(32)


def encrypt_bytes(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    """Encrypt data using AES-256-GCM.

    Args:
        plaintext: Data to encrypt
        key: 256-bit AES key

    Returns:
        Tuple of (ciphertext, nonce)
    """
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return ciphertext, nonce


def decrypt_bytes(ciphertext: bytes, key: bytes, nonce: bytes) -> bytes:
    """Decrypt data using AES-256-GCM.

    Args:
        ciphertext: Encrypted data
        key: 256-bit AES key
        nonce: 12-byte nonce used during encryption

    Returns:
        Decrypted plaintext
    """
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)
