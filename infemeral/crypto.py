"""Cryptographic primitives for zero-trust inference.

This module implements:
1. Orthogonal matrix cloaking (preserves attention dot products)
2. Differential privacy via Gaussian mechanism
3. AES-256-GCM envelope encryption for KV cache
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from infemeral.config import crypto_settings


@dataclass
class CloakingContext:
    """Holds the cloaking matrix and DP parameters for a session."""

    matrix: torch.Tensor  # Orthogonal matrix M (d x d)
    matrix_t: torch.Tensor  # Transposed M for uncloaking
    sigma: float  # DP noise standard deviation
    device: str = "cpu"  # Device where matrix is stored
    seed: int | None = None  # Seed for lazy regeneration


def generate_orthogonal_matrix(dim: int, seed: int | None = None) -> np.ndarray:
    """Generate a random orthogonal matrix using Haar measure.

    The matrix M satisfies M @ M.T = I, which means:
    - (Mx).T @ (My) = x.T @ y (dot products preserved)
    - This is critical for attention: Q'K'.T = QK.T

    Uses QR decomposition of a random Gaussian matrix, which produces
    Haar-distributed orthogonal matrices and is much faster than
    scipy.stats.ortho_group for large dimensions.

    Args:
        dim: Matrix dimension (should match model hidden_dim)
        seed: Random seed for reproducibility (use unique per session)

    Returns:
        Orthogonal matrix of shape (dim, dim) as float32
    """
    rng = np.random.default_rng(seed)
    # Generate random Gaussian matrix
    random_matrix = rng.standard_normal((dim, dim)).astype(np.float32)
    # QR decomposition gives orthogonal Q
    q, r = np.linalg.qr(random_matrix)
    # Ensure uniform distribution by adjusting signs based on diagonal of R
    d = np.diag(r)
    ph = np.sign(d)
    q = q * ph
    return q


def compute_dp_sigma(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    """Compute Gaussian mechanism noise scale for (epsilon, delta)-DP.

    Formula: sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Args:
        epsilon: Privacy budget (lower = more private)
        delta: Failure probability
        sensitivity: L2 sensitivity of the function (default 1.0 for normalized embeddings)

    Returns:
        Standard deviation for Gaussian noise
    """
    return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon


def create_cloaking_context(
    seed: int | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> CloakingContext:
    """Create a new cloaking context for a session.

    Args:
        seed: Random seed for orthogonal matrix (should be unique per session)
        device: Target device for matrix storage (cpu or cuda)
        dtype: Storage dtype (float16 saves 50% memory, float32 for max precision)

    Returns:
        CloakingContext with matrix and DP parameters on the specified device
    """
    dim = crypto_settings.hidden_dim

    # Generate in float32 for numerical stability during QR decomposition
    matrix_np = generate_orthogonal_matrix(dim, seed)
    matrix = torch.from_numpy(matrix_np)

    # Convert to target dtype and device
    matrix = matrix.to(device=device, dtype=dtype)
    matrix_t = matrix.T.contiguous()

    sigma = compute_dp_sigma(crypto_settings.dp_epsilon, crypto_settings.dp_delta)

    return CloakingContext(
        matrix=matrix,
        matrix_t=matrix_t,
        sigma=sigma,
        device=device,
        seed=seed,
    )


def cloak(
    hidden: torch.Tensor,
    ctx: CloakingContext,
    add_noise: bool = True,
) -> torch.Tensor:
    """Apply cloaking to hidden states: DP noise + orthogonal rotation.

    Args:
        hidden: Input tensor of shape [batch, seq_len, hidden_dim]
        ctx: Cloaking context with matrix and noise parameters
        add_noise: Whether to add DP noise (disable for testing)

    Returns:
        Cloaked tensor of same shape and dtype
    """
    device = hidden.device
    compute_dtype = hidden.dtype

    # Avoid .to() call if matrix is already on correct device and dtype
    matrix = ctx.matrix
    needs_transfer = matrix.device != device or matrix.dtype != compute_dtype
    if needs_transfer:
        matrix = matrix.to(device=device, dtype=compute_dtype)

    if add_noise:
        noise = torch.randn_like(hidden) * ctx.sigma
        hidden = hidden + noise

    # Apply orthogonal rotation: x' = x @ M.T (equivalent to M @ x for each vector)
    # Using einsum for clarity: 'ij,...j->...i' means M @ x for last dimension
    return torch.einsum("ij,...j->...i", matrix, hidden)


def uncloak(cloaked: torch.Tensor, ctx: CloakingContext) -> torch.Tensor:
    """Remove orthogonal rotation from cloaked hidden states.

    Note: DP noise cannot be removed (by design).

    Args:
        cloaked: Cloaked tensor of shape [batch, seq_len, hidden_dim]
        ctx: Cloaking context with inverse matrix

    Returns:
        Uncloaked tensor of same shape and dtype
    """
    device = cloaked.device
    compute_dtype = cloaked.dtype

    # Avoid .to() call if matrix is already on correct device and dtype
    matrix_t = ctx.matrix_t
    needs_transfer = matrix_t.device != device or matrix_t.dtype != compute_dtype
    if needs_transfer:
        matrix_t = matrix_t.to(device=device, dtype=compute_dtype)

    # Apply inverse rotation: x = x' @ M (since M.T @ M = I)
    return torch.einsum("ij,...j->...i", matrix_t, cloaked)


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
