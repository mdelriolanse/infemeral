"""Python ctypes wrapper for NVIB cloaking C library."""

import ctypes
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# Determine library extension based on platform
if platform.system() == "Windows":
    LIB_EXT = ".dll"
elif platform.system() == "Darwin":
    LIB_EXT = ".dylib"
else:
    LIB_EXT = ".so"

# Try to find the compiled library
LIB_NAME = f"nvib_cloak{LIB_EXT}"
LIB_PATH = Path(__file__).parent / LIB_NAME

# Fallback: look in current directory or common library paths
if not LIB_PATH.exists():
    LIB_PATH = Path.cwd() / LIB_NAME
if not LIB_PATH.exists():
    # Try system library paths
    for lib_dir in ["/usr/local/lib", "/usr/lib"]:
        candidate = Path(lib_dir) / LIB_NAME
        if candidate.exists():
            LIB_PATH = candidate
            break


class NVIBCloaker:
    """Python wrapper for NVIB cloaking C library.

    Provides a high-level interface to apply NVIB (Nonparametric Variational
    Information Bottleneck) noise to embedding vectors for privacy preservation.

    Example:
        >>> cloaker = NVIBCloaker(dim=4096, beta=1.0)
        >>> embedding = torch.randn(1, 4096)
        >>> noised = cloaker.cloak(embedding)
    """

    def __init__(
        self,
        dim: int = 4096,
        beta: float = 1.0,
        mu_init: float = 0.0,
        log_sigma2_init: float = 0.0,
        seed: Optional[int] = None,
    ):
        """Initialize NVIB cloaker.

        Args:
            dim: Embedding dimension (typically 4096 for Llama)
            beta: Privacy budget parameter (higher = less noise, better utility)
            mu_init: Initial mean value for noise distribution
            log_sigma2_init: Initial log variance for noise distribution
            seed: PRNG seed (None = use system entropy)
        """
        self.dim = dim
        self.beta = beta

        # Load C library
        if not LIB_PATH.exists():
            raise RuntimeError(
                f"NVIB library not found at {LIB_PATH}. "
                f"Please compile it first with: make nvib_cloak.so"
            )

        self._lib = ctypes.CDLL(str(LIB_PATH))

        # Define function signatures
        self._setup_ctypes()

        # Initialize C context
        seed_val = seed if seed is not None else 0
        self._ctx = self._lib.nvib_cloak_init(
            ctypes.c_int(dim),
            ctypes.c_float(beta),
            ctypes.c_float(mu_init),
            ctypes.c_float(log_sigma2_init),
            ctypes.c_uint64(seed_val),
        )

        if not self._ctx:
            raise RuntimeError("Failed to initialize NVIB context")

    def _setup_ctypes(self):
        """Setup ctypes function signatures."""
        # nvib_cloak_init
        self._lib.nvib_cloak_init.argtypes = [
            ctypes.c_int,      # dim
            ctypes.c_float,    # beta
            ctypes.c_float,    # mu_init
            ctypes.c_float,    # log_sigma2_init
            ctypes.c_uint64,   # seed
        ]
        self._lib.nvib_cloak_init.restype = ctypes.c_void_p

        # nvib_cloak_forward
        self._lib.nvib_cloak_forward.argtypes = [
            ctypes.c_void_p,   # ctx
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        self._lib.nvib_cloak_forward.restype = ctypes.c_int

        # nvib_cloak_set_seed
        self._lib.nvib_cloak_set_seed.argtypes = [
            ctypes.c_void_p,   # ctx
            ctypes.c_uint64,   # seed
        ]
        self._lib.nvib_cloak_set_seed.restype = None

        # nvib_cloak_set_beta
        self._lib.nvib_cloak_set_beta.argtypes = [
            ctypes.c_void_p,   # ctx
            ctypes.c_float,    # beta
        ]
        self._lib.nvib_cloak_set_beta.restype = None

        # nvib_cloak_free
        self._lib.nvib_cloak_free.argtypes = [ctypes.c_void_p]
        self._lib.nvib_cloak_free.restype = None

    def cloak(self, embedding: torch.Tensor, beta: Optional[float] = None) -> torch.Tensor:
        """Apply NVIB cloaking to embedding tensor.

        Args:
            embedding: Input embedding tensor of shape (..., dim)
            beta: Optional privacy budget override (uses instance default if None)

        Returns:
            Noised embedding tensor with same shape as input
        """
        # Use provided beta or instance default
        old_beta = None
        if beta is not None and beta != self.beta:
            # Update beta in context temporarily
            old_beta = self.beta
            self._lib.nvib_cloak_set_beta(self._ctx, ctypes.c_float(beta))
            self.beta = beta

        # Ensure contiguous float32 tensor
        if embedding.dtype != torch.float32:
            embedding = embedding.float()
        embedding = embedding.contiguous()

        # Get shape and flatten for processing
        original_shape = embedding.shape
        flat_embedding = embedding.view(-1, self.dim)
        batch_size = flat_embedding.shape[0]

        # Prepare output tensor
        output = torch.empty_like(flat_embedding)

        # Process each embedding in batch
        for i in range(batch_size):
            input_ptr = flat_embedding[i].numpy().ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            )
            output_ptr = output[i].numpy().ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            )

            result = self._lib.nvib_cloak_forward(
                self._ctx,
                input_ptr,
                output_ptr,
            )

            if result != 0:
                raise RuntimeError(f"NVIB cloaking failed with error code {result}")

        # Restore beta if it was temporarily changed
        if old_beta is not None:
            self._lib.nvib_cloak_set_beta(self._ctx, ctypes.c_float(old_beta))
            self.beta = old_beta

        # Reshape to original shape
        return output.view(original_shape)

    def set_seed(self, seed: int):
        """Set PRNG seed for deterministic behavior.

        Args:
            seed: PRNG seed value
        """
        self._lib.nvib_cloak_set_seed(self._ctx, ctypes.c_uint64(seed))

    def __del__(self):
        """Cleanup C context on deletion."""
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.nvib_cloak_free(self._ctx)
            self._ctx = None


# Convenience function for quick usage
def nvib_cloak(
    embedding: torch.Tensor,
    beta: float = 1.0,
    dim: int = 4096,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Apply NVIB cloaking to embedding (convenience function).

    Args:
        embedding: Input embedding tensor
        beta: Privacy budget parameter
        dim: Embedding dimension
        seed: PRNG seed (None = random)

    Returns:
        Noised embedding tensor
    """
    cloaker = NVIBCloaker(dim=dim, beta=beta, seed=seed)
    try:
        return cloaker.cloak(embedding)
    finally:
        del cloaker
