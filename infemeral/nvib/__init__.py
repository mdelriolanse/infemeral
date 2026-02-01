"""NVIB (Nonparametric Variational Information Bottleneck) Cloaking Library.

This package provides high-performance C-based NVIB cloaking for privacy-preserving
embedding transformations.
"""

from infemeral.nvib.nvib_wrapper import NVIBCloaker, nvib_cloak

__all__ = ["NVIBCloaker", "nvib_cloak"]
