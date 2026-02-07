"""Test NVIB compatibility across different model dimensions.

This test module verifies that NVIB cloaking works correctly with
various model architectures and hidden dimensions.
"""

import pytest
import torch


class TestNVIBMultiModel:
    """Tests for NVIB across model architectures."""

    @pytest.fixture(params=[4096, 5120, 8192])
    def dim(self, request):
        """Parametrized fixture for common model dimensions."""
        return request.param

    def test_nvib_various_dimensions(self, dim):
        """NVIB should work with common model dimensions."""
        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=dim, beta=100.0, seed=42)
        embedding = torch.randn(1, dim, dtype=torch.float32)

        cloaked = cloaker.cloak(embedding)

        assert cloaked.shape == (1, dim)
        assert not torch.allclose(cloaked, embedding)

    def test_nvib_batch_processing(self):
        """NVIB should handle batch embeddings correctly."""
        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=5120, beta=100.0, seed=42)

        # Batch of 8 embeddings
        batch = torch.randn(8, 5120, dtype=torch.float32)
        cloaked = cloaker.cloak(batch)

        assert cloaked.shape == (8, 5120)

    def test_nvib_sequence_processing(self):
        """NVIB should handle sequence embeddings (batch, seq_len, dim)."""
        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=5120, beta=100.0, seed=42)

        # Sequence of 10 tokens, hidden dim 5120
        sequence = torch.randn(1, 10, 5120, dtype=torch.float32)

        # Flatten for NVIB processing
        flat = sequence.view(-1, 5120)
        cloaked_flat = cloaker.cloak(flat)

        # Reshape back
        cloaked = cloaked_flat.view(1, 10, 5120)

        assert cloaked.shape == sequence.shape

    @pytest.mark.parametrize(
        "beta,expected_noise",
        [
            (1.0, "high"),
            (100.0, "low"),
            (1000.0, "minimal"),
        ],
    )
    def test_nvib_beta_affects_noise(self, beta, expected_noise):
        """Beta parameter should control noise level (higher beta = less noise)."""
        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=4096, beta=beta, seed=42)
        embedding = torch.randn(1, 4096, dtype=torch.float32)

        cloaked = cloaker.cloak(embedding)

        # Calculate MSE between original and cloaked
        mse = ((cloaked - embedding) ** 2).mean().item()

        # Higher beta should result in lower MSE (less distortion)
        if expected_noise == "high":
            assert mse > 0.1
        elif expected_noise == "low":
            assert mse < 1.0
        else:  # minimal
            assert mse < 0.1


class TestNVIBDimensionValidation:
    """Tests for NVIB dimension validation and error handling."""

    def test_nvib_rejects_mismatched_dimension(self):
        """NVIB should handle dimension mismatch gracefully."""
        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=4096, beta=100.0, seed=42)

        # Try to cloak 5120-dim embedding with 4096-dim cloaker
        wrong_dim = torch.randn(1, 5120, dtype=torch.float32)

        # This should either raise an error or handle gracefully
        with pytest.raises((RuntimeError, ValueError)):
            cloaker.cloak(wrong_dim)

    def test_nvib_accepts_matching_dimension(self):
        """NVIB should accept correctly dimensioned embeddings."""
        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=5120, beta=100.0, seed=42)

        correct_dim = torch.randn(1, 5120, dtype=torch.float32)
        cloaked = cloaker.cloak(correct_dim)

        assert cloaked.shape == correct_dim.shape


class TestNVIBPerformance:
    """Performance tests for NVIB at various dimensions."""

    @pytest.mark.slow
    @pytest.mark.parametrize("dim", [4096, 5120, 8192])
    def test_nvib_latency_under_threshold(self, dim):
        """NVIB cloaking should complete in under 5ms for single embedding."""
        import time

        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=dim, beta=100.0, seed=42)
        embedding = torch.randn(1, dim, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            cloaker.cloak(embedding)

        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            cloaker.cloak(embedding)
            times.append((time.perf_counter() - start) * 1000)

        median_ms = sorted(times)[len(times) // 2]

        # Should be under 5ms for any dimension
        assert median_ms < 5.0, f"NVIB latency {median_ms:.2f}ms exceeds 5ms threshold"
