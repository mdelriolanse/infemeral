"""NVIB integration tests."""

import importlib
import sys
import time
from unittest import mock

import numpy as np
import pytest
import torch


class TestNVIBGracefulDegradation:
    """Tests for NVIB graceful degradation."""

    def test_client_works_without_nvib(self, mock_client_weights):
        """Client should work when NVIB library is missing."""
        # Save original modules
        original_nvib = sys.modules.get("infemeral.nvib")
        original_client = sys.modules.get("infemeral.client")

        try:
            # Remove NVIB module from cache to simulate it being unavailable
            if "infemeral.nvib" in sys.modules:
                del sys.modules["infemeral.nvib"]
            if "infemeral.nvib.nvib_wrapper" in sys.modules:
                del sys.modules["infemeral.nvib.nvib_wrapper"]

            # Mock the import to fail
            with mock.patch.dict(
                sys.modules,
                {"infemeral.nvib": None, "infemeral.nvib.nvib_wrapper": None},
            ):
                # Force reimport of client module
                if "infemeral.client" in sys.modules:
                    del sys.modules["infemeral.client"]

                from infemeral import client

                importlib.reload(client)

                # Should not raise - client works without NVIB
                c = client.Client(
                    weights_path=mock_client_weights, server_url="localhost:50051"
                )
                assert c.nvib_cloaker is None
                c.close()
        finally:
            # Restore original modules
            if original_nvib is not None:
                sys.modules["infemeral.nvib"] = original_nvib
            if original_client is not None:
                sys.modules["infemeral.client"] = original_client

    def test_client_has_nvib_attribute(self, mock_client_weights):
        """Client should always have nvib_cloaker attribute."""
        from infemeral.client import Client

        c = Client(weights_path=mock_client_weights, server_url="localhost:50051")

        # Should have the attribute (may be None if NVIB not compiled)
        assert hasattr(c, "nvib_cloaker")
        c.close()


class TestNVIBCloaking:
    """Tests for NVIB cloaking behavior."""

    @pytest.fixture
    def nvib_cloaker(self):
        """Get NVIB cloaker if available."""
        try:
            from infemeral.nvib import NVIBCloaker

            return NVIBCloaker(dim=4096, beta=1.0, seed=42)
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"NVIB library not available: {e}")

    def test_cloaking_changes_embedding(self, nvib_cloaker):
        """NVIB should add noise to embedding."""
        embedding = torch.randn(1, 4096, dtype=torch.float32)
        cloaked = nvib_cloaker.cloak(embedding)

        # Should be different
        assert not torch.allclose(embedding, cloaked)

        # Should preserve shape
        assert cloaked.shape == embedding.shape

    def test_cloaking_preserves_shape(self, nvib_cloaker):
        """NVIB should preserve embedding shape."""
        # Test various shapes
        shapes = [
            (1, 4096),
            (4, 4096),
            (1, 10, 4096),
            (2, 5, 4096),
        ]

        for shape in shapes:
            embedding = torch.randn(*shape, dtype=torch.float32)
            cloaked = nvib_cloaker.cloak(embedding)
            assert cloaked.shape == embedding.shape, f"Shape mismatch for {shape}"

    def test_cloaking_deterministic_with_seed(self, nvib_cloaker):
        """Same seed should produce same noise."""
        embedding = torch.randn(1, 4096, dtype=torch.float32)

        nvib_cloaker.set_seed(123)
        cloaked1 = nvib_cloaker.cloak(embedding)

        nvib_cloaker.set_seed(123)
        cloaked2 = nvib_cloaker.cloak(embedding)

        torch.testing.assert_close(cloaked1, cloaked2)

    def test_different_seeds_produce_different_noise(self, nvib_cloaker):
        """Different seeds should produce different noise."""
        embedding = torch.randn(1, 4096, dtype=torch.float32)

        nvib_cloaker.set_seed(123)
        cloaked1 = nvib_cloaker.cloak(embedding)

        nvib_cloaker.set_seed(456)
        cloaked2 = nvib_cloaker.cloak(embedding)

        assert not torch.allclose(cloaked1, cloaked2)

    def test_beta_affects_noise_level(self, nvib_cloaker):
        """Higher beta should produce less noise."""
        from scipy.spatial.distance import cosine

        embedding = torch.randn(1, 4096, dtype=torch.float32)

        # High beta = low noise
        nvib_cloaker.set_seed(42)
        cloaked_high_beta = nvib_cloaker.cloak(embedding, beta=10.0)
        sim_high = 1 - cosine(
            embedding.numpy().flatten(), cloaked_high_beta.numpy().flatten()
        )

        # Low beta = high noise
        nvib_cloaker.set_seed(42)
        cloaked_low_beta = nvib_cloaker.cloak(embedding, beta=0.1)
        sim_low = 1 - cosine(
            embedding.numpy().flatten(), cloaked_low_beta.numpy().flatten()
        )

        # Higher beta should have higher similarity (less noise)
        assert sim_high > sim_low, f"sim_high={sim_high:.4f}, sim_low={sim_low:.4f}"

    def test_cloaking_handles_float16_input(self, nvib_cloaker):
        """NVIB should handle float16 input by converting to float32."""
        embedding = torch.randn(1, 4096, dtype=torch.float16)
        cloaked = nvib_cloaker.cloak(embedding)

        # Output should be float32
        assert cloaked.dtype == torch.float32
        assert cloaked.shape == embedding.shape

    def test_cloaking_performance(self, nvib_cloaker):
        """NVIB cloaking should be fast (<2ms for 4096-dim)."""
        embedding = torch.randn(1, 4096, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = nvib_cloaker.cloak(embedding)

        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = nvib_cloaker.cloak(embedding)
            times.append((time.perf_counter() - start) * 1000)

        p95 = np.percentile(times, 95)
        assert p95 < 2.0, f"P95 latency {p95:.3f}ms exceeds 2ms target"


class TestTokenTimingWithNVIB:
    """Tests for timing metrics including NVIB."""

    def test_token_timing_has_nvib_field(self):
        """TokenTiming should have nvib_ms field."""
        from infemeral.client import TokenTiming

        timing = TokenTiming()
        assert hasattr(timing, "nvib_ms")
        assert timing.nvib_ms == 0.0

    def test_token_timing_nvib_populated(self):
        """TokenTiming.nvib_ms should be populated when NVIB is active."""
        from infemeral.client import TokenTiming

        timing = TokenTiming(
            embed_ms=1.0,
            nvib_ms=0.5,
            network_ms=10.0,
            de_embed_ms=0.8,
            sample_ms=0.2,
            total_ms=12.5,
        )

        assert timing.nvib_ms == 0.5
        assert timing.total_ms == 12.5


class TestNVIBSettings:
    """Tests for NVIB configuration."""

    def test_nvib_settings_defaults(self):
        """NVIB settings should have correct defaults."""
        from infemeral.config import nvib_settings

        assert nvib_settings.beta == 100.0  # Changed from 1.0 for coherent output
        assert nvib_settings.dim == 0  # Auto-detect from model
        assert nvib_settings.mu_init == 0.0
        assert nvib_settings.log_sigma2_init == 0.0
        assert nvib_settings.simd_level == "auto"

    def test_nvib_settings_from_env(self, monkeypatch):
        """NVIB settings should be configurable via environment."""
        monkeypatch.setenv("INFEMERAL_NVIB_BETA", "2.5")
        monkeypatch.setenv("INFEMERAL_NVIB_DIM", "2048")

        # Force reload of settings
        from infemeral.config import NVIBSettings

        settings = NVIBSettings()

        assert settings.beta == 2.5
        assert settings.dim == 2048
