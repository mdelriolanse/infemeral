"""DeepSeek-R1-32B integration tests.

These tests verify the DeepSeek model loading, NVIB compatibility,
and end-to-end inference with the 5120-dimension hidden state.
"""

import os

import pytest
import torch

# Path constants
DEEPSEEK_WEIGHTS = "/workspace/weights/deepseek-r1-32b"
DEEPSEEK_CLIENT_WEIGHTS = "/workspace/weights/deepseek-r1-32b-client/client_weights.safetensors"

# Marker for tests requiring the downloaded model
requires_deepseek_model = pytest.mark.skipif(
    not os.path.exists(DEEPSEEK_WEIGHTS),
    reason="DeepSeek model not downloaded",
)


@pytest.mark.integration
@requires_deepseek_model
class TestDeepSeekModelLoading:
    """Tests for DeepSeek model loading."""

    def test_server_loads_gptq_model(self):
        """Server should load GPTQ model without errors."""
        os.environ["INFEMERAL_SERVER_WEIGHTS_DIR"] = DEEPSEEK_WEIGHTS
        os.environ["INFEMERAL_SERVER_MODEL_ID"] = (
            "dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4"
        )

        from infemeral.server import load_model

        model = load_model()

        assert model is not None
        assert hasattr(model, "model")
        assert hasattr(model.model, "layers")

    @pytest.mark.skipif(
        not os.path.exists(DEEPSEEK_CLIENT_WEIGHTS),
        reason="DeepSeek client weights not extracted",
    )
    def test_client_loads_deepseek_embeddings(self):
        """Client should load DeepSeek embedding weights."""
        from infemeral.client import EmbeddingLayer

        embedding = EmbeddingLayer(DEEPSEEK_CLIENT_WEIGHTS, device="cuda")

        # Verify hidden dimension is 5120 for DeepSeek/Qwen
        assert embedding.embed_tokens.weight.shape[1] == 5120


@pytest.mark.integration
@requires_deepseek_model
class TestDeepSeekWithNVIB:
    """Tests for DeepSeek + NVIB integration."""

    @pytest.mark.skipif(
        not os.path.exists(DEEPSEEK_CLIENT_WEIGHTS),
        reason="DeepSeek client weights not extracted",
    )
    def test_nvib_auto_dimension_detection(self):
        """NVIB should auto-detect 5120 dimension for DeepSeek."""
        os.environ["INFEMERAL_NVIB_DIM"] = "0"  # Auto-detect

        from infemeral.client import Client

        client = Client(
            weights_path=DEEPSEEK_CLIENT_WEIGHTS,
            server_url="localhost:50051",
        )

        if client.nvib_cloaker is not None:
            assert client.nvib_cloaker.dim == 5120

        client.close()

    def test_nvib_cloaking_5120_dim(self):
        """NVIB should work with 5120-dimension embeddings."""
        try:
            from infemeral.nvib import NVIBCloaker
        except (ImportError, RuntimeError):
            pytest.skip("NVIB not available")

        cloaker = NVIBCloaker(dim=5120, beta=100.0, seed=42)

        embedding = torch.randn(1, 5120, dtype=torch.float32)
        cloaked = cloaker.cloak(embedding)

        assert cloaked.shape == embedding.shape
        assert not torch.allclose(cloaked, embedding)

    @pytest.mark.skipif(
        not os.path.exists(DEEPSEEK_CLIENT_WEIGHTS),
        reason="DeepSeek client weights not extracted",
    )
    def test_end_to_end_inference_deepseek(self):
        """Full inference with DeepSeek + NVIB should produce output."""
        pytest.importorskip("infemeral.nvib")

        os.environ["INFEMERAL_SERVER_WEIGHTS_DIR"] = DEEPSEEK_WEIGHTS
        os.environ["INFEMERAL_NVIB_DIM"] = "0"

        from infemeral.client import Client

        client = Client(
            weights_path=DEEPSEEK_CLIENT_WEIGHTS,
            server_url="localhost:50051",
        )

        # Skip if no server running
        if not client.check_channel_health():
            client.close()
            pytest.skip("Server not running")

        result = client.generate("<think>\nWhat is 2+2?", max_new_tokens=20)

        assert result is not None
        assert len(result) > 0

        client.close()


@pytest.mark.integration
class TestDeepSeekPresets:
    """Tests for DeepSeek model presets configuration."""

    def test_model_presets_contains_deepseek(self):
        """MODEL_PRESETS should contain deepseek-r1-32b-gptq."""
        from infemeral.config import MODEL_PRESETS

        assert "deepseek-r1-32b-gptq" in MODEL_PRESETS
        preset = MODEL_PRESETS["deepseek-r1-32b-gptq"]

        assert preset["model_id"] == "dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4"
        assert preset["hidden_dim"] == 5120
        assert preset["num_layers"] == 64
        assert preset["architecture"] == "qwen"

    def test_server_settings_preset_property(self):
        """ServerSettings should have model_config_preset property."""
        os.environ["INFEMERAL_SERVER_MODEL_PRESET"] = "deepseek-r1-32b-gptq"

        from importlib import reload

        import infemeral.config

        reload(infemeral.config)

        from infemeral.config import server_settings

        preset = server_settings.model_config_preset
        assert preset["hidden_dim"] == 5120
