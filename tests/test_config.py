"""Tests for configuration settings."""

import os
from unittest import mock

import pytest

from infemeral.config import (
    ClientSettings,
    CryptoSettings,
    ServerSettings,
)


class TestCryptoSettings:
    """Tests for cryptographic settings."""

    def test_default_values(self):
        """Check default cryptographic parameters."""
        settings = CryptoSettings()

        assert settings.hidden_dim == 4096
        assert settings.dp_epsilon == 2.0
        assert settings.dp_delta == 1e-5

    def test_env_override(self):
        """Environment variables should override defaults."""
        with mock.patch.dict(
            os.environ,
            {
                "INFEMERAL_CRYPTO_HIDDEN_DIM": "3072",
                "INFEMERAL_CRYPTO_DP_EPSILON": "1.0",
                "INFEMERAL_CRYPTO_DP_DELTA": "1e-6",
            },
        ):
            settings = CryptoSettings()

            assert settings.hidden_dim == 3072
            assert settings.dp_epsilon == 1.0
            assert settings.dp_delta == 1e-6

    def test_dp_parameters_valid_range(self):
        """DP parameters should be positive."""
        settings = CryptoSettings()

        assert settings.dp_epsilon > 0
        assert settings.dp_delta > 0
        assert settings.dp_delta < 1  # delta should be small


class TestClientSettings:
    """Tests for client settings."""

    def test_default_values(self):
        """Check default client parameters."""
        settings = ClientSettings()

        assert settings.weights_path == "/workspace/weights/client_weights.safetensors"
        assert settings.server_url == "localhost:50051"
        assert settings.model_id == "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

    def test_env_override(self):
        """Environment variables should override defaults."""
        with mock.patch.dict(
            os.environ,
            {
                "INFEMERAL_CLIENT_WEIGHTS_PATH": "/custom/path/weights.safetensors",
                "INFEMERAL_CLIENT_SERVER_URL": "remote-server:8080",
                "INFEMERAL_CLIENT_MODEL_ID": "custom/model",
            },
        ):
            settings = ClientSettings()

            assert settings.weights_path == "/custom/path/weights.safetensors"
            assert settings.server_url == "remote-server:8080"
            assert settings.model_id == "custom/model"


class TestServerSettings:
    """Tests for server settings."""

    def test_default_values(self):
        """Check default server parameters."""
        settings = ServerSettings()

        assert settings.weights_dir == "/workspace/weights/model"
        assert settings.tensorized_weights_path == "/workspace/weights/model.tensors"
        assert settings.kv_cache_dir == "/workspace/weights/kv"
        assert settings.max_context_length == 2048
        assert settings.attention_sink_tokens == 4
        assert settings.grpc_port == 50051

    def test_env_override(self):
        """Environment variables should override defaults."""
        with mock.patch.dict(
            os.environ,
            {
                "INFEMERAL_SERVER_WEIGHTS_DIR": "/data/model",
                "INFEMERAL_SERVER_TENSORIZED_WEIGHTS_PATH": "/data/model.tensors",
                "INFEMERAL_SERVER_KV_CACHE_DIR": "/data/cache",
                "INFEMERAL_SERVER_MAX_CONTEXT_LENGTH": "4096",
                "INFEMERAL_SERVER_ATTENTION_SINK_TOKENS": "8",
                "INFEMERAL_SERVER_GRPC_PORT": "9090",
            },
        ):
            settings = ServerSettings()

            assert settings.weights_dir == "/data/model"
            assert settings.tensorized_weights_path == "/data/model.tensors"
            assert settings.kv_cache_dir == "/data/cache"
            assert settings.max_context_length == 4096
            assert settings.attention_sink_tokens == 8
            assert settings.grpc_port == 9090

    def test_context_length_positive(self):
        """Context length should be positive."""
        settings = ServerSettings()
        assert settings.max_context_length > 0

    def test_attention_sinks_reasonable(self):
        """Attention sink count should be reasonable."""
        settings = ServerSettings()
        # Typically 2-8 attention sinks
        assert 1 <= settings.attention_sink_tokens <= 16


class TestSettingsInteraction:
    """Tests for settings interactions between modules."""

    def test_hidden_dim_consistency(self):
        """Hidden dim should match model architecture."""
        crypto = CryptoSettings()
        client = ClientSettings()

        # Llama-3.1-8B has hidden_dim=4096
        if "Llama-3.1-8B" in client.model_id:
            assert crypto.hidden_dim == 4096

    def test_singleton_instances_exist(self):
        """Singleton instances should be importable."""
        from infemeral.config import (
            client_settings,
            crypto_settings,
            server_settings,
        )

        assert crypto_settings is not None
        assert client_settings is not None
        assert server_settings is not None
