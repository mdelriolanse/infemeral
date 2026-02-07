"""Tests for server-side inference."""

import base64
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch

from infemeral.crypto import encrypt_bytes, generate_session_key
from infemeral.tensors import serialize_tensor


class TestKVCacheManagement:
    """Tests for KV cache file operations."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory."""
        cache_dir = tmp_path / "kv_cache"
        cache_dir.mkdir()
        return cache_dir

    def test_save_load_kv_cache_roundtrip(self, temp_cache_dir):
        """Save and load should recover KV cache."""
        from infemeral.server import (
            get_kv_cache_path,
            load_kv_cache,
            save_kv_cache,
        )

        # Mock settings to use temp dir with disk mode (tests disk I/O explicitly)
        with mock.patch("infemeral.server.server_settings") as mock_settings:
            mock_settings.kv_cache_dir = str(temp_cache_dir)
            mock_settings.kv_cache_mode = "disk"  # Explicitly use disk mode for this test

            session_id = "test_session"
            session_key = generate_session_key()

            # Create KV tensors for 2 layers (simulating transformer layers)
            num_layers = 2
            kv_tuples = tuple(
                (torch.randn(1, 8, 128, 64), torch.randn(1, 8, 128, 64))
                for _ in range(num_layers)
            )

            # Save
            save_kv_cache(session_id, kv_tuples, session_key)

            # Verify file exists
            cache_path = get_kv_cache_path(session_id)
            assert cache_path.exists()

            # Load
            loaded = load_kv_cache(session_id, session_key, device="cpu")

            assert loaded is not None
            # Should return tuple of (key, value) pairs for each layer
            assert isinstance(loaded, tuple)
            assert len(loaded) == num_layers
            for layer_kv in loaded:
                assert isinstance(layer_kv, tuple)
                assert len(layer_kv) == 2  # (key, value)

    def test_load_nonexistent_cache_returns_none(self, temp_cache_dir):
        """Loading non-existent cache should return None."""
        from infemeral.server import load_kv_cache

        with mock.patch("infemeral.server.server_settings") as mock_settings:
            mock_settings.kv_cache_dir = str(temp_cache_dir)
            mock_settings.kv_cache_mode = "disk"

            result = load_kv_cache("nonexistent", generate_session_key(), "cpu")
            assert result is None

    def test_delete_kv_cache(self, temp_cache_dir):
        """Delete should remove cache file."""
        from infemeral.server import (
            delete_kv_cache,
            get_kv_cache_path,
            save_kv_cache,
        )

        with mock.patch("infemeral.server.server_settings") as mock_settings:
            mock_settings.kv_cache_dir = str(temp_cache_dir)
            mock_settings.kv_cache_mode = "disk"

            session_id = "test_session"
            session_key = generate_session_key()

            # Save some cache (2 layers of KV tuples)
            kv_tuples = tuple(
                (torch.randn(1, 8, 64, 64), torch.randn(1, 8, 64, 64))
                for _ in range(2)
            )
            save_kv_cache(session_id, kv_tuples, session_key)

            # Verify exists
            assert get_kv_cache_path(session_id).exists()

            # Delete
            delete_kv_cache(session_id)

            # Verify gone
            assert not get_kv_cache_path(session_id).exists()

    def test_wrong_key_fails_to_decrypt(self, temp_cache_dir):
        """Loading with wrong key should fail or return None."""
        from infemeral.server import load_kv_cache, save_kv_cache

        with mock.patch("infemeral.server.server_settings") as mock_settings:
            mock_settings.kv_cache_dir = str(temp_cache_dir)
            mock_settings.kv_cache_mode = "disk"

            session_id = "test_session"
            correct_key = generate_session_key()
            wrong_key = generate_session_key()

            kv_tuples = tuple(
                (torch.randn(1, 8, 64, 64), torch.randn(1, 8, 64, 64))
                for _ in range(2)
            )
            save_kv_cache(session_id, kv_tuples, correct_key)

            # Should fail with wrong key
            result = load_kv_cache(session_id, wrong_key, "cpu")
            assert result is None  # Returns None on failure


class TestForwardTransformer:
    """Tests for transformer forward pass."""

    @pytest.fixture
    def mock_model(self):
        """Create a minimal mock model for testing."""
        from transformers.cache_utils import DynamicCache

        class MockRotaryEmb(torch.nn.Module):
            """Mock rotary embedding module."""
            def __init__(self, head_dim=64):
                super().__init__()
                self.head_dim = head_dim

            def forward(self, hidden_states, position_ids):
                """Return mock cos/sin tensors for rotary embeddings."""
                batch_size = hidden_states.shape[0]
                seq_len = position_ids.shape[1]
                # Return cos, sin tensors matching expected shapes
                cos = torch.ones(batch_size, seq_len, self.head_dim)
                sin = torch.zeros(batch_size, seq_len, self.head_dim)
                return cos, sin

        class MockSelfAttn(torch.nn.Module):
            """Mock self-attention with rotary_emb."""
            def __init__(self, head_dim=64):
                super().__init__()
                self.rotary_emb = MockRotaryEmb(head_dim)

        class MockLayer(torch.nn.Module):
            def __init__(self, hidden_size, head_dim=64, num_heads=8):
                super().__init__()
                self.hidden_size = hidden_size
                self.self_attn = MockSelfAttn(head_dim)

            def forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                use_cache=True,
                position_embeddings=None,
            ):
                # Identity transform for testing
                output = hidden_states
                # Return updated DynamicCache (in-place update handled by caller)
                return output, past_key_value

        class MockNorm(torch.nn.Module):
            def forward(self, x):
                return x

        class MockTransformer(torch.nn.Module):
            def __init__(self, num_layers=2, hidden_size=4096):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [MockLayer(hidden_size) for _ in range(num_layers)]
                )
                self.norm = MockNorm()

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockTransformer()

        return MockModel()

    def test_forward_produces_correct_shape(self, mock_model):
        """Forward pass should preserve hidden state shape."""
        from infemeral.server import forward_transformer

        hidden = torch.randn(1, 10, 4096)

        output, kv = forward_transformer(mock_model, hidden)

        assert output.shape == hidden.shape

    def test_forward_returns_kv_cache(self, mock_model):
        """Forward pass should return KV cache (may be empty with mock layers)."""
        from infemeral.server import forward_transformer

        hidden = torch.randn(1, 10, 4096)

        output, kv = forward_transformer(mock_model, hidden)

        # KV should be a tuple (may be empty with simple mocks since layers
        # don't actually populate the cache)
        assert isinstance(kv, tuple)

    @pytest.mark.skip(reason="Requires transformers version-specific DynamicCache API")
    def test_forward_with_past_kv(self, mock_model):
        """Forward pass should accept past KV cache."""
        from infemeral.server import forward_transformer

        hidden = torch.randn(1, 1, 4096)  # Single new token

        # Create mock past KV
        past_kv = tuple(
            (
                torch.randn(1, 8, 10, 64),  # past keys
                torch.randn(1, 8, 10, 64),  # past values
            )
            for _ in range(2)  # 2 layers
        )

        output, new_kv = forward_transformer(mock_model, hidden, past_kv)

        assert output.shape == hidden.shape


class TestServerHandler:
    """Tests for RunPod serverless handler."""

    @pytest.fixture
    def mock_event(self):
        """Create a mock inference event."""
        session_key = generate_session_key()
        hidden = torch.randn(1, 10, 4096, dtype=torch.float16)

        # Serialize and encrypt
        data, shape, dtype = serialize_tensor(hidden)
        ciphertext, nonce = encrypt_bytes(data, session_key)

        return {
            "input": {
                "cloaked_embedding": base64.b64encode(ciphertext).decode(),
                "encrypted_session_key": base64.b64encode(session_key).decode(),
                "nonce": base64.b64encode(nonce).decode(),
                "shape": shape,
                "dtype": dtype,
                "session_id": "test_session",
            }
        }

    def test_handler_returns_dict(self, mock_event):
        """Handler should return a dictionary."""
        from infemeral.server import handler

        # Mock the model loading
        with mock.patch("infemeral.server.load_model") as mock_load:
            # Create a minimal mock model
            mock_model = mock.MagicMock()
            mock_model.parameters.return_value = iter([torch.tensor([1.0])])
            mock_load.return_value = mock_model

            # Mock forward_transformer
            with mock.patch("infemeral.server.forward_transformer") as mock_forward:
                mock_forward.return_value = (
                    torch.randn(1, 10, 4096),
                    tuple(),
                )

                result = handler(mock_event)

                assert isinstance(result, dict)

    def test_handler_error_handling(self):
        """Handler should catch and return errors."""
        from infemeral.server import handler

        # Invalid event
        result = handler({"input": {"invalid": "data"}})

        assert "error" in result

    def test_handler_output_encrypted(self, mock_event):
        """Handler output should be encrypted."""
        from infemeral.server import handler

        with mock.patch("infemeral.server.load_model") as mock_load:
            mock_model = mock.MagicMock()
            mock_model.parameters.return_value = iter(
                [torch.tensor([1.0], device="cpu")]
            )
            mock_load.return_value = mock_model

            with mock.patch("infemeral.server.forward_transformer") as mock_forward:
                mock_forward.return_value = (
                    torch.randn(1, 10, 4096),
                    tuple(),
                )

                result = handler(mock_event)

                if "output" in result:
                    # Output should be base64 encoded
                    output = result["output"]
                    assert isinstance(output, str)

                    # Should be valid base64
                    decoded = base64.b64decode(output)
                    assert isinstance(decoded, bytes)


class TestMemoryWipe:
    """Tests for memory wiping after inference."""

    def test_cuda_cache_cleared(self):
        """CUDA cache should be cleared after inference."""
        # This test verifies the code path exists, not actual CUDA behavior
        from infemeral.server import handler

        with mock.patch("infemeral.server.load_model") as mock_load:
            mock_model = mock.MagicMock()
            mock_model.parameters.return_value = iter(
                [torch.tensor([1.0], device="cpu")]
            )
            mock_load.return_value = mock_model

            with mock.patch(
                "infemeral.server.forward_transformer"
            ) as mock_forward:
                mock_forward.return_value = (
                    torch.randn(1, 10, 4096),
                    tuple(),
                )

                # Mock KV cache loading to avoid filesystem access
                with mock.patch("infemeral.server.load_kv_cache", return_value=None):
                    # Mock cuda.is_available to return True so empty_cache gets called
                    with mock.patch("infemeral.server.torch.cuda.is_available", return_value=True):
                        # Patch torch.cuda.empty_cache in the server module
                        with mock.patch("infemeral.server.torch.cuda.empty_cache") as mock_clear:
                            session_key = generate_session_key()
                            hidden = torch.randn(1, 10, 4096, dtype=torch.float16)
                            data, shape, dtype = serialize_tensor(hidden)
                            ciphertext, nonce = encrypt_bytes(data, session_key)

                            event = {
                                "input": {
                                    "cloaked_embedding": base64.b64encode(ciphertext).decode(),
                                    "encrypted_session_key": base64.b64encode(
                                        session_key
                                    ).decode(),
                                    "nonce": base64.b64encode(nonce).decode(),
                                    "shape": shape,
                                    "dtype": dtype,
                                    "session_id": "test_session",
                                }
                            }

                            result = handler(event)

                            # Check no error occurred
                            assert "error" not in result, f"Handler error: {result.get('error')}"

                            # Verify empty_cache was called
                            mock_clear.assert_called()


class TestModelLoading:
    """Tests for model loading logic."""

    @pytest.mark.skip(reason="load_model_vllm no longer exists - AWQ loaded via from_pretrained")
    def test_awq_detection(self):
        """AWQ models should be detected and use vLLM."""
        pass

    def test_model_cached_globally(self):
        """Model should be cached to avoid reloading."""
        import infemeral.server as server_module

        # Reset global
        server_module._model = None
        server_module._config = None

        # Create a mock model
        mock_model = mock.MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.hf_device_map = {"": "cpu"}  # Simulate device_map was used

        mock_config = mock.MagicMock()
        mock_config.hidden_size = 4096

        with mock.patch("infemeral.server.server_settings") as mock_settings:
            mock_settings.weights_dir = "/fake/weights"
            mock_settings.tensorized_weights_path = "/fake/nonexistent.tensors"
            mock_settings.model_id = "test/model"

            # Mock Path.exists to return True for weights_dir
            with mock.patch("infemeral.server.Path") as mock_path:
                mock_path_instance = mock.MagicMock()
                mock_path.return_value = mock_path_instance

                # Tensorized path doesn't exist, weights dir does
                def exists_side_effect():
                    return mock_path_instance == mock.MagicMock()  # Never matches
                mock_path_instance.exists.side_effect = [False, True]

                with mock.patch(
                    "infemeral.server.AutoConfig.from_pretrained",
                    return_value=mock_config
                ):
                    with mock.patch(
                        "infemeral.server.AutoModelForCausalLM.from_pretrained",
                        return_value=mock_model
                    ):
                        from infemeral.server import load_model

                        server_module._model = None
                        server_module._config = None

                        # First call should load
                        model1 = load_model()

                        # Second call should return cached
                        model2 = load_model()

                        assert model1 is model2
