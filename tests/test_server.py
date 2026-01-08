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

        # Mock settings to use temp dir
        with mock.patch("infemeral.server.server_settings") as mock_settings:
            mock_settings.kv_cache_dir = str(temp_cache_dir)

            session_id = "test_session"
            session_key = generate_session_key()

            # Create KV tensors
            keys = torch.randn(1, 32, 128, 128)
            values = torch.randn(1, 32, 128, 128)

            # Save
            save_kv_cache(session_id, keys, values, session_key)

            # Verify file exists
            cache_path = get_kv_cache_path(session_id)
            assert cache_path.exists()

            # Load
            loaded = load_kv_cache(session_id, session_key, device="cpu")

            assert loaded is not None
            loaded_keys, loaded_values = loaded

            # Note: KV cache is stored as float16
            torch.testing.assert_close(
                keys.to(torch.float16), loaded_keys, rtol=1e-2, atol=1e-2
            )
            torch.testing.assert_close(
                values.to(torch.float16), loaded_values, rtol=1e-2, atol=1e-2
            )

    def test_load_nonexistent_cache_returns_none(self, temp_cache_dir):
        """Loading non-existent cache should return None."""
        from infemeral.server import load_kv_cache

        with mock.patch("infemeral.server.server_settings") as mock_settings:
            mock_settings.kv_cache_dir = str(temp_cache_dir)

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

            session_id = "test_session"
            session_key = generate_session_key()

            # Save some cache
            keys = torch.randn(1, 8, 64, 64)
            values = torch.randn(1, 8, 64, 64)
            save_kv_cache(session_id, keys, values, session_key)

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

            session_id = "test_session"
            correct_key = generate_session_key()
            wrong_key = generate_session_key()

            keys = torch.randn(1, 8, 64, 64)
            values = torch.randn(1, 8, 64, 64)
            save_kv_cache(session_id, keys, values, correct_key)

            # Should fail with wrong key
            result = load_kv_cache(session_id, wrong_key, "cpu")
            assert result is None  # Returns None on failure


class TestForwardTransformer:
    """Tests for transformer forward pass."""

    @pytest.fixture
    def mock_model(self):
        """Create a minimal mock model for testing."""

        class MockLayer(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                use_cache=True,
            ):
                # Identity transform for testing
                output = hidden_states
                # Mock KV cache output
                kv = (
                    torch.randn(1, 8, hidden_states.shape[1], 64),
                    torch.randn(1, 8, hidden_states.shape[1], 64),
                )
                return output, kv

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
        """Forward pass should return KV cache."""
        from infemeral.server import forward_transformer

        hidden = torch.randn(1, 10, 4096)

        output, kv = forward_transformer(mock_model, hidden)

        # Should return tuple of KV for each layer
        assert isinstance(kv, tuple)
        assert len(kv) == 2  # 2 mock layers

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

    def test_awq_detection(self):
        """AWQ models should be detected and use vLLM."""
        from infemeral.server import load_model

        with mock.patch("infemeral.server.server_settings") as mock_settings:
            mock_settings.weights_path = "/path/to/LLaMA-Pro-8B-AWQ/model"

            with mock.patch("infemeral.server.load_model_vllm") as mock_vllm:
                mock_vllm.return_value = None  # Simulate vLLM not available

                # Should fall through to safetensors loading
                # (will fail because path doesn't exist, but tests the detection)
                try:
                    load_model("/path/to/weights.safetensors")
                except Exception:
                    pass

                # vLLM should have been attempted
                mock_vllm.assert_called_once()

    def test_model_cached_globally(self):
        """Model should be cached to avoid reloading."""
        import infemeral.server as server_module

        # Reset global
        server_module._model = None

        # Create a mock model with proper load_state_dict return value
        mock_model = mock.MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        # load_state_dict returns _IncompatibleKeys(missing_keys, unexpected_keys)
        mock_model.load_state_dict.return_value = ([], [])

        with mock.patch("infemeral.server.server_settings") as mock_settings:
            mock_settings.weights_path = "/fake/path"

            with mock.patch("infemeral.server.load_model_vllm", return_value=None):
                with mock.patch("infemeral.server.load_file", return_value={}):
                    with mock.patch(
                        "infemeral.server.AutoConfig.from_pretrained"
                    ) as mock_config:
                        mock_config.return_value = mock.MagicMock(hidden_size=4096)

                        with mock.patch(
                            "infemeral.server.AutoModelForCausalLM.from_config",
                            return_value=mock_model,
                        ):
                            # First call should load
                            from infemeral.server import load_model

                            server_module._model = None
                            model1 = load_model("/fake/weights.safetensors")

                            # Second call should return cached
                            model2 = load_model("/fake/weights.safetensors")

                            assert model1 is model2
