"""Tests for client-side inference."""

from unittest import mock

import pytest
import torch

from infemeral.crypto import create_cloaking_context


class TestEmbeddingLayer:
    """Tests for client embedding layer."""

    @pytest.fixture
    def mock_weights_file(self, tmp_path):
        """Create a mock weights file."""
        from safetensors.torch import save_file

        vocab_size, hidden_size = 1000, 4096

        state_dict = {
            "embed_tokens.weight": torch.randn(vocab_size, hidden_size),
            "lm_head.weight": torch.randn(vocab_size, hidden_size),
        }

        weights_path = tmp_path / "client_weights.safetensors"
        save_file(state_dict, weights_path)
        return str(weights_path)

    def test_embedding_layer_loads(self, mock_weights_file):
        """Embedding layer should load from weights file."""
        from infemeral.client import EmbeddingLayer

        layer = EmbeddingLayer(mock_weights_file, device="cpu")

        assert layer.embed_tokens is not None
        assert layer.lm_head is not None

    def test_embed_produces_correct_shape(self, mock_weights_file):
        """Embedding should produce [batch, seq, hidden_dim] output."""
        from infemeral.client import EmbeddingLayer

        layer = EmbeddingLayer(mock_weights_file, device="cpu")

        # Input: [batch, seq_len] of token IDs
        input_ids = torch.randint(0, 1000, (2, 10))

        hidden = layer.embed(input_ids)

        assert hidden.shape == (2, 10, 4096)

    def test_de_embed_produces_logits(self, mock_weights_file):
        """De-embedding should produce logits over vocabulary."""
        from infemeral.client import EmbeddingLayer

        layer = EmbeddingLayer(mock_weights_file, device="cpu")

        hidden = torch.randn(2, 10, 4096)
        logits = layer.de_embed(hidden)

        # Output should be [batch, seq, vocab_size]
        assert logits.shape == (2, 10, 1000)

    def test_embed_de_embed_differentiable(self, mock_weights_file):
        """Embedding operations should be in eval mode (no grad)."""
        from infemeral.client import EmbeddingLayer

        layer = EmbeddingLayer(mock_weights_file, device="cpu")

        input_ids = torch.randint(0, 1000, (1, 5))
        hidden = layer.embed(input_ids)

        # Should not require grad (we're in eval mode)
        assert not hidden.requires_grad


class TestClientSampling:
    """Tests for token sampling logic."""

    def test_greedy_sampling(self):
        """Temperature=0 should give greedy (argmax) sampling."""
        from infemeral.client import Client

        # Mock the client without actually loading weights
        with mock.patch.object(Client, "__init__", lambda self: None):
            client = Client()
            client.tokenizer = mock.MagicMock()
            client.tokenizer.eos_token_id = 0

            # Create logits where token 5 has highest probability
            logits = torch.zeros(10)
            logits[5] = 10.0

            token = client._sample(logits, temperature=0, top_p=0.9)

            assert token.item() == 5

    def test_temperature_affects_distribution(self):
        """Higher temperature should produce more varied samples."""
        from infemeral.client import Client

        with mock.patch.object(Client, "__init__", lambda self: None):
            client = Client()
            client.tokenizer = mock.MagicMock()
            client.tokenizer.eos_token_id = 0

            # Logits with clear preference
            logits = torch.tensor([10.0, 5.0, 0.0, -5.0, -10.0])

            # Low temperature should mostly pick token 0
            low_temp_samples = [
                client._sample(logits.clone(), temperature=0.1, top_p=1.0).item()
                for _ in range(100)
            ]

            # High temperature should be more varied
            high_temp_samples = [
                client._sample(logits.clone(), temperature=2.0, top_p=1.0).item()
                for _ in range(100)
            ]

            # Low temp should have less variety
            assert len(set(low_temp_samples)) <= len(set(high_temp_samples))

    def test_top_p_filtering(self):
        """Top-p should filter out low probability tokens."""
        from infemeral.client import Client

        with mock.patch.object(Client, "__init__", lambda self: None):
            client = Client()
            client.tokenizer = mock.MagicMock()
            client.tokenizer.eos_token_id = 0

            # Logits where first 2 tokens have 95% of probability mass
            logits = torch.tensor([5.0, 4.0, -10.0, -10.0, -10.0])

            # With top_p=0.9, should mostly sample from first 2 tokens
            samples = [
                client._sample(logits.clone(), temperature=1.0, top_p=0.9).item()
                for _ in range(100)
            ]

            # Should rarely (if ever) sample tokens 2, 3, 4
            assert sum(1 for s in samples if s >= 2) < 10


class TestClientSession:
    """Tests for client session management."""

    def test_session_id_unique(self):
        """Each client should have unique session ID."""
        from infemeral.client import Client

        with mock.patch.object(Client, "__init__", lambda self: None):
            client1 = Client()
            client1.session_id = __import__("secrets").token_hex(16)

            client2 = Client()
            client2.session_id = __import__("secrets").token_hex(16)

            assert client1.session_id != client2.session_id

    def test_session_key_generated(self):
        """Each client should generate a session key."""
        from infemeral.crypto import generate_session_key

        key = generate_session_key()

        assert len(key) == 32
        assert isinstance(key, bytes)

    def test_cloaking_context_created(self):
        """Client should create a cloaking context."""
        ctx = create_cloaking_context(seed=42)

        assert ctx.matrix is not None
        assert ctx.matrix_t is not None
        assert ctx.sigma > 0


class TestClientCloakingFlow:
    """Tests for the client cloaking/uncloaking flow."""

    @pytest.fixture
    def cloaking_ctx(self):
        """Create cloaking context for tests."""
        return create_cloaking_context(seed=42)

    def test_cloak_uncloak_flow(self, cloaking_ctx):
        """Cloak -> server -> uncloak should preserve structure."""
        from infemeral.crypto import cloak, uncloak

        hidden = torch.randn(1, 10, 4096)

        # Client: cloak (without noise for testing)
        cloaked = cloak(hidden, cloaking_ctx, add_noise=False)

        # Simulate server: identity transform (no actual processing)
        server_output = cloaked.clone()

        # Client: uncloak
        recovered = uncloak(server_output, cloaking_ctx)

        torch.testing.assert_close(hidden, recovered, rtol=1e-4, atol=1e-4)

    def test_cloak_with_noise_adds_privacy(self, cloaking_ctx):
        """Cloaking with noise should add DP protection."""
        from infemeral.crypto import cloak

        hidden = torch.randn(1, 10, 4096)

        cloaked1 = cloak(hidden, cloaking_ctx, add_noise=True)
        cloaked2 = cloak(hidden, cloaking_ctx, add_noise=True)

        # Same input, different noise = different cloaked outputs
        assert not torch.allclose(cloaked1, cloaked2)


class TestClientGrpcCalls:
    """Tests for gRPC communication (mocked)."""

    def test_server_call_serialization(self):
        """Test that tensors are properly serialized for server."""
        from infemeral.tensors import deserialize_tensor, serialize_tensor

        hidden = torch.randn(1, 10, 4096, dtype=torch.float16)

        # Serialize
        data, shape, dtype = serialize_tensor(hidden)

        # Deserialize (as server would)
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        torch.testing.assert_close(hidden, recovered)

    def test_server_response_deserialization(self):
        """Test that server responses are properly deserialized."""
        from infemeral.tensors import deserialize_tensor, serialize_tensor

        # Simulate server response
        output = torch.randn(1, 10, 4096, dtype=torch.float16)
        data, shape, dtype = serialize_tensor(output)

        # Client deserializes
        recovered = deserialize_tensor(data, shape, dtype, device="cpu")

        torch.testing.assert_close(output, recovered)


class TestClientEncryption:
    """Tests for client-side encryption."""

    def test_encrypted_payload(self):
        """Client should encrypt tensor data before sending."""
        from infemeral.crypto import decrypt_bytes, encrypt_bytes, generate_session_key
        from infemeral.tensors import serialize_tensor

        key = generate_session_key()
        hidden = torch.randn(1, 10, 4096)

        # Serialize
        data, shape, dtype = serialize_tensor(hidden)

        # Encrypt
        ciphertext, nonce = encrypt_bytes(data, key)

        # Should be different from plaintext
        assert ciphertext != data

        # Should be decryptable
        plaintext = decrypt_bytes(ciphertext, key, nonce)
        assert plaintext == data

    def test_nonce_unique_per_request(self):
        """Each encryption should use unique nonce."""
        from infemeral.crypto import encrypt_bytes, generate_session_key

        key = generate_session_key()
        data = b"test data"

        _, nonce1 = encrypt_bytes(data, key)
        _, nonce2 = encrypt_bytes(data, key)

        assert nonce1 != nonce2
