"""Integration tests for gRPC server."""

from concurrent import futures
from unittest import mock

import grpc
import pytest
import torch

from infemeral.crypto import encrypt_bytes, generate_session_key
from infemeral.tensors import deserialize_tensor, serialize_tensor

# Import gRPC stubs
try:
    from infemeral import tensor_service_pb2, tensor_service_pb2_grpc
except ImportError:
    import tensor_service_pb2
    import tensor_service_pb2_grpc


@pytest.fixture(scope="module")
def mock_model():
    """Create a minimal mock model for testing."""

    class MockRotaryEmb(torch.nn.Module):
        def __init__(self, head_dim=128):
            super().__init__()
            self.dim = head_dim

        def forward(self, hidden_states, position_ids):
            batch_size, seq_len, _ = hidden_states.shape
            device = hidden_states.device
            cos = torch.ones(batch_size, seq_len, self.dim, device=device)
            sin = torch.zeros(batch_size, seq_len, self.dim, device=device)
            return cos, sin

    class MockAttention(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.rotary_emb = MockRotaryEmb()

    class MockLayer(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.self_attn = MockAttention(hidden_size)

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


@pytest.fixture(scope="module")
def grpc_server(mock_model, tmp_path_factory):
    """Start gRPC server in background thread for testing."""
    from infemeral.server import TensorInferenceServicer

    # Create temp directory for KV cache
    temp_dir = tmp_path_factory.mktemp("kv_cache")

    # Create server with mock model
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )

    # Patch server_settings to use temp directory
    with mock.patch("infemeral.server.server_settings") as mock_settings:
        mock_settings.kv_cache_dir = str(temp_dir)
        servicer = TensorInferenceServicer(mock_model, device="cpu")

    # Re-patch for the fixture's lifetime by monkey-patching the servicer
    import infemeral.server as server_module

    original_settings = server_module.server_settings

    class MockSettings:
        kv_cache_dir = str(temp_dir)

    server_module.server_settings = MockSettings()

    tensor_service_pb2_grpc.add_TensorInferenceServicer_to_server(servicer, server)

    # Use a random port to avoid conflicts
    port = server.add_insecure_port("[::]:0")
    server.start()

    yield f"localhost:{port}"

    server.stop(grace=0)
    server_module.server_settings = original_settings


@pytest.fixture
def grpc_stub(grpc_server):
    """Create gRPC stub connected to test server."""
    channel = grpc.insecure_channel(
        grpc_server,
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )
    stub = tensor_service_pb2_grpc.TensorInferenceStub(channel)
    yield stub
    channel.close()


@pytest.mark.integration
class TestGrpcIntegration:
    """Integration tests for gRPC server."""

    def test_single_token_inference(self, grpc_stub):
        """Send cloaked embedding and verify response shape."""
        session_key = generate_session_key()
        session_id = "test_session_single"

        # Create test hidden states
        hidden = torch.randn(1, 1, 4096, dtype=torch.float16)
        data, shape, dtype = serialize_tensor(hidden)

        # Encrypt
        ciphertext, nonce = encrypt_bytes(data, session_key)

        # Build request
        request = tensor_service_pb2.InferenceRequest(
            cloaked_embedding=ciphertext,
            encrypted_session_key=session_key,
            nonce=nonce,
            shape=shape,
            dtype=dtype,
            session_id=session_id,
        )

        # Call server
        response = grpc_stub.Infer(request)

        # Verify response
        assert not response.error, f"Server returned error: {response.error}"
        assert list(response.shape) == shape
        assert response.dtype == dtype
        assert response.tokens_processed == 1

        # Verify we can decrypt and deserialize response
        # Response format: nonce (12 bytes) + ciphertext
        resp_nonce = bytes(response.output[:12])
        resp_ciphertext = bytes(response.output[12:])

        from infemeral.crypto import decrypt_bytes

        decrypted = decrypt_bytes(resp_ciphertext, session_key, resp_nonce)
        output_tensor = deserialize_tensor(decrypted, list(response.shape), response.dtype, "cpu")
        assert output_tensor.shape == hidden.shape

    def test_multi_token_inference(self, grpc_stub):
        """Test inference with multiple tokens."""
        session_key = generate_session_key()
        session_id = "test_session_multi"

        # Create test hidden states with 10 tokens
        hidden = torch.randn(1, 10, 4096, dtype=torch.float16)
        data, shape, dtype = serialize_tensor(hidden)

        # Encrypt
        ciphertext, nonce = encrypt_bytes(data, session_key)

        # Build request
        request = tensor_service_pb2.InferenceRequest(
            cloaked_embedding=ciphertext,
            encrypted_session_key=session_key,
            nonce=nonce,
            shape=shape,
            dtype=dtype,
            session_id=session_id,
        )

        # Call server
        response = grpc_stub.Infer(request)

        # Verify response
        assert not response.error, f"Server returned error: {response.error}"
        assert response.tokens_processed == 10

    def test_session_isolation(self, grpc_stub):
        """Multiple sessions should be independent."""
        # Create two different sessions
        key1 = generate_session_key()
        key2 = generate_session_key()

        hidden1 = torch.randn(1, 1, 4096, dtype=torch.float16)
        hidden2 = torch.randn(1, 1, 4096, dtype=torch.float16)

        data1, shape1, dtype1 = serialize_tensor(hidden1)
        data2, shape2, dtype2 = serialize_tensor(hidden2)

        cipher1, nonce1 = encrypt_bytes(data1, key1)
        cipher2, nonce2 = encrypt_bytes(data2, key2)

        req1 = tensor_service_pb2.InferenceRequest(
            cloaked_embedding=cipher1,
            encrypted_session_key=key1,
            nonce=nonce1,
            shape=shape1,
            dtype=dtype1,
            session_id="session_1",
        )

        req2 = tensor_service_pb2.InferenceRequest(
            cloaked_embedding=cipher2,
            encrypted_session_key=key2,
            nonce=nonce2,
            shape=shape2,
            dtype=dtype2,
            session_id="session_2",
        )

        # Both requests should succeed independently
        resp1 = grpc_stub.Infer(req1)
        resp2 = grpc_stub.Infer(req2)

        assert not resp1.error
        assert not resp2.error

    def test_error_handling_invalid_nonce(self, grpc_stub):
        """Malformed request should return error in response or raise gRPC error."""
        session_key = generate_session_key()

        hidden = torch.randn(1, 1, 4096, dtype=torch.float16)
        data, shape, dtype = serialize_tensor(hidden)
        ciphertext, _ = encrypt_bytes(data, session_key)

        # Use wrong nonce
        wrong_nonce = b"wrong_nonce!"  # 12 bytes but wrong

        request = tensor_service_pb2.InferenceRequest(
            cloaked_embedding=ciphertext,
            encrypted_session_key=session_key,
            nonce=wrong_nonce,
            shape=shape,
            dtype=dtype,
            session_id="test_bad_nonce",
        )

        # Server should handle error gracefully - either return error in response
        # or raise a gRPC error with details
        try:
            response = grpc_stub.Infer(request)
            # If we get a response, it should have an error
            assert response.error
        except grpc.RpcError as e:
            # gRPC error is also acceptable - verify it has details
            assert e.code() == grpc.StatusCode.INTERNAL
            assert e.details()  # Should have error details


@pytest.mark.integration
class TestServicerUnit:
    """Unit tests for TensorInferenceServicer."""

    def test_servicer_init(self, mock_model):
        """Servicer should initialize with model and device."""
        from infemeral.server import TensorInferenceServicer

        servicer = TensorInferenceServicer(mock_model, device="cpu")
        assert servicer.model is mock_model
        assert servicer.device == "cpu"

    def test_servicer_handles_exception_gracefully(self, mock_model, tmp_path):
        """Servicer should catch exceptions and return error response."""
        from infemeral.server import TensorInferenceServicer

        # Mock settings to use temp directory
        with mock.patch("infemeral.server.server_settings") as mock_settings:
            mock_settings.kv_cache_dir = str(tmp_path)
            servicer = TensorInferenceServicer(mock_model, device="cpu")

            # Create a mock context
            mock_context = mock.MagicMock()

            # Create invalid request (will fail to decrypt)
            request = tensor_service_pb2.InferenceRequest(
                cloaked_embedding=b"invalid",
                encrypted_session_key=b"0" * 32,
                nonce=b"0" * 12,
                shape=[1, 1, 4096],
                dtype="float16",
                session_id="test",
            )

            response = servicer.Infer(request, mock_context)

            # Should return error response
            assert response.error
            mock_context.set_code.assert_called()
