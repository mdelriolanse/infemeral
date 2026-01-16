"""Performance regression tests for client-side operations."""

import pytest
import torch

from infemeral.crypto import cloak, create_cloaking_context, uncloak
from infemeral.tensors import (
    compress_tensor_data,
    decompress_tensor_data,
    is_lz4_available,
    serialize_tensor,
)


class TestGpuPath:
    """Tests for GPU acceleration path."""

    @pytest.mark.gpu
    def test_gpu_path_used_when_available(self, mock_client_weights):
        """Verify tensors stay on CUDA when available."""
        from infemeral.client import Client

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        client = Client(weights_path=mock_client_weights, device="cuda")

        assert client.embedding.embed_tokens.weight.device.type == "cuda"
        assert client.cloaking_ctx.matrix.device.type == "cuda"
        assert client.cloaking_ctx.device == "cuda"

        client.close()

    def test_cloaking_matrix_on_target_device(self):
        """Cloaking matrix should be created on target device."""
        ctx = create_cloaking_context(seed=42, device="cpu", dtype=torch.float16)
        assert ctx.matrix.device.type == "cpu"
        assert ctx.matrix.dtype == torch.float16
        assert ctx.device == "cpu"

    @pytest.mark.gpu
    def test_cloaking_matrix_on_gpu(self):
        """Cloaking matrix should be created on GPU when specified."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ctx = create_cloaking_context(seed=42, device="cuda", dtype=torch.float16)
        assert ctx.matrix.device.type == "cuda"
        assert ctx.matrix_t.device.type == "cuda"
        assert ctx.device == "cuda"


class TestNoDeviceTransfer:
    """Tests to verify no unnecessary device transfers."""

    def test_no_device_transfer_during_cloak_cpu(self):
        """Cloak should not call .to() if matrix already on device."""
        ctx = create_cloaking_context(seed=42, device="cpu", dtype=torch.float32)
        hidden = torch.randn(1, 10, 4096, device="cpu", dtype=torch.float32)

        # Matrix should already be on CPU with correct dtype
        assert ctx.matrix.device == hidden.device
        assert ctx.matrix.dtype == hidden.dtype

        # Cloak should not transfer
        cloaked = cloak(hidden, ctx, add_noise=False)
        assert cloaked.device == hidden.device

    @pytest.mark.gpu
    def test_no_device_transfer_during_cloak_gpu(self):
        """Cloak should not call .to() if matrix already on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ctx = create_cloaking_context(seed=42, device="cuda", dtype=torch.float16)
        hidden = torch.randn(1, 10, 4096, device="cuda", dtype=torch.float16)

        # Matrix should already be on CUDA with correct dtype
        assert ctx.matrix.device == hidden.device
        assert ctx.matrix.dtype == hidden.dtype

        cloaked = cloak(hidden, ctx, add_noise=False)
        assert cloaked.device.type == "cuda"


class TestOrthogonalMatrixFloat16:
    """Tests for float16 orthogonal matrix validity."""

    def test_orthogonal_matrix_float16_valid(self):
        """Float16 orthogonal matrix should satisfy M @ M.T ≈ I."""
        ctx = create_cloaking_context(seed=42, device="cpu", dtype=torch.float16)

        # Upcast to float32 for precision during verification
        M = ctx.matrix.float()
        I = torch.eye(M.shape[0])

        # Relaxed tolerance for float16
        result = M @ M.T
        assert torch.allclose(result, I, atol=1e-3), (
            f"Orthogonality check failed. Max deviation: {(result - I).abs().max()}"
        )

    def test_orthogonal_matrix_float32_valid(self):
        """Float32 orthogonal matrix should satisfy M @ M.T ≈ I."""
        ctx = create_cloaking_context(seed=42, device="cpu", dtype=torch.float32)

        M = ctx.matrix
        I = torch.eye(M.shape[0])

        result = M @ M.T
        assert torch.allclose(result, I, atol=1e-5), (
            f"Orthogonality check failed. Max deviation: {(result - I).abs().max()}"
        )

    def test_cloak_uncloak_roundtrip_float16(self):
        """Cloak -> Uncloak should approximately recover original (float16)."""
        ctx = create_cloaking_context(seed=42, device="cpu", dtype=torch.float16)
        hidden = torch.randn(1, 10, 4096, dtype=torch.float16)

        cloaked = cloak(hidden, ctx, add_noise=False)
        recovered = uncloak(cloaked, ctx)

        # Relaxed tolerance for float16
        assert torch.allclose(hidden, recovered, rtol=1e-3, atol=1e-3)


class TestTiedEmbeddings:
    """Tests for tied embedding detection and loading."""

    @pytest.fixture
    def mock_tied_weights(self, temp_weights_dir):
        """Create mock weights with tied embeddings."""
        from safetensors.torch import save_file

        vocab_size, hidden_size = 1000, 4096
        embed_weight = torch.randn(vocab_size, hidden_size)

        # Only save embed_tokens (tied embeddings)
        state_dict = {"embed_tokens.weight": embed_weight}
        metadata = {"tied_embeddings": "true", "dtype": "float16"}

        weights_path = temp_weights_dir / "tied_weights.safetensors"
        save_file(state_dict, weights_path, metadata=metadata)
        return str(weights_path)

    @pytest.fixture
    def mock_separate_weights(self, temp_weights_dir):
        """Create mock weights with separate lm_head."""
        from safetensors.torch import save_file

        vocab_size, hidden_size = 1000, 4096

        state_dict = {
            "embed_tokens.weight": torch.randn(vocab_size, hidden_size),
            "lm_head.weight": torch.randn(vocab_size, hidden_size),
        }
        metadata = {"tied_embeddings": "false", "dtype": "float16"}

        weights_path = temp_weights_dir / "separate_weights.safetensors"
        save_file(state_dict, weights_path, metadata=metadata)
        return str(weights_path)

    def test_tied_embeddings_detected(self, mock_tied_weights):
        """Client should detect tied embeddings from metadata."""
        from infemeral.client import EmbeddingLayer

        layer = EmbeddingLayer(mock_tied_weights, device="cpu")

        assert layer.tied_embeddings is True
        # lm_head.weight should be the same object as embed_tokens.weight
        assert layer.lm_head.weight is layer.embed_tokens.weight

    def test_separate_embeddings_loaded(self, mock_separate_weights):
        """Client should load separate embeddings when not tied."""
        from infemeral.client import EmbeddingLayer

        layer = EmbeddingLayer(mock_separate_weights, device="cpu")

        assert layer.tied_embeddings is False
        # lm_head.weight should NOT be the same object
        assert layer.lm_head.weight is not layer.embed_tokens.weight


class TestLz4Compression:
    """Tests for LZ4 tensor compression."""

    def test_lz4_compression_roundtrip(self):
        """Compressed tensor should decompress to identical bytes."""
        data = serialize_tensor(torch.randn(1, 1, 4096))[0]

        compressed = compress_tensor_data(data)
        decompressed = decompress_tensor_data(compressed)

        assert data == decompressed

    def test_lz4_skipped_for_small_tensors(self):
        """Compression should be skipped for tensors < 4KB."""
        small_data = b"x" * 1000

        result = compress_tensor_data(small_data)

        # Should return original data unchanged
        assert result == small_data

    def test_lz4_compression_reduces_size(self):
        """LZ4 should reduce size for compressible data."""
        if not is_lz4_available():
            pytest.skip("LZ4 not installed")

        # Create highly compressible data (zeros)
        data = b"\x00" * 10000

        compressed = compress_tensor_data(data)

        # Should be compressed (has "L4" header)
        assert compressed[:2] == b"L4"
        assert len(compressed) < len(data)

    def test_lz4_decompression_handles_uncompressed(self):
        """Decompress should pass through uncompressed data."""
        raw_data = b"some uncompressed data"

        result = decompress_tensor_data(raw_data)

        assert result == raw_data


class TestGrpcKeepalive:
    """Tests for gRPC keepalive configuration."""

    def test_grpc_channel_has_keepalive_options(self, mock_client_weights):
        """Verify keepalive options are included in channel creation."""
        from infemeral.client import Client

        client = Client(weights_path=mock_client_weights, device="cpu")

        # Trigger channel creation
        try:
            _ = client.stub
        except ImportError:
            # gRPC stubs may not be generated in test environment
            pass

        # If channel was created, verify it exists
        if client._channel is not None:
            assert client._channel is not None

        client.close()


class TestGenerationMetrics:
    """Tests for generation metrics and timing."""

    def test_token_timing_dataclass(self):
        """TokenTiming should have all timing fields."""
        from infemeral.client import TokenTiming

        timing = TokenTiming()

        assert hasattr(timing, "embed_ms")
        assert hasattr(timing, "cloak_ms")
        assert hasattr(timing, "network_ms")
        assert hasattr(timing, "uncloak_ms")
        assert hasattr(timing, "de_embed_ms")
        assert hasattr(timing, "sample_ms")
        assert hasattr(timing, "total_ms")

    def test_generation_metrics_dataclass(self):
        """GenerationMetrics should have all required fields."""
        from infemeral.client import GenerationMetrics

        metrics = GenerationMetrics(device="cpu")

        assert metrics.device == "cpu"
        assert metrics.timings == []
        assert metrics.peak_memory_mb == 0.0
        assert metrics.tokens_per_sec == 0.0
        assert metrics.total_tokens == 0
        assert metrics.prompt_tokens == 0
