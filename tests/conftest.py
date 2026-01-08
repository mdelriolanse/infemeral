"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from infemeral.crypto import create_cloaking_context, generate_session_key


@pytest.fixture(scope="session")
def device():
    """Get the best available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def session_key():
    """Generate a fresh session key."""
    return generate_session_key()


@pytest.fixture
def cloaking_context():
    """Create a cloaking context with fixed seed for reproducibility."""
    return create_cloaking_context(seed=42)


@pytest.fixture
def sample_hidden_states():
    """Create sample hidden states for testing."""
    return torch.randn(1, 10, 4096)


@pytest.fixture
def sample_hidden_states_f16():
    """Create sample hidden states in float16."""
    return torch.randn(1, 10, 4096, dtype=torch.float16)


@pytest.fixture
def temp_weights_dir(tmp_path):
    """Create a temporary directory for weight files."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    return weights_dir


@pytest.fixture
def mock_client_weights(temp_weights_dir):
    """Create mock client weights file."""
    from safetensors.torch import save_file

    vocab_size, hidden_size = 1000, 4096

    state_dict = {
        "embed_tokens.weight": torch.randn(vocab_size, hidden_size),
        "lm_head.weight": torch.randn(vocab_size, hidden_size),
    }

    weights_path = temp_weights_dir / "client_weights.safetensors"
    save_file(state_dict, weights_path)
    return str(weights_path)


@pytest.fixture
def mock_kv_cache():
    """Create mock KV cache tensors."""
    # Shape: [batch, num_heads, seq_len, head_dim]
    keys = torch.randn(1, 32, 128, 128)
    values = torch.randn(1, 32, 128, 128)
    return keys, values


# Configure pytest marks
def pytest_configure(config):
    """Configure custom pytest marks."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")


# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU is available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
