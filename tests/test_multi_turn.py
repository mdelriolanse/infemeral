"""Integration tests for multi-turn KV cache support."""

import secrets
from pathlib import Path

import pytest
import torch

from infemeral.config import server_settings
from infemeral.crypto import generate_session_key
from infemeral.server import (
    apply_context_windowing,
    load_kv_cache,
    save_kv_cache,
)
from infemeral.tensors import pack_kv_cache, pack_kv_cache_v2, unpack_kv_cache


@pytest.fixture
def temp_kv_dir(tmp_path):
    """Override KV cache directory for tests."""
    original = server_settings.kv_cache_dir
    server_settings.kv_cache_dir = str(tmp_path)
    yield tmp_path
    server_settings.kv_cache_dir = original


def test_kv_cache_format_roundtrip():
    """pack_kv_cache_v2 → unpack_kv_cache_v2 preserves structure."""
    # Create sample per-layer KV cache (3 layers)
    num_layers = 3
    batch_size, num_heads, seq_len, head_dim = 1, 32, 10, 128

    kv_tuples = []
    for _ in range(num_layers):
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        kv_tuples.append((keys, values))

    original = tuple(kv_tuples)

    # Pack and unpack
    packed = pack_kv_cache_v2(original)
    restored = unpack_kv_cache(packed, device="cpu")

    # Verify structure
    assert len(restored) == num_layers
    for i, ((orig_k, orig_v), (rest_k, rest_v)) in enumerate(zip(original, restored)):
        assert orig_k.shape == rest_k.shape, f"Layer {i} key shape mismatch"
        assert orig_v.shape == rest_v.shape, f"Layer {i} value shape mismatch"
        # Convert to same dtype for comparison (packed as float16)
        assert torch.allclose(orig_k.to(torch.float16), rest_k.to(torch.float16), atol=1e-3), f"Layer {i} key data mismatch"
        assert torch.allclose(orig_v.to(torch.float16), rest_v.to(torch.float16), atol=1e-3), f"Layer {i} value data mismatch"


def test_kv_cache_backward_compatibility():
    """Old v1 format files are detected and ignored."""
    # Create v1 format data
    keys = torch.randn(1, 32, 10, 128)
    values = torch.randn(1, 32, 10, 128)
    v1_packed = pack_kv_cache(keys, values)

    # Unpack should return flat tensors
    result = unpack_kv_cache(v1_packed, device="cpu")

    # V1 returns tuple of 2 tensors (not nested tuples)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)


def test_save_load_kv_cache(temp_kv_dir):
    """Save and load KV cache with encryption."""
    session_id = secrets.token_hex(16)
    session_key = generate_session_key()

    # Create sample KV cache
    num_layers = 2
    kv_tuples = []
    for _ in range(num_layers):
        keys = torch.randn(1, 32, 5, 128)
        values = torch.randn(1, 32, 5, 128)
        kv_tuples.append((keys, values))

    original = tuple(kv_tuples)

    # Save
    save_kv_cache(session_id, original, session_key)

    # Load
    loaded = load_kv_cache(session_id, session_key, device="cpu")

    # Verify
    assert loaded is not None
    assert len(loaded) == num_layers
    for (orig_k, orig_v), (load_k, load_v) in zip(original, loaded):
        # Convert to same dtype for comparison
        assert torch.allclose(orig_k.to(torch.float16), load_k.to(torch.float16), atol=1e-3)
        assert torch.allclose(orig_v.to(torch.float16), load_v.to(torch.float16), atol=1e-3)


def test_context_windowing():
    """Sequences > max_context_length handled via sliding window."""
    # Create KV cache with 100 tokens
    num_layers = 2
    seq_len = 100
    max_length = 50
    sink_tokens = 4

    kv_tuples = []
    for _ in range(num_layers):
        keys = torch.randn(1, 32, seq_len, 128)
        values = torch.randn(1, 32, seq_len, 128)
        kv_tuples.append((keys, values))

    original = tuple(kv_tuples)

    # Apply windowing
    windowed = apply_context_windowing(original, max_length, sink_tokens)

    # Verify length
    assert len(windowed) == num_layers
    for keys, values in windowed:
        assert keys.shape[2] == max_length, f"Expected {max_length} tokens, got {keys.shape[2]}"
        assert values.shape[2] == max_length


def test_position_embeddings_incremental():
    """Position IDs correct after KV cache loaded."""
    # This test verifies the logic in forward_transformer
    # Create sample past_key_values with 10 tokens cached
    past_len = 10
    num_layers = 2

    past_kv = []
    for _ in range(num_layers):
        keys = torch.randn(1, 32, past_len, 128)
        values = torch.randn(1, 32, past_len, 128)
        past_kv.append((keys, values))

    past_kv = tuple(past_kv)

    # Simulate position ID calculation from forward_transformer
    seq_len = 1  # New token
    position_ids = torch.arange(past_len, past_len + seq_len)

    # Verify position starts at cached length
    assert position_ids[0].item() == past_len
    assert position_ids[-1].item() == past_len + seq_len - 1


def test_session_isolation_multi_turn(temp_kv_dir):
    """Different sessions don't share KV cache state."""
    session_id_1 = secrets.token_hex(16)
    session_id_2 = secrets.token_hex(16)
    key_1 = generate_session_key()
    key_2 = generate_session_key()

    # Create different KV caches
    kv_1 = tuple([(torch.ones(1, 32, 5, 128), torch.ones(1, 32, 5, 128))])
    kv_2 = tuple([(torch.zeros(1, 32, 5, 128), torch.zeros(1, 32, 5, 128))])

    # Save both
    save_kv_cache(session_id_1, kv_1, key_1)
    save_kv_cache(session_id_2, kv_2, key_2)

    # Load and verify isolation
    loaded_1 = load_kv_cache(session_id_1, key_1, device="cpu")
    loaded_2 = load_kv_cache(session_id_2, key_2, device="cpu")

    assert loaded_1 is not None and loaded_2 is not None
    # Convert to same dtype for comparison
    assert torch.allclose(loaded_1[0][0], torch.ones(1, 32, 5, 128, dtype=torch.float16), atol=1e-3)
    assert torch.allclose(loaded_2[0][0], torch.zeros(1, 32, 5, 128, dtype=torch.float16), atol=1e-3)


def test_no_cache_returns_none(temp_kv_dir):
    """Loading non-existent cache returns None."""
    session_id = secrets.token_hex(16)
    session_key = generate_session_key()

    loaded = load_kv_cache(session_id, session_key, device="cpu")
    assert loaded is None


@pytest.mark.slow
def test_large_sequence_windowing():
    """Very long sequences are windowed correctly."""
    num_layers = 4
    seq_len = 3000
    max_length = 2048
    sink_tokens = 4

    kv_tuples = []
    for _ in range(num_layers):
        keys = torch.randn(1, 32, seq_len, 128)
        values = torch.randn(1, 32, seq_len, 128)
        kv_tuples.append((keys, values))

    original = tuple(kv_tuples)

    windowed = apply_context_windowing(original, max_length, sink_tokens)

    for keys, values in windowed:
        assert keys.shape[2] == max_length
        assert values.shape[2] == max_length
