"""Unit tests for in-memory KV cache functionality."""

import threading
import time

import pytest
import torch

from infemeral.server import (
    CachedSession,
    SessionKVCache,
    get_kv_cache_path,
    load_kv_cache,
    save_kv_cache,
    delete_kv_cache,
    cleanup_old_sessions,
    persist_session,
    get_session_kv_cache,
    load_kv_cache_from_disk,
    save_kv_cache_to_disk,
)
from infemeral.config import server_settings
from infemeral.crypto import generate_session_key


@pytest.fixture
def mock_kv_cache_layers():
    """Create mock KV cache with multiple layers."""
    num_layers = 32
    batch_size = 1
    num_heads = 32
    seq_len = 10
    head_dim = 128

    layers = []
    for _ in range(num_layers):
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
        layers.append((keys, values))
    return tuple(layers)


@pytest.fixture
def session_id():
    """Generate a test session ID."""
    import secrets
    return secrets.token_hex(16)


class TestCachedSession:
    """Tests for CachedSession dataclass."""

    def test_cached_session_creation(self, mock_kv_cache_layers, session_key):
        """Verify CachedSession stores data correctly."""
        cached = CachedSession(
            kv_cache=mock_kv_cache_layers,
            session_key=session_key,
            token_count=10,
        )

        assert cached.kv_cache == mock_kv_cache_layers
        assert cached.session_key == session_key
        assert cached.token_count == 10
        assert cached.last_checkpoint_token == 0
        assert cached.last_access > 0


class TestSessionKVCache:
    """Tests for SessionKVCache LRU cache."""

    def test_put_and_get(self, mock_kv_cache_layers, session_key, session_id):
        """Verify basic put/get operations."""
        cache = SessionKVCache(max_sessions=5)

        cache.put(session_id, mock_kv_cache_layers, session_key, token_count=10)
        result = cache.get(session_id, session_key)

        assert result is not None
        assert len(result) == len(mock_kv_cache_layers)

    def test_get_nonexistent_returns_none(self, session_key):
        """Verify get returns None for missing sessions."""
        cache = SessionKVCache(max_sessions=5)
        result = cache.get("nonexistent", session_key)
        assert result is None

    def test_lru_eviction(self, mock_kv_cache_layers, session_key):
        """Verify LRU eviction when cache is full."""
        cache = SessionKVCache(max_sessions=3)

        # Add 3 sessions
        for i in range(3):
            cache.put(f"session_{i}", mock_kv_cache_layers, session_key)

        # All 3 should exist
        assert len(cache) == 3

        # Add a 4th session - should evict session_0 (oldest)
        cache.put("session_3", mock_kv_cache_layers, session_key)

        assert len(cache) == 3
        assert cache.get("session_0", session_key) is None
        assert cache.get("session_1", session_key) is not None
        assert cache.get("session_2", session_key) is not None
        assert cache.get("session_3", session_key) is not None

    def test_access_updates_lru_order(self, mock_kv_cache_layers, session_key):
        """Verify accessing a session moves it to end of LRU."""
        cache = SessionKVCache(max_sessions=3)

        # Add 3 sessions
        for i in range(3):
            cache.put(f"session_{i}", mock_kv_cache_layers, session_key)

        # Access session_0, making it most recently used
        cache.get("session_0", session_key)

        # Add 2 more sessions to trigger evictions
        cache.put("session_3", mock_kv_cache_layers, session_key)
        cache.put("session_4", mock_kv_cache_layers, session_key)

        # session_0 should still exist (was accessed)
        # session_1 and session_2 should be evicted
        assert cache.get("session_0", session_key) is not None
        assert cache.get("session_1", session_key) is None
        assert cache.get("session_2", session_key) is None

    def test_delete(self, mock_kv_cache_layers, session_key, session_id):
        """Verify delete removes session from cache."""
        cache = SessionKVCache(max_sessions=5)

        cache.put(session_id, mock_kv_cache_layers, session_key)
        assert cache.get(session_id, session_key) is not None

        cache.delete(session_id)
        assert cache.get(session_id, session_key) is None

    def test_cleanup_expired(self, mock_kv_cache_layers, session_key):
        """Verify cleanup removes old sessions."""
        cache = SessionKVCache(max_sessions=5)

        # Add session with old timestamp
        cache.put("old_session", mock_kv_cache_layers, session_key)

        # Manually set last_access to 2 hours ago
        with cache._lock:
            cache._cache["old_session"].last_access = time.time() - 7200

        # Add a recent session
        cache.put("new_session", mock_kv_cache_layers, session_key)

        # Cleanup with 1 hour max age
        cleaned = cache.cleanup_expired(max_age_seconds=3600)

        assert cleaned == 1
        assert cache.get("old_session", session_key) is None
        assert cache.get("new_session", session_key) is not None

    def test_should_checkpoint(self, mock_kv_cache_layers, session_key, session_id, monkeypatch):
        """Verify checkpoint trigger based on token count."""
        monkeypatch.setattr(server_settings, "session_checkpoint_interval", 10)
        cache = SessionKVCache(max_sessions=5)

        cache.put(session_id, mock_kv_cache_layers, session_key, token_count=5)
        assert not cache.should_checkpoint(session_id)

        cache.put(session_id, mock_kv_cache_layers, session_key, token_count=15)
        assert cache.should_checkpoint(session_id)

    def test_thread_safety(self, mock_kv_cache_layers, session_key):
        """Verify cache is thread-safe."""
        cache = SessionKVCache(max_sessions=100)
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    session_id = f"worker_{worker_id}_session_{i}"
                    cache.put(session_id, mock_kv_cache_layers, session_key)
                    result = cache.get(session_id, session_key)
                    if result is None:
                        errors.append(f"Failed to get {session_id}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"


class TestKVCacheIntegration:
    """Integration tests for load/save KV cache with memory mode."""

    def test_memory_mode_no_disk_io(
        self, mock_kv_cache_layers, session_key, session_id, tmp_path, monkeypatch
    ):
        """Verify memory mode doesn't write to disk."""
        monkeypatch.setattr(server_settings, "kv_cache_mode", "memory")
        monkeypatch.setattr(server_settings, "kv_cache_dir", str(tmp_path / "kv"))
        monkeypatch.setattr(server_settings, "max_cached_sessions", 10)

        # Reset global cache
        import infemeral.server as server_module
        server_module._session_kv_cache = None

        # Save to memory
        save_kv_cache(session_id, mock_kv_cache_layers, session_key)

        # Verify no disk file
        cache_path = get_kv_cache_path(session_id)
        assert not cache_path.exists()

        # Verify can load from memory
        result = load_kv_cache(session_id, session_key)
        assert result is not None

    def test_disk_mode_legacy_behavior(
        self, mock_kv_cache_layers, session_key, session_id, tmp_path, monkeypatch
    ):
        """Verify disk mode writes directly to disk."""
        monkeypatch.setattr(server_settings, "kv_cache_mode", "disk")
        monkeypatch.setattr(server_settings, "kv_cache_dir", str(tmp_path / "kv"))

        # Save to disk
        save_kv_cache(session_id, mock_kv_cache_layers, session_key)

        # Verify disk file exists
        cache_path = get_kv_cache_path(session_id)
        assert cache_path.exists()

        # Verify can load from disk
        result = load_kv_cache(session_id, session_key)
        assert result is not None
        assert len(result) == len(mock_kv_cache_layers)

    def test_hybrid_mode_memory_with_checkpoint(
        self, mock_kv_cache_layers, session_key, session_id, tmp_path, monkeypatch
    ):
        """Verify hybrid mode uses memory with periodic disk checkpoints."""
        monkeypatch.setattr(server_settings, "kv_cache_mode", "hybrid")
        monkeypatch.setattr(server_settings, "kv_cache_dir", str(tmp_path / "kv"))
        monkeypatch.setattr(server_settings, "max_cached_sessions", 10)
        monkeypatch.setattr(server_settings, "session_checkpoint_interval", 20)

        # Reset global cache
        import infemeral.server as server_module
        server_module._session_kv_cache = None

        # Get cache path for verification
        cache_path = get_kv_cache_path(session_id)

        # First save - token count is 10 (from mock_kv_cache seq_len)
        # Below checkpoint interval of 20, so no disk write
        save_kv_cache(session_id, mock_kv_cache_layers, session_key)
        assert not cache_path.exists(), "Should not checkpoint below interval"

        # Update the cache with more tokens to trigger checkpoint
        # Create a larger KV cache to simulate token growth
        larger_kv = []
        for keys, values in mock_kv_cache_layers:
            # Extend seq_len dimension from 10 to 30
            new_keys = torch.cat([keys, keys, keys], dim=2)  # 30 tokens
            new_values = torch.cat([values, values, values], dim=2)
            larger_kv.append((new_keys, new_values))

        # Save with 30 tokens - should trigger checkpoint (30 - 0 >= 20)
        save_kv_cache(session_id, tuple(larger_kv), session_key)

        # Now disk file should exist
        assert cache_path.exists(), "Should checkpoint when token count exceeds interval"

    def test_delete_removes_from_both(
        self, mock_kv_cache_layers, session_key, session_id, tmp_path, monkeypatch
    ):
        """Verify delete removes from memory and disk."""
        monkeypatch.setattr(server_settings, "kv_cache_mode", "hybrid")
        monkeypatch.setattr(server_settings, "kv_cache_dir", str(tmp_path / "kv"))
        monkeypatch.setattr(server_settings, "max_cached_sessions", 10)

        # Reset global cache
        import infemeral.server as server_module
        server_module._session_kv_cache = None

        # Save and persist to disk
        save_kv_cache(session_id, mock_kv_cache_layers, session_key)
        persist_session(session_id)

        # Verify both exist
        cache_path = get_kv_cache_path(session_id)
        assert cache_path.exists()
        cache = get_session_kv_cache()
        assert session_id in cache._cache

        # Delete
        delete_kv_cache(session_id)

        # Verify both removed
        assert not cache_path.exists()
        assert session_id not in cache._cache


class TestDiskFallback:
    """Tests for disk fallback in hybrid mode."""

    def test_hybrid_loads_from_disk_on_miss(
        self, mock_kv_cache_layers, session_key, session_id, tmp_path, monkeypatch
    ):
        """Verify hybrid mode loads from disk when not in memory."""
        monkeypatch.setattr(server_settings, "kv_cache_mode", "hybrid")
        monkeypatch.setattr(server_settings, "kv_cache_dir", str(tmp_path / "kv"))
        monkeypatch.setattr(server_settings, "max_cached_sessions", 10)

        # Reset global cache
        import infemeral.server as server_module
        server_module._session_kv_cache = None

        # Write directly to disk (simulate previous session)
        save_kv_cache_to_disk(session_id, mock_kv_cache_layers, session_key)

        # Verify disk file exists
        cache_path = get_kv_cache_path(session_id)
        assert cache_path.exists()

        # Load should find it on disk and populate memory
        result = load_kv_cache(session_id, session_key)
        assert result is not None
        assert len(result) == len(mock_kv_cache_layers)

        # Should now be in memory cache too
        cache = get_session_kv_cache()
        assert session_id in cache._cache
