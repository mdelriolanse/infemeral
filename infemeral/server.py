"""Server-side inference: transformer blocks on encrypted hidden states.

The server processes encrypted hidden states through transformer layers,
never seeing raw tokens or being able to reconstruct user intent.

Uses Tensorizer for fast model loading with AWQ quantization support.
"""

import logging
import signal
import threading
import time
from collections import OrderedDict
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from infemeral.config import server_settings
from infemeral.crypto import decrypt_bytes, encrypt_bytes
from infemeral.tensors import (
    deserialize_tensor,
    pack_kv_cache_v2,
    serialize_tensor,
    unpack_kv_cache,
)
from transformers.cache_utils import DynamicCache

# Import gRPC stubs - handle both installed and development import paths
try:
    from infemeral import tensor_service_pb2, tensor_service_pb2_grpc
except ImportError:
    import tensor_service_pb2
    import tensor_service_pb2_grpc

logger = logging.getLogger(__name__)

# Global model instance (persists across serverless invocations)
_model = None
_config = None


# Forward declarations for disk I/O functions (defined after load_model)
def get_kv_cache_path(session_id: str) -> Path:
    """Get path for session's KV cache."""
    cache_dir = Path(server_settings.kv_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{session_id}.bin"


def load_kv_cache_from_disk(
    session_id: str,
    session_key: bytes,
    device: str | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
    """Load encrypted KV cache from disk (Network Volume).

    Returns None if no cache exists for this session.
    Returns tuple of (key, value) pairs for each layer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_path = get_kv_cache_path(session_id)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            # Read nonce (first 12 bytes)
            nonce = f.read(12)
            ciphertext = f.read()

        # Decrypt
        plaintext = decrypt_bytes(ciphertext, session_key, nonce)

        # Unpack KV tensors (auto-detects v1 or v2 format)
        result = unpack_kv_cache(plaintext, device)

        # If v1 format (flat tensors), return None to force rebuild
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], torch.Tensor):
            logger.warning(f"Session {session_id[:8]} has v1 KV cache, ignoring (will rebuild)")
            return None

        # Validate layer count matches model
        if _config is not None:
            expected_layers = _config.num_hidden_layers
            if len(result) != expected_layers:
                logger.warning(
                    f"KV cache layer count mismatch: expected {expected_layers}, got {len(result)}"
                )
                return None

        return result

    except Exception as e:
        logger.warning(f"Failed to load KV cache from disk: {e}")
        return None


def save_kv_cache_to_disk(
    session_id: str,
    kv_tuples: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    session_key: bytes,
) -> None:
    """Save encrypted KV cache to disk (Network Volume).

    Args:
        session_id: Session identifier
        kv_tuples: Tuple of (key, value) pairs for each layer
        session_key: Encryption key
    """
    cache_path = get_kv_cache_path(session_id)

    # Pack and encrypt
    plaintext = pack_kv_cache_v2(kv_tuples)
    ciphertext, nonce = encrypt_bytes(plaintext, session_key)

    # Write nonce + ciphertext
    with open(cache_path, "wb") as f:
        f.write(nonce)
        f.write(ciphertext)


@dataclass
class CachedSession:
    """In-memory cached session data."""

    kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    session_key: bytes
    last_access: float = field(default_factory=time.time)
    token_count: int = 0
    last_checkpoint_token: int = 0


class SessionKVCache:
    """In-memory LRU cache for KV tensors across sessions.

    Keeps KV cache in GPU memory between token generations to avoid
    disk I/O on every token. Implements LRU eviction when max_sessions
    is reached.
    """

    def __init__(self, max_sessions: int = 10):
        self._cache: OrderedDict[str, CachedSession] = OrderedDict()
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def get(
        self, session_id: str, session_key: bytes, device: str | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
        """Get KV cache for session from memory, falling back to disk.

        Args:
            session_id: Session identifier
            session_key: Encryption key for disk fallback
            device: Target device for tensors

        Returns:
            KV cache tuples or None if not found
        """
        with self._lock:
            if session_id in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(session_id)
                cached = self._cache[session_id]
                cached.last_access = time.time()
                logger.debug(f"Memory cache hit for session {session_id[:8]}")
                return cached.kv_cache

        # Fallback to disk if not in memory
        if server_settings.kv_cache_mode in ("disk", "hybrid"):
            disk_cache = load_kv_cache_from_disk(session_id, session_key, device)
            if disk_cache is not None:
                # Load into memory cache
                self.put(session_id, disk_cache, session_key, token_count=disk_cache[0][0].shape[2])
                logger.debug(f"Loaded session {session_id[:8]} from disk into memory")
                return disk_cache

        return None

    def put(
        self,
        session_id: str,
        kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        session_key: bytes,
        token_count: int = 0,
    ) -> None:
        """Store KV cache for session in memory.

        Args:
            session_id: Session identifier
            kv_cache: KV cache tuples
            session_key: Encryption key (for checkpointing)
            token_count: Current token count in the cache
        """
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_sessions:
                evicted_id, evicted_session = self._cache.popitem(last=False)
                logger.info(f"Evicting session {evicted_id[:8]} from memory cache")
                # Checkpoint to disk before eviction (hybrid mode)
                if server_settings.kv_cache_mode == "hybrid":
                    self._checkpoint_to_disk(evicted_id, evicted_session)

            if session_id in self._cache:
                # Update existing
                self._cache.move_to_end(session_id)
                cached = self._cache[session_id]
                cached.kv_cache = kv_cache
                cached.last_access = time.time()
                cached.token_count = token_count
            else:
                # Insert new
                self._cache[session_id] = CachedSession(
                    kv_cache=kv_cache,
                    session_key=session_key,
                    token_count=token_count,
                )

    def should_checkpoint(self, session_id: str) -> bool:
        """Check if session should be checkpointed to disk.

        Returns True if token_count has grown by checkpoint_interval since last checkpoint.
        """
        interval = server_settings.session_checkpoint_interval
        if interval <= 0:
            return False

        with self._lock:
            if session_id not in self._cache:
                return False
            cached = self._cache[session_id]
            return (cached.token_count - cached.last_checkpoint_token) >= interval

    def checkpoint(self, session_id: str) -> None:
        """Checkpoint session to disk.

        Args:
            session_id: Session identifier
        """
        with self._lock:
            if session_id not in self._cache:
                return
            cached = self._cache[session_id]
            self._checkpoint_to_disk(session_id, cached)
            cached.last_checkpoint_token = cached.token_count

    def _checkpoint_to_disk(self, session_id: str, cached: CachedSession) -> None:
        """Write session KV cache to disk (called within lock)."""
        try:
            save_kv_cache_to_disk(session_id, cached.kv_cache, cached.session_key)
            logger.debug(f"Checkpointed session {session_id[:8]} to disk")
        except Exception as e:
            logger.warning(f"Failed to checkpoint session {session_id[:8]}: {e}")

    def delete(self, session_id: str) -> None:
        """Remove session from memory cache."""
        with self._lock:
            if session_id in self._cache:
                del self._cache[session_id]
                logger.debug(f"Deleted session {session_id[:8]} from memory cache")

    def persist_session(self, session_id: str) -> None:
        """Explicitly persist session to disk."""
        with self._lock:
            if session_id in self._cache:
                self._checkpoint_to_disk(session_id, self._cache[session_id])

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Remove sessions older than max_age from memory.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of sessions cleaned up
        """
        now = time.time()
        to_remove = []

        with self._lock:
            for session_id, cached in self._cache.items():
                if now - cached.last_access > max_age_seconds:
                    to_remove.append(session_id)
                    # Checkpoint before removal in hybrid mode
                    if server_settings.kv_cache_mode == "hybrid":
                        self._checkpoint_to_disk(session_id, cached)

            for session_id in to_remove:
                del self._cache[session_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} expired sessions from memory")

        return len(to_remove)

    def __len__(self) -> int:
        """Return number of cached sessions."""
        return len(self._cache)


# Global in-memory KV cache (shared across requests)
_session_kv_cache: SessionKVCache | None = None


def get_session_kv_cache() -> SessionKVCache:
    """Get or create the global session KV cache."""
    global _session_kv_cache
    if _session_kv_cache is None:
        _session_kv_cache = SessionKVCache(max_sessions=server_settings.max_cached_sessions)
    return _session_kv_cache


def load_model(device: str | None = None) -> torch.nn.Module:
    """Load transformer model using Tensorizer (preferred) or from_pretrained.

    Priority:
    1. Tensorizer from server_settings.tensorized_weights_path - fastest loading
    2. AutoModelForCausalLM.from_pretrained() from server_settings.weights_dir

    Args:
        device: Device to load model on
    """
    global _model, _config

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _model is not None:
        return _model

    tensorized_path = Path(server_settings.tensorized_weights_path)
    weights_dir = Path(server_settings.weights_dir)

    print(f"Tensorized path: {tensorized_path}")
    print(f"Weights directory: {weights_dir}")

    # Try Tensorizer first (preferred for fast cold starts)
    if tensorized_path.exists():
        try:
            from tensorizer import TensorDeserializer

            print(f"Using Tensorizer for fast model loading from {tensorized_path}...")
            deserializer = TensorDeserializer(str(tensorized_path), device=device)

            # Load config
            try:
                _config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
            except Exception:
                _config = AutoConfig.from_pretrained(
                    server_settings.model_id, trust_remote_code=True
                )

            _model = AutoModelForCausalLM.from_config(_config, torch_dtype=torch.float16)
            deserializer.load_into_module(_model)
            deserializer.close()
            print("Model loaded successfully with Tensorizer")

        except ImportError:
            print("Tensorizer not installed, falling back to from_pretrained")
        except Exception as e:
            print(f"Tensorizer loading failed: {e}, falling back to from_pretrained")

    # Fallback: load directly from weights directory
    if _model is None:
        if not weights_dir.exists():
            raise FileNotFoundError(
                f"Model weights directory not found: {weights_dir}\n"
                f"Run 'python -m infemeral.model_prep' to download and prepare the model."
            )

        print(f"Loading model from {weights_dir} using from_pretrained...")

        # Load config
        _config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)

        # Load model (handles AWQ automatically)
        # Use eager attention to avoid SDPA contiguity issues with cached KV tensors
        _model = AutoModelForCausalLM.from_pretrained(
            weights_dir,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        print("Model loaded successfully")

    # Only move to device if not already placed by device_map
    if not hasattr(_model, 'hf_device_map'):
        _model = _model.to(device)
    _model.eval()
    print(f"Model loaded. Hidden size: {_config.hidden_size}")

    return _model


def load_kv_cache(
    session_id: str,
    session_key: bytes,
    device: str | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
    """Load KV cache for session, using in-memory cache when available.

    Behavior depends on kv_cache_mode setting:
    - 'memory': Check memory only, no disk fallback
    - 'disk': Load directly from disk (legacy behavior)
    - 'hybrid': Check memory first, fall back to disk

    Returns None if no cache exists for this session.
    Returns tuple of (key, value) pairs for each layer.
    """
    mode = server_settings.kv_cache_mode

    if mode == "disk":
        # Legacy behavior: direct disk I/O
        return load_kv_cache_from_disk(session_id, session_key, device)

    # Memory or hybrid mode: use in-memory cache
    cache = get_session_kv_cache()
    return cache.get(session_id, session_key, device)


def save_kv_cache(
    session_id: str,
    kv_tuples: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    session_key: bytes,
) -> None:
    """Save KV cache for session, using in-memory cache when available.

    Behavior depends on kv_cache_mode setting:
    - 'memory': Store in memory only, no disk write
    - 'disk': Write directly to disk (legacy behavior)
    - 'hybrid': Store in memory, checkpoint to disk periodically

    Args:
        session_id: Session identifier
        kv_tuples: Tuple of (key, value) pairs for each layer
        session_key: Encryption key
    """
    mode = server_settings.kv_cache_mode

    if mode == "disk":
        # Legacy behavior: direct disk I/O
        save_kv_cache_to_disk(session_id, kv_tuples, session_key)
        return

    # Memory or hybrid mode: use in-memory cache
    cache = get_session_kv_cache()
    token_count = kv_tuples[0][0].shape[2] if kv_tuples else 0
    cache.put(session_id, kv_tuples, session_key, token_count)

    # Periodic disk checkpoint in hybrid mode
    if mode == "hybrid" and cache.should_checkpoint(session_id):
        cache.checkpoint(session_id)


def delete_kv_cache(session_id: str) -> None:
    """Delete KV cache for a session from both memory and disk."""
    # Remove from memory cache
    if server_settings.kv_cache_mode != "disk":
        cache = get_session_kv_cache()
        cache.delete(session_id)

    # Remove from disk
    cache_path = get_kv_cache_path(session_id)
    if cache_path.exists():
        cache_path.unlink()


def persist_session(session_id: str) -> None:
    """Explicitly persist a session's KV cache to disk.

    Useful for checkpointing before expected server restarts or
    for important sessions that should survive memory eviction.

    Args:
        session_id: Session identifier
    """
    if server_settings.kv_cache_mode != "disk":
        cache = get_session_kv_cache()
        cache.persist_session(session_id)


def cleanup_old_sessions(max_age_seconds: int = 3600) -> int:
    """Delete KV cache files older than max_age.

    Cleans both in-memory cache and disk files.

    Args:
        max_age_seconds: Maximum age in seconds (default: 1 hour)

    Returns:
        Number of sessions cleaned up
    """
    cleaned = 0

    # Cleanup memory cache
    if server_settings.kv_cache_mode != "disk":
        cache = get_session_kv_cache()
        cleaned += cache.cleanup_expired(max_age_seconds)

    # Cleanup disk cache
    kv_dir = Path(server_settings.kv_cache_dir)
    if kv_dir.exists():
        now = time.time()
        for cache_file in kv_dir.glob("*.bin"):
            try:
                if now - cache_file.stat().st_mtime > max_age_seconds:
                    cache_file.unlink()
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to clean {cache_file.name}: {e}")

    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} old KV cache sessions")

    return cleaned


def _detect_actual_head_dim(
    attention_layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    config: Any | None = None,
) -> int:
    """Get head_dim from layer attributes or config.

    For AWQ models, we don't adjust head_dim - the query tensor shape
    matches the reported head_dim. The issue is elsewhere.

    Args:
        attention_layer: The attention layer (e.g., layers[0].self_attn)
        hidden_states: Sample hidden states tensor [batch, seq_len, hidden_dim]
        config: Model config (optional, for getting num_heads)

    Returns:
        Head_dim from layer attributes or config
    """
    # Get reported head_dim
    reported_head_dim = getattr(attention_layer, "head_dim", None)
    if reported_head_dim is None and config is not None:
        reported_head_dim = config.hidden_size // getattr(
            config, "num_attention_heads", 32
        )

    if reported_head_dim is not None:
        return reported_head_dim

    # Last resort fallback
    return 128


def _create_rotary_embeddings_with_correct_dim(
    rotary_emb: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create rotary embeddings with correct head_dim.

    Args:
        rotary_emb: Rotary embedding module
        hidden_states: Hidden states tensor
        position_ids: Position IDs tensor
        head_dim: The head dimension (from layer attributes/config)

    Returns:
        Tuple of (cos, sin) tensors with correct dimensions
    """
    # Compute rotary embeddings and return as-is
    cos, sin = rotary_emb(hidden_states, position_ids)
    return cos, sin


def apply_context_windowing(
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    max_length: int,
    sink_tokens: int = 4,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Apply attention sink + sliding window to KV cache.

    Strategy: Keep first `sink_tokens` and last `max_length - sink_tokens` tokens.

    Args:
        past_key_values: Per-layer KV cache
        max_length: Maximum context length
        sink_tokens: Number of initial tokens to always preserve

    Returns:
        Windowed KV cache
    """
    if not past_key_values:
        return past_key_values

    # Check current length (from first layer's key tensor)
    current_len = past_key_values[0][0].shape[2]

    if current_len <= max_length:
        return past_key_values

    # Apply windowing to each layer
    windowed = []
    keep_recent = max_length - sink_tokens

    for keys, values in past_key_values:
        # Keep: [0:sink_tokens] + [-(keep_recent):]
        windowed_keys = torch.cat([keys[:, :, :sink_tokens, :], keys[:, :, -keep_recent:, :]], dim=2)
        windowed_values = torch.cat(
            [values[:, :, :sink_tokens, :], values[:, :, -keep_recent:, :]], dim=2
        )
        windowed.append((windowed_keys, windowed_values))

    logger.warning(
        f"Applied context windowing: {current_len} -> {max_length} "
        f"(kept {sink_tokens} sink + {keep_recent} recent tokens)"
    )

    return tuple(windowed)


def forward_transformer(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    past_key_values: tuple | None = None,
) -> tuple[torch.Tensor, tuple]:
    """Forward pass through transformer layers only.

    This bypasses the embedding layer and directly feeds
    hidden states into the transformer blocks.

    Handles both standard and AWQ quantized models by computing
    position embeddings per-layer using each layer's rotary_emb module.
    """
    # Get the transformer layers
    if hasattr(model, "model"):
        # Llama-style architecture
        transformer = model.model
        layers = transformer.layers
        norm = transformer.norm
    else:
        raise ValueError("Unsupported model architecture")

    # Convert hidden_states to model's expected dtype (float16 for quantized models)
    # Client sends float32 for precision, but model weights are float16
    model_dtype = next(model.parameters()).dtype
    if hidden_states.dtype != model_dtype:
        hidden_states = hidden_states.to(model_dtype)

    # Apply context windowing if needed
    if past_key_values is not None:
        past_key_values = apply_context_windowing(
            past_key_values,
            server_settings.max_context_length,
            server_settings.attention_sink_tokens,
        )

    # Prepare attention mask and position IDs
    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Always initialize DynamicCache - layers require a cache object to return updated cache
    # When past_key_values=None is passed, layers return Tensor instead of (Tensor, Cache) tuple
    cache = DynamicCache()
    past_len = 0
    if past_key_values is not None:
        # Directly populate cache lists instead of using update() to avoid torch.cat() non-contiguity
        # DynamicCache.update() concatenates tensors, creating non-contiguous views
        for layer_idx, (k, v) in enumerate(past_key_values):
            # Ensure tensors are contiguous for SDPA attention (only if needed)
            k_contig = k if k.is_contiguous() else k.contiguous()
            v_contig = v if v.is_contiguous() else v.contiguous()
            # Directly append to cache lists - avoids torch.cat() in update()
            cache.key_cache.append(k_contig)
            cache.value_cache.append(v_contig)
        past_len = past_key_values[0][0].shape[2]

    # Position IDs
    if past_len > 0:
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=device
        ).unsqueeze(0)
    else:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # Create 4D causal attention mask for eager attention
    # Shape: [batch_size, 1, seq_len, total_len] where total_len = past_len + seq_len
    total_len = past_len + seq_len
    # Create causal mask: each position can only attend to positions <= its own
    causal_mask = torch.triu(
        torch.ones(seq_len, total_len, dtype=dtype, device=device) * float("-inf"),
        diagonal=past_len + 1,  # Allow attending to past + current position
    )
    # Expand to [batch_size, 1, seq_len, total_len]
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    for i, layer in enumerate(layers):
        layer_attn = layer.self_attn

        # Get rotary embedding module (try layer-specific first, then global)
        layer_rotary_emb = getattr(layer_attn, "rotary_emb", None)
        if layer_rotary_emb is None:
            layer_rotary_emb = getattr(transformer, "rotary_emb", None)

        if layer_rotary_emb is None:
            raise ValueError(f"Layer {i} has no rotary_emb - cannot compute position_embeddings")

        # Compute rotary embeddings
        cos, sin = layer_rotary_emb(hidden_states, position_ids)
        layer_position_embeddings = (cos, sin)

        layer_out = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,  # singular - transformers 4.51.x API
            use_cache=True,
            position_embeddings=layer_position_embeddings,
        )

        # Handle both tensor and tuple returns
        if isinstance(layer_out, torch.Tensor):
            # Layer returned tensor directly (no cache)
            hidden_states = layer_out
            layer_cache = None
        else:
            # Layer returned tuple (hidden_states, cache, ...)
            hidden_states = layer_out[0]
            layer_cache = layer_out[1] if len(layer_out) > 1 else None

        # Ensure hidden_states maintains 3D shape [batch, seq_len, hidden_dim]
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)  # Add batch dim, not seq dim

        # Update cache reference (layer may return updated cache or modify in-place)
        if layer_cache is not None:
            cache = layer_cache

    # Final layer norm
    hidden_states = norm(hidden_states)

    # Extract KV tuples from DynamicCache for storage (supports both 4.51 and 4.57+ API)
    new_key_values = []
    if hasattr(cache, 'key_cache') and len(cache.key_cache) > 0:
        # Transformers 4.51.x API: key_cache/value_cache lists
        for layer_idx in range(len(cache.key_cache)):
            k = cache.key_cache[layer_idx]
            v = cache.value_cache[layer_idx]
            new_key_values.append((k, v))
    elif hasattr(cache, 'layers') and len(cache.layers) > 0:
        # Transformers 4.57+ API: layers list with CacheLayer objects
        for layer_idx, layer_cache in enumerate(cache.layers):
            k = layer_cache.keys
            v = layer_cache.values
            new_key_values.append((k, v))

    return hidden_states, tuple(new_key_values) if new_key_values else ()


class TensorInferenceServicer(tensor_service_pb2_grpc.TensorInferenceServicer):
    """gRPC servicer for tensor inference requests."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device

    def Infer(  # noqa: N802 (gRPC method name convention)
        self, request: tensor_service_pb2.InferenceRequest, context: grpc.ServicerContext
    ) -> tensor_service_pb2.InferenceResponse:
        """Process a single inference request.

        Args:
            request: InferenceRequest with encrypted hidden states and session data
            context: gRPC context for setting status codes

        Returns:
            InferenceResponse with encrypted output tensor
        """
        try:
            # Extract fields from request
            session_key = bytes(request.encrypted_session_key)
            nonce = bytes(request.nonce)
            encrypted_data = bytes(request.cloaked_embedding)
            shape = list(request.shape)
            dtype = request.dtype
            session_id = request.session_id

            logger.debug(f"Processing inference request for session {session_id[:8]}...")

            # Decrypt hidden states
            plaintext = decrypt_bytes(encrypted_data, session_key, nonce)

            # Deserialize tensor
            hidden = deserialize_tensor(plaintext, shape, dtype, device=self.device)

            # Load KV cache if exists
            past_key_values = load_kv_cache(session_id, session_key, self.device)

            # Forward through transformer
            with torch.no_grad():
                output, new_kv = forward_transformer(self.model, hidden, past_key_values)

            # Save updated KV cache
            if new_kv:
                save_kv_cache(session_id, new_kv, session_key)

            # Serialize output
            output_data, output_shape, output_dtype = serialize_tensor(output)

            # Encrypt output
            encrypted_output, output_nonce = encrypt_bytes(output_data, session_key)

            # Memory wipe
            del hidden, output, new_kv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.debug(f"Inference complete for session {session_id[:8]}")

            # Build response with nonce prepended to output (client expects this format)
            return tensor_service_pb2.InferenceResponse(
                output=output_nonce + encrypted_output,
                shape=output_shape,
                dtype=output_dtype,
                tokens_processed=shape[1],
            )

        except Exception as e:
            error_msg = str(e) or type(e).__name__
            logger.exception(f"Inference failed: {error_msg}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return tensor_service_pb2.InferenceResponse(error=error_msg)


def serve_grpc(port: int | None = None, max_workers: int = 4) -> None:
    """Start the gRPC inference server.

    Args:
        port: Port to bind to (defaults to server_settings.grpc_port)
        max_workers: Maximum number of worker threads
    """
    if port is None:
        port = server_settings.grpc_port

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Cleanup old sessions on startup
    cleaned = cleanup_old_sessions()
    if cleaned > 0:
        logger.info(f"Cleaned {cleaned} old session(s) on startup")

    logger.info("Loading model...")
    model = load_model()
    device = next(model.parameters()).device
    logger.info(f"Model loaded on {device}")

    # Create gRPC server with large message options (100MB)
    options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)

    # Register servicer
    servicer = TensorInferenceServicer(model, device=str(device))
    tensor_service_pb2_grpc.add_TensorInferenceServicer_to_server(servicer, server)

    # Bind to port
    server.add_insecure_port(f"[::]:{port}")

    # Graceful shutdown handling
    shutdown_event = futures.Future()

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set_result(True)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Start server
    server.start()
    logger.info(f"gRPC server started on port {port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        server.stop(grace=5)


def handler(event: dict) -> dict:
    """RunPod serverless handler.

    Receives encrypted hidden states, runs transformer inference,
    returns transformed hidden states.

    Input event:
        {
            "input": {
                "cloaked_embedding": bytes (base64),
                "encrypted_session_key": bytes,
                "nonce": bytes,
                "shape": [batch, seq_len, hidden_dim],
                "dtype": str,
                "session_id": str,
            }
        }

    Output:
        {
            "output": bytes (base64),
            "shape": [batch, seq_len, hidden_dim],
            "dtype": str,
        }
    """
    try:
        inp = event.get("input", event)

        # Load model on first request (cold start)
        # load_model will use server_settings.tensorized_weights_path first,
        # then fall back to server_settings.weights_path if needed
        model = load_model()
        device = next(model.parameters()).device

        # Decrypt input tensor
        session_key = inp["encrypted_session_key"]
        if isinstance(session_key, str):
            import base64
            session_key = base64.b64decode(session_key)

        nonce = inp["nonce"]
        if isinstance(nonce, str):
            import base64
            nonce = base64.b64decode(nonce)

        encrypted_data = inp["cloaked_embedding"]
        if isinstance(encrypted_data, str):
            import base64
            encrypted_data = base64.b64decode(encrypted_data)

        plaintext = decrypt_bytes(encrypted_data, session_key, nonce)

        # Deserialize tensor
        hidden = deserialize_tensor(
            plaintext,
            inp["shape"],
            inp["dtype"],
            device=str(device),
        )

        session_id = inp["session_id"]

        # Load KV cache if exists
        past_key_values = load_kv_cache(session_id, session_key, str(device))

        # Forward through transformer
        with torch.no_grad():
            output, new_kv = forward_transformer(model, hidden, past_key_values)

        # Save updated KV cache
        if new_kv:
            save_kv_cache(session_id, new_kv, session_key)

        # Serialize output
        output_data, output_shape, output_dtype = serialize_tensor(output)

        # Encrypt output
        encrypted_output, output_nonce = encrypt_bytes(output_data, session_key)

        # Memory wipe
        del hidden, output, new_kv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import base64
        return {
            "output": base64.b64encode(encrypted_output).decode(),
            "nonce": base64.b64encode(output_nonce).decode(),
            "shape": output_shape,
            "dtype": output_dtype,
            "tokens_processed": inp["shape"][1],
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Entry point: supports both gRPC server and RunPod serverless modes
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Infemeral inference server")
    parser.add_argument(
        "--mode",
        choices=["grpc", "runpod"],
        default="grpc",
        help="Server mode: 'grpc' for standalone gRPC server, 'runpod' for serverless",
    )
    parser.add_argument("--port", type=int, default=None, help="gRPC port (default: 50051)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")

    args = parser.parse_args()

    if args.mode == "grpc":
        serve_grpc(port=args.port, max_workers=args.workers)
    else:
        import runpod

        runpod.serverless.start({"handler": handler})
