"""Configuration settings for Infemeral client and server."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Model presets with architecture-specific defaults
MODEL_PRESETS: dict[str, dict] = {
    "llama-3.1-8b-awq": {
        "model_id": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "hidden_dim": 4096,
        "num_layers": 32,
        "architecture": "llama",
    },
    "deepseek-r1-32b-gptq": {
        "model_id": "dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4",
        "hidden_dim": 5120,
        "num_layers": 64,
        "architecture": "qwen",
    },
}


class ClientSettings(BaseSettings):
    """Client-side settings."""

    model_config = SettingsConfigDict(env_prefix="INFEMERAL_CLIENT_")

    weights_path: str = Field(
        default="/workspace/weights/client_weights.safetensors",
        description="Path to client embedding weights",
    )
    server_url: str = Field(
        default="localhost:50051",
        description="gRPC server URL",
    )
    model_id: str = Field(
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        description="Base model ID for tokenizer",
    )


class ServerSettings(BaseSettings):
    """Server-side settings."""

    model_config = SettingsConfigDict(env_prefix="INFEMERAL_SERVER_")

    model_preset: str = Field(
        default="llama-3.1-8b-awq",
        description="Model preset name (llama-3.1-8b-awq, deepseek-r1-32b-gptq)",
    )
    weights_dir: str = Field(
        default="/workspace/weights/model",
        description="Directory containing the full AWQ model (loaded via from_pretrained)",
    )
    tensorized_weights_path: str = Field(
        default="/workspace/weights/model.tensors",
        description="Path to tensorized server model weights (optional, for fast loading)",
    )
    model_id: str = Field(
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        description="HuggingFace model ID for architecture/config",
    )
    kv_cache_dir: str = Field(
        default="/workspace/weights/kv",
        description="Directory for encrypted KV cache storage",
    )
    max_context_length: int = Field(default=2048, description="Maximum context length")
    attention_sink_tokens: int = Field(
        default=4, description="Number of attention sink tokens to preserve"
    )
    grpc_port: int = Field(default=50051, description="gRPC server port")
    kv_cache_mode: str = Field(
        default="memory",
        description="KV cache storage mode: 'memory' (in-GPU), 'disk' (file-based), 'hybrid' (memory with disk fallback)",
    )
    max_cached_sessions: int = Field(
        default=10,
        description="Maximum number of sessions to keep in memory cache (LRU eviction)",
    )
    session_checkpoint_interval: int = Field(
        default=50,
        description="Number of tokens between disk checkpoints (0 to disable checkpointing)",
    )

    @property
    def model_config_preset(self) -> dict:
        """Get preset configuration for the selected model."""
        return MODEL_PRESETS.get(self.model_preset, MODEL_PRESETS["llama-3.1-8b-awq"])


class NVIBSettings(BaseSettings):
    """NVIB (Nonparametric Variational Information Bottleneck) cloaking settings."""

    model_config = SettingsConfigDict(env_prefix="INFEMERAL_NVIB_")

    beta: float = Field(
        default=100.0,
        description="Privacy budget parameter (higher = less noise, better utility)",
    )
    mu_init: float = Field(
        default=0.0,
        description="Initial mean μ for noise distribution",
    )
    log_sigma2_init: float = Field(
        default=0.0,
        description="Initial log variance log σ² for noise distribution",
    )
    prng_seed: int | None = Field(
        default=None,
        description="PRNG seed (None = use system entropy)",
    )
    simd_level: str = Field(
        default="auto",
        description="SIMD level: auto/avx512/avx2/sse4/none",
    )
    dim: int = Field(
        default=0,
        description="Embedding dimension (0 = auto-detect from model hidden_size)",
    )


# Singleton instances for easy import
client_settings = ClientSettings()
server_settings = ServerSettings()
nvib_settings = NVIBSettings()