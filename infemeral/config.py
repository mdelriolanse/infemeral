"""Configuration settings for Infemeral client and server."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CryptoSettings(BaseSettings):
    """Cryptographic parameters."""

    model_config = SettingsConfigDict(env_prefix="INFEMERAL_CRYPTO_")

    hidden_dim: int = Field(default=4096, description="Model hidden dimension (d)")
    dp_epsilon: float = Field(default=2.0, description="Differential privacy epsilon")
    dp_delta: float = Field(default=1e-5, description="Differential privacy delta")


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


# Singleton instances for easy import
crypto_settings = CryptoSettings()
client_settings = ClientSettings()
server_settings = ServerSettings()
