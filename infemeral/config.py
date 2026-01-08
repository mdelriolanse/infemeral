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
        default="./client_weights.safetensors",
        description="Path to client embedding weights",
    )
    server_url: str = Field(
        default="localhost:50051",
        description="gRPC server URL",
    )
    model_id: str = Field(
        default="TheBloke/LLaMA-Pro-8B-AWQ",
        description="Base model ID for tokenizer",
    )


class ServerSettings(BaseSettings):
    """Server-side settings."""

    model_config = SettingsConfigDict(env_prefix="INFEMERAL_SERVER_")

    weights_path: str = Field(
        default="/workspace/server_weights.safetensors",
        description="Path to tensorized server weights",
    )
    kv_cache_dir: str = Field(
        default="/workspace/kv",
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
