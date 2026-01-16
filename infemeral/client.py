"""Client-side inference: embedding, cloaking, and de-embedding.

The client holds the semantic entry and exit points (embed_tokens + lm_head),
ensuring the server never sees raw tokens or human-readable outputs.
"""

import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

import grpc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from infemeral.config import client_settings, crypto_settings


@dataclass
class TokenTiming:
    """Timing breakdown for a single token generation."""

    embed_ms: float = 0.0
    cloak_ms: float = 0.0
    network_ms: float = 0.0
    uncloak_ms: float = 0.0
    de_embed_ms: float = 0.0
    sample_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class GenerationMetrics:
    """Performance metrics for a generation call."""

    timings: list[TokenTiming] = field(default_factory=list)
    device: str = "cpu"
    peak_memory_mb: float = 0.0
    tokens_per_sec: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0


from infemeral.crypto import (
    cloak,
    create_cloaking_context,
    encrypt_bytes,
    generate_session_key,
    uncloak,
)
from infemeral.tensors import deserialize_tensor, serialize_tensor

# Import generated protobuf (will be generated from tensor_service.proto)
try:
    from infemeral import tensor_service_pb2, tensor_service_pb2_grpc
except ImportError:
    tensor_service_pb2 = None
    tensor_service_pb2_grpc = None


class EmbeddingLayer(nn.Module):
    """Client-side embedding and de-embedding layers."""

    def __init__(self, weights_path: str, device: str = "cuda"):
        super().__init__()
        from safetensors import safe_open

        # Load weights and check for tied embeddings metadata
        with safe_open(weights_path, framework="pt") as f:
            metadata = f.metadata() or {}
            embed_weight = f.get_tensor("embed_tokens.weight")
            has_lm_head = "lm_head.weight" in f.keys()
            lm_head_weight = f.get_tensor("lm_head.weight") if has_lm_head else None

        vocab_size, hidden_size = embed_weight.shape
        self.tied_embeddings = not has_lm_head or metadata.get("tied_embeddings") == "true"

        # Create embedding layer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_tokens.weight.data = embed_weight

        # Create lm_head layer
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if self.tied_embeddings:
            # Share weights between embed_tokens and lm_head
            self.lm_head.weight = self.embed_tokens.weight
            memory_saved_mb = vocab_size * hidden_size * 2 / (1024 * 1024)
            print(f"Using tied embeddings ({memory_saved_mb:.0f} MB saved)")
        else:
            self.lm_head.weight.data = lm_head_weight

        self.to(device)
        self.eval()

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to hidden states."""
        with torch.no_grad():
            return self.embed_tokens(input_ids)

    def de_embed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convert hidden states to logits."""
        with torch.no_grad():
            return self.lm_head(hidden_states)


class Client:
    """Zero-trust inference client.

    Security invariants:
    1. Raw tokens never leave the client
    2. Raw embeddings never leave the client
    3. Server only sees cloaked (rotated + noised) hidden states
    """

    def __init__(
        self,
        weights_path: str | None = None,
        server_url: str | None = None,
        tokenizer_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        weights_path = weights_path or client_settings.weights_path
        server_url = server_url or client_settings.server_url

        # Load tokenizer
        if tokenizer_path and Path(tokenizer_path).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                client_settings.model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load embedding layers
        self.embedding = EmbeddingLayer(weights_path, device)
        self.device = device

        # Session state
        self.session_id = secrets.token_hex(16)
        self.session_key = generate_session_key()
        # Create cloaking context on target device with float16 for memory efficiency
        self.cloaking_ctx = create_cloaking_context(
            seed=secrets.randbelow(2**31),
            device=device,
            dtype=torch.float16,
        )

        # gRPC channel (lazy init)
        self._channel: grpc.Channel | None = None
        self._stub = None
        self.server_url = server_url

    @property
    def stub(self):
        """Lazy-initialize gRPC stub with keepalive."""
        if self._stub is None:
            if tensor_service_pb2_grpc is None:
                raise ImportError(
                    "gRPC stubs not generated. Run: "
                    "python -m grpc_tools.protoc -I. --python_out=infemeral "
                    "--grpc_python_out=infemeral tensor_service.proto"
                )
            self._channel = grpc.insecure_channel(
                self.server_url,
                options=[
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", 1),
                    ("grpc.http2.min_time_between_pings_ms", 10000),
                    ("grpc.http2.max_pings_without_data", 0),
                ],
            )
            self._stub = tensor_service_pb2_grpc.TensorInferenceStub(
                self._channel)
        return self._stub

    def check_channel_health(self) -> bool:
        """Check if the gRPC channel is connected and healthy."""
        if self._channel is None:
            return False
        try:
            state = self._channel._channel.check_connectivity_state(True)
            return state in (
                grpc.ChannelConnectivity.READY,
                grpc.ChannelConnectivity.IDLE,
            )
        except Exception:
            return False

    def reconnect(self) -> None:
        """Force reconnection of the gRPC channel."""
        self.close()
        _ = self.stub

    def _call_server(self, cloaked: torch.Tensor) -> torch.Tensor:
        """Send cloaked embedding to server, receive transformed output."""
        # Serialize tensor
        data, shape, dtype = serialize_tensor(cloaked)

        # Encrypt with session key
        encrypted_data, nonce = encrypt_bytes(data, self.session_key)

        # Build request
        request = tensor_service_pb2.InferenceRequest(
            cloaked_embedding=encrypted_data,
            encrypted_session_key=self.session_key,  # TODO: RSA-wrap this
            nonce=nonce,
            shape=shape,
            dtype=dtype,
            session_id=self.session_id,
            max_new_tokens=1,
            temperature=0.7,
        )

        # Call server
        response = self.stub.Infer(request)

        if response.error:
            raise RuntimeError(f"Server error: {response.error}")

        # Deserialize response
        return deserialize_tensor(
            response.output,
            list(response.shape),
            response.dtype,
            device=self.device,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_metrics: bool = False,
    ) -> str | tuple[str, GenerationMetrics]:
        """Generate text from a prompt.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            return_metrics: If True, return (text, metrics) tuple

        Returns:
            Generated text, or (text, metrics) if return_metrics=True
        """
        metrics = GenerationMetrics(device=self.device) if return_metrics else None

        if return_metrics and torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        generation_start = time.perf_counter()

        # Tokenize (client-only operation)
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()

        if metrics:
            metrics.prompt_tokens = input_ids.shape[1]

        # PROMPT PHASE: Send full sequence for initial KV cache build
        timing = self._generate_token(
            generated_ids, temperature, top_p, return_timing=return_metrics
        )
        if return_metrics and timing:
            metrics.timings.append(timing[1])
            hidden, cloaked, server_output, uncloaked, logits, next_token = (
                timing[0]["hidden"],
                timing[0]["cloaked"],
                timing[0]["server_output"],
                timing[0]["uncloaked"],
                timing[0]["logits"],
                timing[0]["next_token"],
            )
        else:
            hidden = self.embedding.embed(generated_ids)
            cloaked = cloak(hidden, self.cloaking_ctx)
            server_output = self._call_server(cloaked)
            uncloaked = uncloak(server_output, self.cloaking_ctx)
            logits = self.embedding.de_embed(uncloaked[:, -1:, :])
            next_token = self._sample(logits[:, -1, :], temperature, top_p)

        if next_token.item() == self.tokenizer.eos_token_id:
            result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if return_metrics:
                self._finalize_metrics(metrics, generation_start, 1)
                return result, metrics
            return result

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

        # GENERATION PHASE: Send only new token for subsequent iterations
        for _ in range(max_new_tokens - 1):
            last_token = generated_ids[:, -1:]

            if return_metrics:
                timing = self._generate_token(
                    last_token, temperature, top_p, return_timing=True
                )
                metrics.timings.append(timing[1])
                next_token = timing[0]["next_token"]
            else:
                hidden = self.embedding.embed(last_token)
                cloaked = cloak(hidden, self.cloaking_ctx)
                server_output = self._call_server(cloaked)
                uncloaked = uncloak(server_output, self.cloaking_ctx)
                logits = self.embedding.de_embed(uncloaked[:, -1:, :])
                next_token = self._sample(logits[:, -1, :], temperature, top_p)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated_ids = torch.cat(
                [generated_ids, next_token.unsqueeze(0)], dim=1)

        result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if return_metrics:
            total_generated = generated_ids.shape[1] - metrics.prompt_tokens
            self._finalize_metrics(metrics, generation_start, total_generated)
            return result, metrics

        return result

    def _generate_token(
        self,
        input_ids: torch.Tensor,
        temperature: float,
        top_p: float,
        return_timing: bool = False,
    ) -> tuple[dict, TokenTiming] | None:
        """Generate a single token with optional timing instrumentation."""
        if not return_timing:
            return None

        timing = TokenTiming()
        total_start = time.perf_counter()

        # Embed
        t0 = time.perf_counter()
        hidden = self.embedding.embed(input_ids)
        timing.embed_ms = (time.perf_counter() - t0) * 1000

        # Cloak
        t0 = time.perf_counter()
        cloaked = cloak(hidden, self.cloaking_ctx)
        timing.cloak_ms = (time.perf_counter() - t0) * 1000

        # Network (includes serialize + encrypt + RPC + decrypt + deserialize)
        t0 = time.perf_counter()
        server_output = self._call_server(cloaked)
        timing.network_ms = (time.perf_counter() - t0) * 1000

        # Uncloak
        t0 = time.perf_counter()
        uncloaked = uncloak(server_output, self.cloaking_ctx)
        timing.uncloak_ms = (time.perf_counter() - t0) * 1000

        # De-embed
        t0 = time.perf_counter()
        logits = self.embedding.de_embed(uncloaked[:, -1:, :])
        timing.de_embed_ms = (time.perf_counter() - t0) * 1000

        # Sample
        t0 = time.perf_counter()
        next_token = self._sample(logits[:, -1, :], temperature, top_p)
        timing.sample_ms = (time.perf_counter() - t0) * 1000

        timing.total_ms = (time.perf_counter() - total_start) * 1000

        intermediates = {
            "hidden": hidden,
            "cloaked": cloaked,
            "server_output": server_output,
            "uncloaked": uncloaked,
            "logits": logits,
            "next_token": next_token,
        }

        return intermediates, timing

    def _finalize_metrics(
        self,
        metrics: GenerationMetrics,
        start_time: float,
        total_tokens: int,
    ) -> None:
        """Finalize generation metrics with computed values."""
        total_time = time.perf_counter() - start_time
        metrics.total_tokens = total_tokens
        metrics.tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0

        if torch.cuda.is_available() and self.device == "cuda":
            metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Sample next token from logits using nucleus sampling."""
        if temperature == 0:
            return logits.argmax(dim=-1)

        # Apply temperature
        logits = logits / temperature

        # Top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def close(self):
        """Close gRPC channel."""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None


def print_metrics(metrics: GenerationMetrics) -> None:
    """Print timing breakdown and performance metrics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Device: {metrics.device}")
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Prompt tokens: {metrics.prompt_tokens}")
    print(f"Tokens/sec: {metrics.tokens_per_sec:.2f}")
    if metrics.peak_memory_mb > 0:
        print(f"Peak GPU memory: {metrics.peak_memory_mb:.1f} MB")

    if metrics.timings:
        print("\n--- Per-Token Timing (ms) ---")
        print(f"{'Phase':<12} {'Min':>8} {'Median':>8} {'Max':>8} {'Total':>10}")
        print("-" * 48)

        phases = ["embed", "cloak", "network", "uncloak", "de_embed", "sample", "total"]
        for phase in phases:
            values = [getattr(t, f"{phase}_ms") for t in metrics.timings]
            sorted_vals = sorted(values)
            min_v = sorted_vals[0]
            max_v = sorted_vals[-1]
            median_v = sorted_vals[len(sorted_vals) // 2]
            total_v = sum(values)
            print(f"{phase:<12} {min_v:>8.2f} {median_v:>8.2f} {max_v:>8.2f} {total_v:>10.2f}")

    print("=" * 60)


def main():
    """Simple CLI for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Infemeral client")
    parser.add_argument(
        "--weights", default="./weights/client_weights.safetensors")
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument(
        "--profile", action="store_true", help="Print timing breakdown")

    args = parser.parse_args()

    client = Client(
        weights_path=args.weights,
        server_url=args.server,
    )

    try:
        if args.profile:
            output, metrics = client.generate(
                args.prompt, max_new_tokens=args.max_tokens, return_metrics=True
            )
            print(output)
            print_metrics(metrics)
        else:
            output = client.generate(args.prompt, max_new_tokens=args.max_tokens)
            print(output)
    finally:
        client.close()


if __name__ == "__main__":
    main()
