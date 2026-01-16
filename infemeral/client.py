"""Client-side inference: embedding, cloaking, and de-embedding.

The client holds the semantic entry and exit points (embed_tokens + lm_head),
ensuring the server never sees raw tokens or human-readable outputs.
"""

import secrets
from pathlib import Path

import grpc
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from infemeral.config import client_settings, crypto_settings
from infemeral.crypto import (
    CloakingContext,
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
        state_dict = load_file(weights_path)

        # Get dimensions from weights
        embed_weight = state_dict["embed_tokens.weight"]
        vocab_size, hidden_size = embed_weight.shape

        # Create layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_tokens.weight.data = embed_weight

        if "lm_head.weight" in state_dict:
            lm_head_weight = state_dict["lm_head.weight"]
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.lm_head.weight.data = lm_head_weight
        else:
            # Tied embeddings: lm_head shares weights with embed_tokens
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

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
        self.cloaking_ctx = create_cloaking_context(
            seed=secrets.randbelow(2**31))

        # gRPC channel (lazy init)
        self._channel: grpc.Channel | None = None
        self._stub = None
        self.server_url = server_url

    @property
    def stub(self):
        """Lazy-initialize gRPC stub."""
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
                ],
            )
            self._stub = tensor_service_pb2_grpc.TensorInferenceStub(
                self._channel)
        return self._stub

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
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold

        Returns:
            Generated text including the prompt
        """
        # Tokenize (client-only operation)
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Embed (client-only)
            hidden = self.embedding.embed(generated_ids)

            # Cloak (orthogonal rotation + DP noise)
            cloaked = cloak(hidden, self.cloaking_ctx)

            # Send to server
            server_output = self._call_server(cloaked)

            # Uncloak
            uncloaked = uncloak(server_output, self.cloaking_ctx)

            # De-embed to logits (client-only)
            logits = self.embedding.de_embed(uncloaked[:, -1:, :])

            # Sample next token
            next_token = self._sample(logits[:, -1, :], temperature, top_p)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Append to sequence
            generated_ids = torch.cat(
                [generated_ids, next_token.unsqueeze(0)], dim=1)

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

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


def main():
    """Simple CLI for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Infemeral client")
    parser.add_argument(
        "--weights", default="./weights/client_weights.safetensors")
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=50)

    args = parser.parse_args()

    client = Client(
        weights_path=args.weights,
        server_url=args.server,
    )

    try:
        output = client.generate(args.prompt, max_new_tokens=args.max_tokens)
        print(output)
    finally:
        client.close()


if __name__ == "__main__":
    main()
