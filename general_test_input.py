import json
import base64
import torch
from infemeral.crypto import encrypt_bytes
from infemeral.tensors import serialize_tensor

# Generate a random 32-byte session key
session_key = b'\x01' * 32  # Use fixed key for testing

# Create a dummy hidden state tensor (as if from embedding layer)
hidden_states = torch.randn(1, 1, 4096, dtype=torch.float32)

# Serialize the tensor
tensor_bytes, shape, dtype = serialize_tensor(hidden_states)

# Encrypt with AES-GCM
ciphertext, nonce = encrypt_bytes(tensor_bytes, session_key)

# Build test input
test_input = {
"input": {
"cloaked_embedding": base64.b64encode(ciphertext).decode(),
"encrypted_session_key": base64.b64encode(session_key).decode(),
"nonce": base64.b64encode(nonce).decode(),
            "shape": shape,
                  "dtype": dtype,
                            "session_id": "test-session-001"
                                  },
"id": "local_test"
}

with open("test_input.json", "w") as f:
    json.dump(test_input, f, indent=2)

print("Generated test_input.json")
