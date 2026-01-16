# Infemeral Backend System Rundown

## Executive Summary

Infemeral is a **zero-trust distributed LLM inference system** that splits model execution between client and server to ensure privacy. The server never sees raw tokens, embeddings, or conversation context. The system supports two deployment modes: **traditional gRPC** (planned) and **RunPod serverless** (HTTP-based, currently implemented).

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT (Trusted Domain)                  │
│                                                             │
│  Text → Tokenizer → Embeddings → Cloaking → Encryption    │
│                                                             │
│  • embed_tokens (vocab → hidden_dim)                       │
│  • lm_head (hidden_dim → vocab)                            │
│  • Orthogonal rotation matrix M (4096×4096)               │
│  • DP noise injection (ε=2.0, δ=1e-5)                      │
│  • AES-256-GCM encryption                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ gRPC (port 50051) or HTTP
                          │ Encrypted cloaked tensors
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              SERVER (Untrusted Domain)                     │
│                                                             │
│  Decryption → Transformer Layers → Encryption              │
│                                                             │
│  • Transformer blocks only (no embeddings)                 │
│  • AWQ quantized weights (4-bit)                           │
│  • KV cache management (encrypted)                         │
│  • No access to rotation matrix M                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Encrypted KV cache
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              KV CACHE STORAGE (File-based)                 │
│                                                             │
│  • Filesystem: /workspace/weights/kv/{session_id}.bin      │
│  • AES-256-GCM encrypted                                   │
│  • Session-scoped keys                                      │
│  • No Redis (despite README mentions)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Communication Methods & Network Layer

### Protocol Usage Summary

| Component | Protocol | Status | Code Location | When Used |
|-----------|----------|--------|----------------|-----------|
| **Client** | gRPC | ✅ Implemented | `infemeral/client.py::Client._call_server()` | Every token generation (line 160: `stub.Infer()`) |
| **Client** | HTTP | ❌ Missing | N/A | Would be used for RunPod serverless |
| **Server** | gRPC | ❌ Missing | N/A | Would be used for traditional deployment |
| **Server** | HTTP | ✅ Implemented | `infemeral/server.py::handler()` | Every RunPod request (via `runpod.serverless.start()`) |

### Current State: Protocol Mismatch

**Critical Finding**: The codebase has a **protocol mismatch**:
- ✅ **Client**: Implements **gRPC only** (`infemeral/client.py`)
- ✅ **Server**: Implements **HTTP only** (RunPod serverless, `infemeral/server.py`)
- ❌ **No gRPC server** implementation exists
- ❌ **No HTTP client** implementation exists

**Result**: The current client cannot communicate with the current server without modifications.

**When Network is Used**:
- **Once per token generation**: Client sends cloaked embedding → Server processes → Server returns transformed hidden states
- **No network for**: Tokenization, embedding, cloaking, de-embedding, sampling, KV cache I/O
- **Frequency**: 2 network round-trips per token (1 request + 1 response)

---

### 1. gRPC (Client-Side Only)

**Status**: Client implementation complete, server missing

**Where Used**: `infemeral/client.py::Client._call_server()`

**Protocol**: `tensor_service.proto`
- **Service**: `TensorInference`
- **RPC**: `Infer(InferenceRequest) → InferenceResponse`
- **Port**: 50051 (default, configurable via `INFEMERAL_CLIENT_SERVER_URL`)

**Network Stack**:
```
Application Layer:  Client._call_server()
    ↓
gRPC Layer:         TensorInferenceStub.Infer()
    ↓
Transport Layer:    HTTP/2 (gRPC uses HTTP/2)
    ↓
TLS Layer:         ❌ NONE (insecure_channel)
    ↓
TCP Layer:          Port 50051
```

**Client Implementation Details** (`infemeral/client.py`):

1. **Channel Initialization** (lazy, on first `stub` access):
   ```python
   self._channel = grpc.insecure_channel(
       self.server_url,  # e.g., "localhost:50051"
       options=[
           ("grpc.max_send_message_length", 100 * 1024 * 1024),    # 100MB
           ("grpc.max_receive_message_length", 100 * 1024 * 1024), # 100MB
       ],
   )
   self._stub = tensor_service_pb2_grpc.TensorInferenceStub(self._channel)
   ```

2. **Request Construction**:
   ```python
   request = tensor_service_pb2.InferenceRequest(
       cloaked_embedding=encrypted_data,      # bytes (AES-GCM encrypted)
       encrypted_session_key=self.session_key, # bytes (TODO: RSA-wrap)
       nonce=nonce,                            # bytes (12-byte AES-GCM nonce)
       shape=shape,                            # list[int64] [batch, seq_len, hidden_dim]
       dtype=dtype,                            # str "float16"
       session_id=self.session_id,            # str (hex)
       max_new_tokens=1,                       # int32
       temperature=0.7,                        # float
   )
   ```

3. **Network Call**:
   ```python
   response = self.stub.Infer(request)  # Synchronous unary RPC
   ```

4. **Response Handling**:
   ```python
   if response.error:
       raise RuntimeError(f"Server error: {response.error}")
   
   tensor = deserialize_tensor(
       response.output,      # bytes (encrypted)
       list(response.shape), # list[int64]
       response.dtype,      # str
       device=self.device,
   )
   ```

**Message Format** (protobuf):
```protobuf
InferenceRequest {
    bytes cloaked_embedding      // AES-GCM encrypted tensor bytes
    bytes encrypted_session_key  // Currently plain (TODO: RSA-wrap)
    bytes nonce                  // 12-byte AES-GCM nonce
    repeated int64 shape         // [batch, seq_len, hidden_dim]
    string dtype                 // "float16", "float32", "bfloat16"
    string session_id            // Unique session identifier
    int32 max_new_tokens         // Generation limit
    float temperature            // Sampling temperature
}

InferenceResponse {
    bytes output                 // Encrypted output tensor
    repeated int64 shape         // Output shape
    string dtype                 // Output dtype
    int32 tokens_processed       // Number of tokens processed
    string error                 // Error message if any
}
```

**Security**:
- ❌ **No TLS**: Uses `grpc.insecure_channel()` (plaintext HTTP/2)
- ⚠️ **Application-level encryption**: AES-256-GCM (encrypts payload, not transport)
- ⚠️ **Session key**: Sent plaintext in `encrypted_session_key` field (TODO: RSA-OAEP)

**Server Implementation**: **MISSING**
- `tensor_service_pb2_grpc.py` has base class `TensorInferenceServicer`
- No actual server implementation found
- Would need to:
  1. Implement `TensorInferenceServicer` subclass
  2. Override `Infer()` method
  3. Convert protobuf → dict format
  4. Call `handler()` function
  5. Convert dict → protobuf response
  6. Start gRPC server on port 50051

---

### 2. HTTP/RunPod Serverless (Server-Side Only)

**Status**: Server implementation complete, client missing

**Where Used**: `infemeral/server.py::handler()`

**Network Stack**:
```
Application Layer:  handler(event: dict) -> dict
    ↓
RunPod SDK:        runpod.serverless.start({"handler": handler})
    ↓
HTTP Layer:        POST /run (RunPod API endpoint)
    ↓
TLS Layer:         ✅ HTTPS (RunPod handles TLS termination)
    ↓
TCP Layer:          Port 443 (HTTPS)
```

**Server Implementation Details** (`infemeral/server.py`):

1. **Entry Point**:
   ```python
   if __name__ == "__main__":
       import runpod
       runpod.serverless.start({"handler": handler})
   ```

2. **Request Format** (RunPod event dict):
   ```python
   {
       "input": {
           "cloaked_embedding": str,      # Base64-encoded encrypted tensor
           "encrypted_session_key": str,   # Base64-encoded AES key
           "nonce": str,                   # Base64-encoded nonce
           "shape": [batch, seq_len, hidden_dim],  # list[int]
           "dtype": "float16",             # str
           "session_id": str               # str
       }
   }
   ```

3. **Base64 Decoding** (HTTP-specific):
   ```python
   if isinstance(session_key, str):
       session_key = base64.b64decode(session_key)
   if isinstance(nonce, str):
       nonce = base64.b64decode(nonce)
   if isinstance(cloaked_data, str):
       cloaked_data = base64.b64decode(cloaked_data)
   ```

4. **Response Format**:
   ```python
   {
       "output": str,              # Base64-encoded encrypted output
       "nonce": str,               # Base64-encoded output nonce
       "shape": [batch, seq_len, hidden_dim],  # list[int]
       "dtype": "float16",         # str
       "tokens_processed": int     # int
   }
   ```

5. **Base64 Encoding** (HTTP-specific):
   ```python
   return {
       "output": base64.b64encode(encrypted_output).decode(),
       "nonce": base64.b64encode(output_nonce).decode(),
       ...
   }
   ```

**Deployment**: RunPod serverless
- **Invocation**: RunPod API calls `handler()` function
- **Scaling**: Automatic (serverless)
- **Cold Start**: Model loads on first request (~30-60s)
- **Cost**: Pay-per-request

**Security**:
- ✅ **TLS**: HTTPS (RunPod handles TLS termination)
- ✅ **Application-level encryption**: AES-256-GCM (encrypts payload)
- ⚠️ **Session key**: Sent plaintext (base64-encoded, but unencrypted)

**Client Implementation**: **MISSING**
- No HTTP client exists in codebase
- Would need to:
  1. Implement HTTP client in `Client._call_server()`
  2. Use `requests` or `httpx` library
  3. POST to RunPod endpoint
  4. Base64 encode request fields
  5. Base64 decode response fields

---

### Network Flow Comparison

#### gRPC Flow (Client → Server, if server existed)

```
CLIENT                                    SERVER
─────────────────────────────────────────────────────────────────
1. Serialize tensor → bytes
2. Encrypt (AES-256-GCM) → ciphertext
3. Build protobuf InferenceRequest
   └─ cloaked_embedding: bytes (raw)
   └─ encrypted_session_key: bytes (raw)
   └─ nonce: bytes (raw)
   └─ shape: list[int64]
   └─ dtype: string
   └─ session_id: string
4. gRPC call: stub.Infer(request)
   └─ HTTP/2 POST /infemeral.TensorInference/Infer
   └─ Protobuf binary encoding
   └─ ❌ No TLS (insecure)
   └─ Port 50051
                                      ↓
                                      ↓
                                   5. Receive protobuf
                                   6. Decode InferenceRequest
                                   7. Extract bytes fields (no base64)
                                   8. Decrypt → plaintext
                                   9. Deserialize → tensor
                                   10. Process (transformer forward)
                                   11. Serialize → bytes
                                   12. Encrypt → ciphertext
                                   13. Build protobuf InferenceResponse
                                   14. gRPC response
                                      └─ HTTP/2 response
                                      └─ Protobuf binary encoding
                                      ↓
                                      ↓
15. Receive protobuf InferenceResponse
16. Extract bytes fields (no base64)
17. Decrypt → plaintext
18. Deserialize → tensor
```

#### HTTP/RunPod Flow (Current, but no client)

```
CLIENT                                    SERVER (RunPod)
─────────────────────────────────────────────────────────────────
1. Serialize tensor → bytes
2. Encrypt (AES-256-GCM) → ciphertext
3. Base64 encode ciphertext → str
4. Build JSON dict
   └─ cloaked_embedding: str (base64)
   └─ encrypted_session_key: str (base64)
   └─ nonce: str (base64)
   └─ shape: list[int]
   └─ dtype: string
   └─ session_id: string
5. HTTP POST to RunPod endpoint
   └─ Content-Type: application/json
   └─ ✅ HTTPS (TLS)
   └─ Port 443
   └─ Body: JSON string
                                      ↓
                                      ↓
                                   6. RunPod receives request
                                   7. Calls handler(event)
                                   8. Extract JSON fields
                                   9. Base64 decode → bytes
                                   10. Decrypt → plaintext
                                   11. Deserialize → tensor
                                   12. Process (transformer forward)
                                   13. Serialize → bytes
                                   14. Encrypt → ciphertext
                                   15. Base64 encode → str
                                   16. Build JSON dict response
                                   17. HTTP response
                                      └─ ✅ HTTPS (TLS)
                                      └─ Body: JSON string
                                      ↓
                                      ↓
18. Receive JSON response
19. Extract string fields
20. Base64 decode → bytes
21. Decrypt → plaintext
22. Deserialize → tensor
```

---

### When Each Protocol is Used

**Current Codebase**:
- **gRPC**: Only in client (`Client._call_server()`), but **no server to connect to**
- **HTTP**: Only in server (`handler()`), but **no client to call it**

**Intended Usage** (per README):
- **gRPC**: Traditional always-on deployment (client + server both gRPC)
- **HTTP**: RunPod serverless deployment (client + server both HTTP)

**Actual Usage**:
- **Neither works end-to-end** without implementing the missing pieces

---

### Protocol Differences

| Aspect | gRPC | HTTP/RunPod |
|--------|------|-------------|
| **Encoding** | Protobuf (binary) | JSON (text) |
| **Base64** | ❌ No (bytes sent raw) | ✅ Yes (bytes → base64 string) |
| **Transport** | HTTP/2 | HTTP/1.1 or HTTP/2 |
| **TLS** | ❌ None (insecure_channel) | ✅ Yes (RunPod HTTPS) |
| **Message Size** | 100MB limit (configurable) | RunPod limit (~6MB body) |
| **Connection** | Persistent (HTTP/2) | Stateless (per request) |
| **Latency** | Lower (binary, HTTP/2) | Higher (JSON, HTTP/1.1) |
| **Streaming** | ✅ Supported (not used) | ❌ Not supported |
| **Error Handling** | gRPC status codes | HTTP status codes + JSON error |

---

## Network Layer in the Pipeline

### Exact Points Where Network is Used

The network layer is **only used once per token generation** - to send cloaked embeddings to the server and receive transformed hidden states back.

#### Step-by-Step Pipeline with Network Calls

```
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: Tokenization (Local, No Network)                        │
└─────────────────────────────────────────────────────────────────┘
  Text: "Hello, how are you?"
    ↓
  Tokenizer.encode() → [15496, 11, 527, 499, 527, 499]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: Embedding (Local, No Network)                           │
└─────────────────────────────────────────────────────────────────┘
  embed_tokens(input_ids) → hidden_states [1, 6, 4096]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: Cloaking (Local, No Network)                            │
└─────────────────────────────────────────────────────────────────┘
  cloak(hidden_states, ctx) → cloaked [1, 6, 4096]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: Serialization (Local, No Network)                      │
└─────────────────────────────────────────────────────────────────┘
  serialize_tensor(cloaked) → (bytes, shape, dtype)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: Encryption (Local, No Network)                        │
└─────────────────────────────────────────────────────────────────┘
  encrypt_bytes(data, session_key) → (ciphertext, nonce)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ ⚡ NETWORK CALL #1: Client → Server ⚡                          │
│                                                                 │
│ Protocol: gRPC (if server existed) OR HTTP (if client existed) │
│ Location: Client._call_server() → stub.Infer() OR HTTP POST    │
│                                                                 │
│ Data Transmitted:                                               │
│   - Encrypted tensor bytes (AES-256-GCM)                       │
│   - Session key (plaintext, TODO: RSA-wrap)                    │
│   - Nonce (12 bytes)                                            │
│   - Shape, dtype, session_id (metadata)                          │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: Decryption (Remote, After Network)                     │
└─────────────────────────────────────────────────────────────────┘
  decrypt_bytes(ciphertext, session_key, nonce) → plaintext
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: Deserialization (Remote, After Network)               │
└─────────────────────────────────────────────────────────────────┘
  deserialize_tensor(plaintext, shape, dtype) → tensor
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: KV Cache Load (Local Filesystem, No Network)           │
└─────────────────────────────────────────────────────────────────┘
  load_kv_cache(session_id, session_key) → past_key_values
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: Transformer Forward (Remote, GPU, No Network)          │
└─────────────────────────────────────────────────────────────────┘
  forward_transformer(model, tensor, past_key_values) → output
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: KV Cache Save (Local Filesystem, No Network)            │
└─────────────────────────────────────────────────────────────────┘
  save_kv_cache(session_id, new_kv, session_key)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: Serialization (Remote, No Network)                     │
└─────────────────────────────────────────────────────────────────┘
  serialize_tensor(output) → (bytes, shape, dtype)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: Encryption (Remote, No Network)                        │
└─────────────────────────────────────────────────────────────────┘
  encrypt_bytes(output_bytes, session_key) → (ciphertext, nonce)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ ⚡ NETWORK CALL #2: Server → Client ⚡                          │
│                                                                 │
│ Protocol: gRPC (if server existed) OR HTTP (if client existed)│
│ Location: Server returns InferenceResponse OR JSON dict         │
│                                                                 │
│ Data Transmitted:                                               │
│   - Encrypted output tensor bytes (AES-256-GCM)                 │
│   - Output nonce (12 bytes)                                     │
│   - Output shape, dtype (metadata)                              │
│   - tokens_processed (metadata)                                 │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: Decryption (Local, After Network)                      │
└─────────────────────────────────────────────────────────────────┘
  decrypt_bytes(ciphertext, session_key, nonce) → plaintext
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: Deserialization (Local, After Network)                │
└─────────────────────────────────────────────────────────────────┘
  deserialize_tensor(plaintext, shape, dtype) → tensor
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: Uncloaking (Local, No Network)                        │
└─────────────────────────────────────────────────────────────────┘
  uncloak(tensor, ctx) → uncloaked [1, 6, 4096]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: De-embedding (Local, No Network)                      │
└─────────────────────────────────────────────────────────────────┘
  lm_head(uncloaked[:, -1:, :]) → logits [1, 1, 128256]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: Sampling (Local, No Network)                          │
└─────────────────────────────────────────────────────────────────┘
  nucleus_sampling(logits) → next_token_id: 527
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT: Decoding (Local, No Network)                           │
└─────────────────────────────────────────────────────────────────┘
  Tokenizer.decode([15496, 11, 527, ...]) → "Hello, how are you? I"
```

### Network Call Frequency

**Per Token Generation**:
- **2 network calls** per token: 1 request + 1 response
- **No network calls** for:
  - Tokenization
  - Embedding
  - Cloaking/uncloaking
  - De-embedding
  - Sampling
  - KV cache I/O (filesystem, not network)

**Example**: Generating 100 tokens = **200 network round-trips** (100 requests + 100 responses)

### Code Locations

**Client Network Call** (`infemeral/client.py`):
```python
# Line 139-171: Client._call_server()
def _call_server(self, cloaked: torch.Tensor) -> torch.Tensor:
    # ... serialize, encrypt ...
    
    # ⚡ NETWORK CALL HAPPENS HERE ⚡
    response = self.stub.Infer(request)  # gRPC call (line 160)
    # OR: response = requests.post(url, json=payload)  # HTTP (not implemented)
    
    # ... decrypt, deserialize ...
```

**Server Network Handler** (`infemeral/server.py`):
```python
# Line 478-578: handler() function
def handler(event: dict) -> dict:
    # ⚡ NETWORK REQUEST RECEIVED HERE ⚡
    # (RunPod calls this function via HTTP)
    
    # ... decrypt, process, encrypt ...
    
    # ⚡ NETWORK RESPONSE SENT HERE ⚡
    return {
        "output": base64.b64encode(encrypted_output).decode(),
        # ...
    }
```

### Network Protocol Decision Point

**Current Code**: No decision point exists - client hardcoded to gRPC, server hardcoded to HTTP.

**Would Need** (to support both):
```python
class Client:
    def __init__(self, transport: str = "grpc", ...):
        self.transport = transport  # "grpc" or "http"
    
    def _call_server(self, cloaked: torch.Tensor):
        if self.transport == "grpc":
            return self._call_server_grpc(cloaked)
        elif self.transport == "http":
            return self._call_server_http(cloaked)
```

---

## Control Flow

### Client-Side Flow (`infemeral/client.py`)

1. **Initialization** (`Client.__init__`):
   - Load tokenizer from HuggingFace (`client_settings.model_id`)
   - Load embedding layers from `client_settings.weights_path`
   - Generate session ID (16-byte hex)
   - Generate session key (32-byte AES-256)
   - Create cloaking context (orthogonal matrix + DP params)

2. **Generation Loop** (`Client.generate()`):
   ```
   FOR each token to generate:
       a. Tokenize prompt → input_ids
       b. Embed: embed_tokens(input_ids) → hidden_states [batch, seq_len, 4096]
       c. Cloak: cloak(hidden_states, ctx) → cloaked [batch, seq_len, 4096]
          - Add DP noise: hidden + N(0, σ²)
          - Rotate: cloaked = hidden @ M.T
       d. Serialize: serialize_tensor(cloaked) → bytes, shape, dtype
       e. Encrypt: encrypt_bytes(data, session_key) → ciphertext, nonce
       f. Send to server (gRPC or HTTP)
       g. Receive encrypted response
       h. Decrypt: decrypt_bytes(ciphertext, session_key, nonce) → plaintext
       i. Deserialize: deserialize_tensor(plaintext, shape, dtype) → tensor
       j. Uncloak: uncloak(tensor, ctx) → uncloaked
       k. De-embed: lm_head(uncloaked[:, -1:, :]) → logits [vocab_size]
       l. Sample: nucleus_sampling(logits, temperature, top_p) → next_token_id
       m. Append to generated_ids
       n. Check EOS → break if done
   ```

3. **Server Communication** (`Client._call_server()`):
   - Serialize tensor to bytes (NumPy array → bytes)
   - Encrypt with AES-256-GCM
   - Build protobuf request (gRPC) or dict (HTTP)
   - Send via stub.Infer() or HTTP POST
   - Deserialize and decrypt response

### Server-Side Flow (`infemeral/server.py`)

1. **Cold Start** (`handler()` called):
   - Load model: `load_model()` → cached global `_model`
   - Model loading priority:
     1. Tensorizer format (`.tensors`) - fastest, ~5GB/s streaming
     2. Generate tensorized from safetensors if missing
     3. Safetensors fallback (`.safetensors`)
   - AWQ detection: checks for `WQLinear_GEMM` modules
   - Device: CUDA if available, else CPU

2. **Request Processing**:
   ```
   a. Decode base64 inputs (if HTTP)
   b. Decrypt: decrypt_bytes(cloaked_embedding, session_key, nonce) → plaintext
   c. Deserialize: deserialize_tensor(plaintext, shape, dtype) → tensor [batch, seq_len, 4096]
   d. Load KV cache: load_kv_cache(session_id, session_key) → past_key_values
      - File: /workspace/weights/kv/{session_id}.bin
      - Encrypted with AES-256-GCM
      - Format: nonce (12B) + ciphertext
      - Returns None if not found
   e. Forward pass: forward_transformer(model, tensor, past_key_values)
      - Iterate through transformer.layers
      - Compute rotary embeddings per layer
      - Handle AWQ quantization quirks
      - Return: output_hidden_states, new_kv_cache
   f. Save KV cache: save_kv_cache(session_id, new_kv, session_key)
      - Pack KV tensors: pack_kv_cache(keys, values) → bytes
      - Encrypt: encrypt_bytes(packed_data, session_key) → ciphertext, nonce
      - Write: nonce (12B) + ciphertext to file
   g. Serialize output: serialize_tensor(output) → bytes, shape, dtype
   h. Encrypt output: encrypt_bytes(output_bytes, session_key) → ciphertext, nonce
   i. Encode base64 (if HTTP)
   j. Return response dict
   k. Memory cleanup: del tensors, torch.cuda.empty_cache()
   ```

3. **Transformer Forward Pass** (`forward_transformer()`):
   - Extract layers: `model.model.layers` (Llama-style)
   - Create attention mask: `torch.ones(batch, seq_len)`
   - Compute position IDs: `torch.arange(seq_len)` or `past_len + seq_len`
   - For each layer:
     - Get rotary embeddings: `layer.self_attn.rotary_emb(hidden_states, position_ids)`
     - Handle AWQ-specific dimension issues
     - Call layer: `layer(hidden_states, attention_mask, position_ids, past_key_value, use_cache=True, position_embeddings=(cos, sin))`
     - Accumulate KV cache
   - Final layer norm: `transformer.norm(hidden_states)`
   - Return: `(hidden_states, tuple(new_kv_cache))`

---

## Data Storage

### KV Cache Storage

**Location**: Filesystem-based (not Redis despite README mentions)

**Path**: `/workspace/weights/kv/{session_id}.bin` (configurable via `INFEMERAL_SERVER_KV_CACHE_DIR`)

**Format**:
```
[12 bytes: nonce] + [AES-GCM ciphertext]
```

**Ciphertext Structure** (after decryption):
```
[4 bytes: key_data_length (uint32)]
[4 bytes: num_dimensions (uint32)]
[8 bytes × ndim: shape (int64 each)]
[key_data_length bytes: key tensor data (float16)]
[key_data_length bytes: value tensor data (float16)]
```

**Encryption**: AES-256-GCM
- Key: Session-specific (32 bytes)
- Nonce: Random 12 bytes per write
- Authentication: Built-in GCM tag

**Operations**:
- `save_kv_cache(session_id, keys, values, session_key)`: Encrypt and write
- `load_kv_cache(session_id, session_key, device)`: Read, decrypt, unpack
- `delete_kv_cache(session_id)`: Unlink file

**Current Limitations**:
- No TTL/expiration (files persist indefinitely)
- No LRU eviction
- No Redis integration (despite README architecture diagram)
- TODO: Implement tide-windowing for context > 2048 tokens

### Model Weights Storage

**Client Weights**: `client_settings.weights_path` (default: `/workspace/weights/client_weights.safetensors`)
- Contains: `embed_tokens.weight`, `lm_head.weight`
- Format: SafeTensors
- Size: ~1.5 GB (for Llama 3.1 8B)

**Server Weights**: Two formats supported
1. **Tensorizer** (preferred): `server_settings.tensorized_weights_path`
   - Format: `.tensors` (binary streaming format)
   - Loading speed: ~5 GB/s
   - Generated from safetensors via `tensorize_server_weights()`
2. **SafeTensors** (fallback): `server_settings.weights_path`
   - Format: `.safetensors`
   - Contains: Transformer layers only (no embeddings)
   - AWQ quantized: 4-bit weights

**Model Preparation** (`infemeral/model_prep.py`):
- `split_model()`: Extracts client/server components
- `tensorize_server_weights()`: Converts safetensors → tensorizer format
- Handles AWQ quantization extraction (qweight, qzeros, scales)

---

## Processing Locations

### Client Processing (`infemeral/client.py`)

**On Client Device**:
- ✅ Tokenization (text ↔ token IDs)
- ✅ Embedding (token IDs → hidden states)
- ✅ Cloaking (DP noise + orthogonal rotation)
- ✅ Encryption/decryption (AES-256-GCM)
- ✅ De-embedding (hidden states → logits)
- ✅ Sampling (logits → token ID)
- ✅ Decoding (token IDs → text)

**Client Components**:
- `EmbeddingLayer`: `embed_tokens` + `lm_head` modules
- `CloakingContext`: Orthogonal matrix M (4096×4096), DP sigma
- `Client`: Orchestrates generation loop

### Server Processing (`infemeral/server.py`)

**On Server (RunPod/GPU)**:
- ✅ Decryption (AES-256-GCM)
- ✅ Tensor deserialization
- ✅ Transformer forward pass (all layers)
- ✅ KV cache management
- ✅ Tensor serialization
- ✅ Encryption (AES-256-GCM)

**Server Components**:
- `load_model()`: Loads transformer layers (cached globally)
- `forward_transformer()`: Runs transformer blocks
- `load_kv_cache()` / `save_kv_cache()`: Encrypted KV storage
- `handler()`: RunPod serverless entry point

**What Server Cannot Access**:
- ❌ Raw token IDs
- ❌ Raw embeddings (only sees cloaked)
- ❌ Rotation matrix M
- ❌ LM head (logits generation)
- ❌ Tokenizer

---

## Security Architecture

### Encryption Layers

1. **Transport Encryption**:
   - gRPC: Currently insecure (TODO: TLS)
   - HTTP: HTTPS (handled by RunPod)

2. **Application Encryption**:
   - **Algorithm**: AES-256-GCM
   - **Key**: Session-specific (32 bytes, random)
   - **Nonce**: 12 bytes, random per message
   - **Scope**: All tensor data (input/output)
   - **KV Cache**: Also encrypted at rest

3. **Cloaking** (Privacy-Preserving):
   - **Orthogonal Rotation**: `cloaked = (hidden + noise) @ M.T`
   - **Matrix M**: Random orthogonal (4096×4096), session-specific
   - **DP Noise**: Gaussian, σ computed from (ε=2.0, δ=1e-5)
   - **Property**: Preserves attention dot products (M is orthogonal)

### Key Management

**Session Keys**:
- Generated: `generate_session_key()` → `os.urandom(32)`
- Currently sent plaintext (TODO: RSA-OAEP wrapping)
- Scope: Single session (ephemeral)
- Used for: AES-256-GCM encryption

**Cloaking Context**:
- Seed: Random per session (`secrets.randbelow(2**31)`)
- Matrix: Generated from seed (deterministic)
- Scope: Single session

### Threat Model

**Assumptions**:
- Server operator has full access to server code/memory
- Server can observe network traffic (encrypted)
- Server can modify behavior (but client detects tampering)
- KV cache storage is untrusted (encrypted)

**Mitigations**:
- ✅ Embedding privacy: Orthogonal rotation + DP noise
- ✅ Forward secrecy: Session keys rotate per session
- ✅ State confidentiality: AES-256-GCM encrypted KV cache
- ✅ Model extraction prevention: Client holds embedding layers
- ⚠️ Known-plaintext attacks: DP noise (ε=2.0)
- ⚠️ Session correlation: Fresh rotation per session

---

## Configuration

### Environment Variables

**Client** (`INFEMERAL_CLIENT_*`):
- `INFEMERAL_CLIENT_WEIGHTS_PATH`: Client weights path
- `INFEMERAL_CLIENT_SERVER_URL`: gRPC server URL (default: `localhost:50051`)
- `INFEMERAL_CLIENT_MODEL_ID`: HuggingFace model ID for tokenizer

**Server** (`INFEMERAL_SERVER_*`):
- `INFEMERAL_SERVER_TENSORIZED_WEIGHTS_PATH`: Tensorizer weights path
- `INFEMERAL_SERVER_WEIGHTS_PATH`: SafeTensors fallback path
- `INFEMERAL_SERVER_MODEL_ID`: Model ID for config/architecture
- `INFEMERAL_SERVER_KV_CACHE_DIR`: KV cache directory (default: `/workspace/weights/kv`)
- `INFEMERAL_SERVER_MAX_CONTEXT_LENGTH`: Max context (default: 2048)
- `INFEMERAL_SERVER_GRPC_PORT`: gRPC port (default: 50051)

**Crypto** (`INFEMERAL_CRYPTO_*`):
- `INFEMERAL_CRYPTO_HIDDEN_DIM`: Model hidden dimension (default: 4096)
- `INFEMERAL_CRYPTO_DP_EPSILON`: DP epsilon (default: 2.0)
- `INFEMERAL_CRYPTO_DP_DELTA`: DP delta (default: 1e-5)

---

## Model Architecture

### Split Model Design

**Client Holds**:
- `embed_tokens`: Token ID → Hidden State (vocab_size × 4096)
- `lm_head`: Hidden State → Logits (4096 × vocab_size)

**Server Holds**:
- Transformer layers (32 layers for Llama 3.1 8B)
- Layer norms
- Rotary embeddings (computed, not stored)
- Attention mechanisms
- Feed-forward networks

**Model Format**:
- Base: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`
- Quantization: AWQ (4-bit weights)
- Precision: float16 activations
- Architecture: Llama-style (transformer.layers)

---

## Performance Characteristics

### Latency Breakdown (Estimated)

- **Client Cloaking**: ~2.3ms (matrix multiplication)
- **DP Noise**: ~0.8ms
- **Serialization**: ~5.2ms (512 tokens)
- **Network**: Variable (gRPC/HTTP)
- **Server Decryption**: ~1ms
- **Transformer Forward**: ~180ms (TTFT, first token)
- **Server Encryption**: ~1ms
- **Client Uncloaking**: ~2.3ms
- **De-embedding**: ~1ms
- **Sampling**: ~0.5ms

**Total Overhead**: ~8ms per request (privacy operations)

### Memory Usage

- **Rotation Matrix**: 64 MB (4096² × float32)
- **KV Cache**: ~512 MB (2048 tokens, 32 layers)
- **Client Embeddings**: ~1.5 GB
- **Server Model**: ~4-8 GB (AWQ quantized)

---

## Current Limitations & TODOs

### Missing Implementations

1. **gRPC Server**: No actual server implementation (only handler exists)
2. **RSA Key Wrapping**: Session keys sent plaintext
3. **TLS**: gRPC uses insecure channel
4. **KV Cache Conversion**: `past_key_values` format conversion incomplete
5. **Tide-Windowing**: Context compression for >2048 tokens not implemented
6. **Redis Integration**: README mentions Redis, but code uses filesystem

### Known Issues

1. **AWQ Rotary Embeddings**: Dimension mismatches handled with workarounds
2. **KV Cache Loading**: Returns None but doesn't convert to `past_key_values` format
3. **Error Handling**: Basic, could be more robust
4. **Session Management**: No cleanup/expiration for KV cache files

---

## Deployment Modes

### 1. RunPod Serverless (Current)

- **Entry**: `handler()` function
- **Transport**: HTTP (via RunPod)
- **Scaling**: Automatic (serverless)
- **Cost**: Pay-per-request
- **Cold Start**: Model loading on first request (~30-60s)

### 2. Traditional gRPC (Planned)

- **Entry**: Would need `TensorInferenceServicer` implementation
- **Transport**: gRPC (port 50051)
- **Scaling**: Manual (horizontal scaling)
- **Cost**: Always-on pricing
- **Cold Start**: None (model stays loaded)

---

## File Structure Reference

```
infemeral/
├── client.py              # Client implementation (gRPC/HTTP)
├── server.py              # Server handler (RunPod serverless)
├── config.py              # Configuration (Pydantic settings)
├── crypto.py              # Encryption, cloaking, DP
├── tensors.py             # Tensor serialization
├── model_prep.py          # Model splitting/tensorization
├── tensor_service_pb2.py   # Generated protobuf messages
└── tensor_service_pb2_grpc.py  # Generated gRPC stubs

tensor_service.proto       # gRPC service definition
```

---

## Summary

**Infemeral** is a privacy-preserving LLM inference system that:

1. **Splits computation**: Client handles embeddings, server handles transformer layers
2. **Uses cloaking**: Orthogonal rotation + DP noise prevents server from seeing raw data
3. **Encrypts everything**: AES-256-GCM for transport and KV cache storage
4. **Supports two modes**: RunPod serverless (HTTP, implemented) and gRPC (planned)
5. **Stores state**: Encrypted KV cache on filesystem (not Redis)
6. **Processes**: Client does tokenization/embedding/sampling, server does transformer inference

The system is **functional for RunPod serverless deployment** but **missing the gRPC server implementation** for traditional deployments.
