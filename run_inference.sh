#!/bin/bash
# Run inference on RunPod from local machine
# Usage: ./run_inference.sh "Your prompt here" [max_tokens] [model]
#
# Models:
#   llama (default): Llama 3.1 8B AWQ
#   deepseek: DeepSeek-R1-32B GPTQ

POD_IP="203.57.40.175"
POD_PORT="10271"
PROMPT="${1:-Hello, how are you?}"
MAX_TOKENS="${2:-20}"
MODEL="${3:-llama}"

# Set model-specific paths
if [ "$MODEL" = "deepseek" ]; then
    WEIGHTS_PATH="/workspace/weights/deepseek-r1-32b"
    CLIENT_WEIGHTS="/workspace/weights/deepseek-r1-32b-client/client_weights.safetensors"
    MODEL_ID="dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4"
    echo "Using DeepSeek-R1-32B model"
else
    WEIGHTS_PATH="/workspace/weights/model"
    CLIENT_WEIGHTS="/workspace/weights/client_weights.safetensors"
    MODEL_ID="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    echo "Using Llama 3.1 8B model"
fi

ssh -o StrictHostKeyChecking=no -p "$POD_PORT" root@"$POD_IP" "
cd /workspace/infemeral-src
export INFEMERAL_SERVER_WEIGHTS_DIR=$WEIGHTS_PATH
export INFEMERAL_SERVER_MODEL_ID=$MODEL_ID
export INFEMERAL_CLIENT_WEIGHTS_PATH=$CLIENT_WEIGHTS
export INFEMERAL_CLIENT_MODEL_ID=$MODEL_ID
/mnt/.venv/bin/python -c \"
from infemeral.client import Client

client = Client(
    weights_path='$CLIENT_WEIGHTS',
    server_url='localhost:50051',
    device='cuda'
)

result = client.generate('$PROMPT', max_new_tokens=$MAX_TOKENS)
print(result)
client.close()
\"
"
