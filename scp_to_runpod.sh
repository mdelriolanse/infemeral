#!/bin/bash
# Transfer a file to a RunPod pod using scp
# Usage: ./scp_to_runpod.sh <pod_ip> <port> <filename> [remote_target_dir]

set -e

SSH_KEY="$HOME/.ssh/runpod"
USERNAME="root"
BASE_DIR="/workspace/infemeral-src"

# Validate arguments
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 <pod_ip> <port> <filename> [remote_target_dir]"
    echo "  pod_ip          - IP address of the RunPod pod"
    echo "  port            - SSH port of the RunPod pod"
    echo "  filename        - Path to the file to transfer"
    echo "  remote_target_dir - (Optional) Remote target directory"
    echo "                      If starts with '/', treated as absolute path"
    echo "                      Otherwise, relative to $BASE_DIR"
    echo "                      Default: $BASE_DIR/infemeral/"
    exit 1
fi

POD_IP="$1"
PORT="$2"
FILENAME="$3"

# Determine destination directory
if [ $# -eq 4 ]; then
    REMOTE_DIR="$4"
    # If it starts with '/', use as absolute path
    if [[ "$REMOTE_DIR" == /* ]]; then
        DEST_DIR="$REMOTE_DIR"
    else
        # Otherwise, make it relative to BASE_DIR
        DEST_DIR="$BASE_DIR/$REMOTE_DIR"
    fi
    # Ensure DEST_DIR ends with /
    if [[ "$DEST_DIR" != */ ]]; then
        DEST_DIR="$DEST_DIR/"
    fi
else
    # Default to infemeral directory
    DEST_DIR="$BASE_DIR/infemeral/"
fi

# Check that the source file exists
if [ ! -f "$FILENAME" ]; then
    echo "Error: File '$FILENAME' does not exist"
    exit 1
fi

# Check that the SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "Error: SSH key not found at '$SSH_KEY'"
    exit 1
fi

# Execute scp command
echo "Transferring '$FILENAME' to $USERNAME@$POD_IP:$DEST_DIR ..."
scp -i "$SSH_KEY" -P "$PORT" "$FILENAME" "$USERNAME@$POD_IP:$DEST_DIR"

if [ $? -eq 0 ]; then
    echo "Transfer complete!"
else
    echo "Error: Transfer failed"
    exit 1
fi
