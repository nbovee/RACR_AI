#!/bin/bash

# Check if required argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_remote_container.sh <PORT>"
    exit 1
fi

PORT=$2

nvidia-docker run --gpus all --ipc=host -p 8893:8893 -e PORT=$PORT cuda-remote-oas
