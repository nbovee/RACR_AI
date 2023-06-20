#!/bin/bash

# Check if required argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_remote_container.sh <PORT>"
    exit 1
fi

PORT=$1

nvidia-docker run --gpus all --ipc=host -p $PORT:$PORT -e PORT=$PORT -e PYTHONUNBUFFERED=1 cuda-remote-oas
