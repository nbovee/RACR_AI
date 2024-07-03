#!/bin/bash

set -e

TRACR_IMAGE_NAME="tracr-app"
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Volume mapping
HOST_VOLUME_PATH="$ROOT_DIR"
CONTAINER_VOLUME_PATH="/app"

# Port mappings
RLOG_SERVER_PORT=9000
RPC_REGISTRY_SERVER_PORT=18812

# Check if image exists
if ! docker image inspect "$TRACR_IMAGE_NAME" > /dev/null 2>&1; then
    echo "Image $TRACR_IMAGE_NAME does not exist. Building it now..."
    docker build -t "$TRACR_IMAGE_NAME" "$ROOT_DIR"
else
    echo "Image $TRACR_IMAGE_NAME exists."
fi

# Determine the command based on the arguments
if [ "$1" = "experiment" ] && [ "$2" = "run" ] && [ -n "$3" ]; then
    EXPERIMENT_NAME=$3
    CMD="python -m tracr.app_api.deploy experiment run $EXPERIMENT_NAME"
elif [ "$1" = "observer" ] || [ "$1" = "participant" ]; then
    ROLE=$1
    if [ "$ROLE" = "observer" ]; then
        CMD="python -m tracr.app_api.deploy"
    elif [ "$ROLE" = "participant" ]; then
        CMD="python -m tracr.experiment_design.services.basic_split_inference"
    fi
else
    echo "Invalid command. Usage: ./run.sh [observer|participant] or ./run.sh experiment run <EXPERIMENT_NAME>"
    exit 1
fi

# Run container
echo "Running container from $TRACR_IMAGE_NAME..."
docker run -p $RLOG_SERVER_PORT:9000 -it --name tracr-container --net=host -v ${HOST_VOLUME_PATH}:${CONTAINER_VOLUME_PATH} "$TRACR_IMAGE_NAME" $CMD