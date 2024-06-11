#!/bin/bash

TRACR_IMAGE_NAME="tracr_base_image"
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Volume mapping
HOST_VOLUME_PATH="$ROOT_DIR"

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

# Run container
echo "Running container from $TRACR_IMAGE_NAME..."
docker run -it --net=host -v ${HOST_VOLUME_PATH}:${CONTAINER_VOLUME_PATH} "$TRACR_IMAGE_NAME" python "${CONTAINER_VOLUME_PATH}/app.py" "$@"
