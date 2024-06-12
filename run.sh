#!/bin/bash

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

# Determine the role and command
ROLE=$1
if [ -z "$ROLE" ]; then
    echo "No role provided. Please provide a role (observer or dummy) as a non-option argument."
    exit 1
fi

# Determine the command based on the role
if [ "$ROLE" = "observer" ]; then
  CMD="python /app/src/tracr/app_api/deploy.py"
elif [ "$ROLE" = "dummy" ]; then
  CMD="python /app/src/tracr/experiment_design/services/basic_split_inference.py"
else
  echo "Invalid role: $ROLE. Please provide either 'observer' or 'dummy'."
  exit 1
fi

# Run container
echo "Running container from $TRACR_IMAGE_NAME with role $ROLE..."
docker run -p $RLOG_SERVER_PORT:9000 -it --name tracr-$ROLE --net=host -v ${HOST_VOLUME_PATH}:${CONTAINER_VOLUME_PATH} "$TRACR_IMAGE_NAME" $ROLE $CMD