#!/bin/bash

# deploy_node.sh - Runs (after building, if an image doesn't already exist) a
# client or remote node for cooperative DNN inference. Here, "node" refers to a
# Docker container running on either a Jetson, Raspberry Pi, or PC, which
# communicates with other nodes on the network using the gRPC protocol.
#
# FLAGS:
# 	-n		"no CUDA"	Use for hosts without CUDA support (like the Pi 3)
#	-r		"remote"	Deploys a remote/server host (rather than client)

# Default values if no flags are set
CUDA_STATE="cuda"
BUILD_ARG="client"
ARCH="x86"

# parse through flags
while getopts ":nr" opt; do
  case $opt in
    n)
      CUDA_STATE="no-cuda"
      ;;
    r)
      BUILD_ARG="remote"
	  ;;
	a)
      ARCH="arm64"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Define the image name
IMAGE_NAME="colab-vision-$BUILD_ARG-$CUDA_STATE"

# Get the name of the right Dockerfile to build with
DOCKERFILE="Dockerfile.$CUDA_STATE"

# Check if the image exists
IMAGE_EXISTS=$(docker images -q $IMAGE_NAME)

# Build the image if it does not exist
if [ -z "$IMAGE_EXISTS" ]; then
    echo "Building a $BUILD_ARG image using $DOCKERFILE..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE --build-arg branch=$BUILD_ARG .
else
    echo "Docker image for $BUILD_ARG already exists on this machine, skipping build."
fi

# Run the container with the appropriate port mapping
docker run -p 3001:3000 --name "$IMAGE_NAME-instance" $IMAGE_NAME

