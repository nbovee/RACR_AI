#!/bin/bash

# deploy_node.sh - Runs (after building, if an image doesn't already exist) a
# client or remote node for cooperative DNN inference.
#
# USAGE:
#			./deploy_node.sh [FLAGS] SUFFIX
#
# FLAGS:
# 	-n		"no CUDA"	Use for hosts without CUDA support
#	-r		"remote"	Deploys a remote/server host
#	-a		"arm64"		Use for arm64 hosts (default is x86)
#	-t		"terminal"	Opens an interactive terminal session inside the container
#
# SUFFIX
#			The suffix is a required argument that tells the script how to find
#			which Dockerfile to use from the Dockerfiles directory.

# Default values if no flags are set
CUDA_STATE="cuda"
BUILD_ARG="client"
ARCH="x86"
TERMINAL="false"

# Parse through flags
while getopts ":nrat" opt; do
  case $opt in
    n)
      CUDA_STATE="nocuda"
      ;;
    r)
      BUILD_ARG="remote"
	  ;;
	a)
      ARCH="arm64"
      ;;
	t)
	  TERMINAL="true"
	  ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Remove the processed options
shift $((OPTIND-1))

# Check if a suffix was provided
if [ $# -eq 0 ]; then
    echo "No Dockerfile suffix provided. Please provide a suffix as a non-option argument."
    exit 1
fi

# Save the suffix
SUFFIX="$1"

# Infer the base image name
BASE_IMAGE_NAME="$ARCH-$CUDA_STATE-$BUILD_ARG-base"

# Get the name of the right Dockerfile to build with
DOCKERFILE="./Dockerfile.$SUFFIX"

# Check if the image was preloaded
BASE_IMAGE_PRELOADED=$(docker images -q $BASE_IMAGE_NAME)

# Load the image if it was not preloaded
if [ -z "$BASE_IMAGE_PRELOADED" ]; then
	PATH_TO_BASE_IMAGE_TARBALL="./docker_images/$BASE_IMAGE_NAME.tar"
    echo "Loading the $BASE_IMAGE_NAME image using $PATH_TO_BASE_IMAGE_TARBALL..."
	docker load -i $PATH_TO_BASE_IMAGE_TARBALL || echo "Problem loading image from tarball. Is it there?"
else
    echo "Base image for $BASE_IMAGE_NAME already exists on this machine, skipping load."
fi

# Build on top of the base image now    
docker build -t "$SUFFIX-$BUILD_ARG" -f $DOCKERFILE --build-arg branch=$BUILD_ARG .

# if the -t flag was passed, run the container in interactive terminal mode
if [ "$TERMINAL" = "true" ]; then
	docker run -p 3001:3000 -it --name "$BASE_IMAGE_NAME-instance" $BASE_IMAGE_NAME /bin/bash
else
	# Run the container normally and with the appropriate port mapping
	docker run -p 3001:3000 --name "$BASE_IMAGE_NAME-instance" $BASE_IMAGE_NAME
fi
