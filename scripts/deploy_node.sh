#!/bin/bash

# deploy_node.sh - Runs (after building, if an image doesn't already exist) a
# client or remote node for cooperative DNN inference. Here, "node" refers to a
# Docker container running on either a Jetson, Raspberry Pi, or PC, which
# communicates with other nodes on the network using the gRPC protocol.
#
# USAGE:
#			./deploy_node [FLAGS] SUFFIX
#
# FLAGS:
# 	-n		"no CUDA"	Use for hosts without CUDA support (like the Pi 3)
#	-r		"remote"	Deploys a remote/server host (rather than client)
#	-a		"arm64"		Use for arm64 hosts (default is x86)
#	-t		"terminal"	Opens an interactive terminal session inside the container
#
# SUFFIX
#			The suffix is a required argument that tells the script how to find
#			which Dockerfile to use from the Dockerfiles directory. (The only
#			difference between the filenames is the suffix).
#
#			The base image has the correct Python and PyTorch dependencies installed;
#			all that's really left to do is specify the script you'd like to execute
#			when the container is run, including any extra setup you want to include
#			in the Dockerfile before that.
#
#			When choosing a suffix for your Dockerfile, try to make it descriptive
#			as it will be used to tag the final image as well.

# Default values if no flags are set (a client machine with CUDA and x86 arch)
CUDA_STATE="cuda"
BUILD_ARG="client"
ARCH="x86"
TERMINAL="false"

# parse through flags
while getopts ":nr" opt; do
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

# TODO: The script should check the machine's CUDA_STATE and ARCH itself,
#		either giving the user a warning if things don't match up, or even
#		making the flags unnecessary if the checking is robust enough.

# Infer the base image name (right now the convention is arch-cuda_state-build_arg-base
BASE_IMAGE_NAME="$ARCH-$CUDA_STATE-$BUILD_ARG-base"

# Get the name of the right Dockerfile to build with
DOCKERFILE="../Dockerfiles/Dockerfile.$SUFFIX"

# Check if the image was preloaded
BASE_IMAGE_PRELOADED=$(docker images -q $BASE_IMAGE_NAME)

# Load the image if it was not preloaded
if [ -z "$BASE_IMAGE_PRELOADED" ]; then
	PATH_TO_BASE_IMAGE_TARBALL="../docker_images/$BASE_IMAGE_NAME"
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

