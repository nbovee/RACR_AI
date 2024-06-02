#!/bin/bash

# deploy_node.sh - Runs (after building, if an image doesn't already exist) a
# client or remote node for cooperative DNN inference.
#
# USAGE:
#			./deploy_node.sh [FLAGS] ROLE
#
# FLAGS:
# 	-t		"terminal"	Opens an interactive terminal session inside the container
#
# ROLE
#			The role is a required argument that tells the script which role to run (observer or dummy).

# Default values if no flags are set
TERMINAL="false"

# Parse through flags
while getopts ":t" opt; do
  case $opt in
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

# Check if a role was provided
if [ $# -eq 0 ]; then
    echo "No role provided. Please provide a role (observer or dummy) as a non-option argument."
    exit 1
fi

# Save the role
ROLE="$1"

# Build the Docker image
docker build -t tracr-app .

# Set the command based on the role
if [ "$ROLE" = "observer" ]; then
  CMD="python src/app_api/deploy.py"
elif [ "$ROLE" = "dummy" ]; then
  CMD="python src/experiment_design/services/basic_split_inference.py"
else
  echo "Invalid role: $ROLE. Please provide either 'observer' or 'dummy'."
  exit 1
fi

# Run the container
if [ "$TERMINAL" = "true" ]; then
  docker run -p 9000:9000 -it --name tracr-$ROLE tracr-app /bin/bash
else
  docker run -p 9000:9000 --name tracr-$ROLE tracr-app $CMD
fi
