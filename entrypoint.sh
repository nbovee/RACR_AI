#!/bin/bash

# Function to install torch and torchvision if needed
install_torch() {
  echo "Installing torch and torchvision..."
  pip install torch==1.10.0 torchvision==0.11.1
}

# Check the role and install dependencies accordingly
if [[ "$1" == "observer" ]]; then
  echo "Setting up observer node..."
  # Add any observer-specific setup here
elif [[ "$1" == "participant" ]]; then
  echo "Setting up participant node..."
  install_torch
  # Add any participant node-specific setup here
else
  echo "Unknown role. Please specify 'observer' or 'participant'."
  exit 1
fi

# Shift out the first argument (role) and execute the remaining command
shift
exec "$@"