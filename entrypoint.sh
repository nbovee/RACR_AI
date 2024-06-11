#!/bin/bash

# Function to install torch and torchvision if needed
install_torch() {
  echo "Installing torch and torchvision..."
  pip install torch==1.10.0 torchvision==0.11.1
}

# Check if the observer node is being run and do not install torch if it is
if [[ "$1" == "python" && "$2" == "/src/tracr/app_api/deploy.py" ]]; then
  echo "Running observer node without installing torch."
else
  install_torch
fi

# Execute the passed command
exec "$@"
