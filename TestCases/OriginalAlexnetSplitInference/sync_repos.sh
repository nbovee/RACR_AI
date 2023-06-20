#!/bin/bash

# List the IP addresses of the Raspberry Pis
RPI_IPS=("192.168.1.201" "192.168.1.202" "192.168.1.203")

# Define the source and target directories
SOURCE_DIR="."
TARGET_DIR="/home/racr/RACR_AI"

# Loop through the IP addresses and sync the repositories
for ip in "${RPI_IPS[@]}"; do
  echo "Syncing with Raspberry Pi at $ip..."
  rsync -avz -e ssh --delete --progress "$SOURCE_DIR" "racr@$ip:$TARGET_DIR"
done

echo "Sync completed."

