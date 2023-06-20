#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_client_container.sh <HOST_IP> <PORT>"
    exit 1
fi

HOST_IP=$1
PORT=$2

docker run -p 8893:8893 -v ./test_results:/usr/src/app/test_results -e HOST_IP=$HOST_IP -e PORT=$PORT nocuda-client-oas