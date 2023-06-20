
  # docker run --gpus all --name "grpc_server" -t --mount type=bind,source="F:\Nick\Documents\Code\Work\Summer Research\collab-vision",target=/app nvcr.io/nvidia/pytorch:21.08-py3 nvidia-smi
  # docker run --gpus all --name "grpc_server" --memory=16g --oom-kill-disable -t --mount type=bind,source="F:\Nick\Documents\Code\Work\Summer Research\collab-vision",target=/app nvcr.io/nvidia/pytorch:21.08-py3
  # docker run --name "grpc_client" --memory=4g --oom-kill-disable -t --mount type=bind,source="F:\Nick\Documents\Code\Work\Summer Research\collab-vision",target=/app nvcr.io/nvidia/pytorch:21.08-py3
  # arm64 pytorch l4t-pytorch:r32.7.1-pth1.10-py3
  # wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
  # pip install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
