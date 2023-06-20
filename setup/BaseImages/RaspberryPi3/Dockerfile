# using slim Python 3.7 image as base to keep size down
FROM arm64v8/python:3.7-slim-buster AS base

WORKDIR /app

COPY ./requirements.txt .

RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==1.10.0 torchvision==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
