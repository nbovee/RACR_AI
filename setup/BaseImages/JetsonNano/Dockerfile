FROM nvcr.io/nvidia/pytorch:21.08-py3 AS base

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
