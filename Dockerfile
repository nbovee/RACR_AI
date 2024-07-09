# Use an official Python runtime as a parent image
FROM python:3.11.7

# Set the working directory
WORKDIR /app/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy the pyproject.toml and README.md first to leverage Docker cache
COPY pyproject.toml README.md ./

# Install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Install the tracr package from local source
RUN pip install .

# Copy the entire project into the container
COPY . /app/

# Expose the necessary ports
EXPOSE 9000
