# Use an official Python runtime as a parent image
FROM python:3.11.7

# Set the working directory
WORKDIR /app

# Copy the entire project into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx openssh-client openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies and the tracr module from the local source
RUN pip install setuptools wheel && pip install /app

# Expose the necessary ports
EXPOSE 9000

# Copy entrypoint script into the container
COPY entrypoint.sh /app/entrypoint.sh

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint to the script
ENTRYPOINT ["/app/entrypoint.sh"]
