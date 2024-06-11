# Use an official Python runtime as a parent image
FROM python:3.11.7

# Copy the entire project into the container
COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx openssh-client openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Set PYTHONPATH environment variable
ENV PYTHONPATH="/src"

# Install build dependencies and the tracr module from the local source
RUN pip install setuptools wheel && pip install .

# Expose the necessary ports
EXPOSE 9000

# Copy the entry point script into the container
COPY entrypoint.sh entrypoint.sh

# Make the entry point script executable
RUN chmod +x entrypoint.sh

# Set the entry point to the entry point script
ENTRYPOINT ["entrypoint.sh"]
