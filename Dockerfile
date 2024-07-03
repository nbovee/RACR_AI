# Use an official Python runtime as a parent image
FROM python:3.11.7

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx openssh-client openssh-server dos2unix && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire project into the container at /app
COPY . /app

# Convert all .sh files to Unix format
RUN find /app -type f -name "*.sh" -exec dos2unix {} +

# Make all .sh files executable
RUN find /app -type f -name "*.sh" -exec chmod +x {} +

# Install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Install the tracr package from local source
RUN pip install -e /app

# Expose the necessary ports
EXPOSE 9000

# Set the entrypoint to the script
ENTRYPOINT ["/app/entrypoint.sh"]