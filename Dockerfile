# Use an official Python runtime as a parent image
FROM python:3.11.7

# Copy the entire project into the root directory of the container
COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx openssh-client openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="/src"

# Install build dependencies and the tracr module from the local source
RUN pip install setuptools wheel && pip install .

# Install additional dependencies for the model test
COPY ./requirements-test.txt .
RUN pip install -r requirements-test.txt

# Expose the necessary ports
EXPOSE 9000

# Default command to run (can be overridden by docker run command)
CMD ["python", "app.py"]
