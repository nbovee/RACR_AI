FROM python:3.11.7

WORKDIR /usr/src/tracr/

# Copy the requirements file into the container
COPY ./requirements.txt .

# Install dependencies and libGL
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the necessary ports
EXPOSE 9000

# Command to run the application
CMD ["python", "app.py"]
