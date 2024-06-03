# Use an official Python runtime as a parent image
FROM python:3.11.7

# Copy the rest of the application code into the container
COPY . .

# Install tracr package and its dependencies from the local source
RUN pip install .

# Expose the necessary ports
EXPOSE 9000

# Default command to run (can be overridden by docker run command)
CMD ["python", "app.py"]
