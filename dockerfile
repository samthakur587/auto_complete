# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app


COPY . .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container

EXPOSE 8000
# Set the command to run the application
CMD ["python","api.py"]