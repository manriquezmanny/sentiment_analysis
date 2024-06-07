# Stage 1: Build stage
FROM python:3.10.14

# Copy the entire project directory to the container
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app

# Expose port 5000
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "main.py"]
