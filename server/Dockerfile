# Stage 1: Build stage
FROM python:3.10.14 AS builder

# Copy the entire project directory to the container
COPY . /app

# Stage 2: Final stage
FROM python:3.10.14

# Set the working directory
WORKDIR /app

# Copy only the necessary files from the builder stage
COPY --from=builder /app/models/model1.2.keras /app/models/
COPY --from=builder /app/vectorizer_models/vectorizer_model1.2 /app/vectorizer_models/
COPY --from=builder /app/main.py /app/
# Copy the requirements file and install dependencies
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "main.py"]
