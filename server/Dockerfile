# Defining my runtime and version
FROM python:3.10.14 AS builder

#Copying the current directory contents into container at /app
COPY . /app

# Defining my runtime and version
FROM python:3.10.14

# Setting the working directory
WORKDIR /app

# Adding a step to copy the model file to the correct directory
COPY --from=builder /app/models/model1.2.keras /app/models/
COPY --from=builder /app/vectorizer_models/vectorizer_model1.2 /app/vectorizer_models/vectorizer_model1.2

# Copying the requirements filea nd installing dependencies
COPY requirements.txt /app


# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Exposing port
EXPOSE 5000

# Command to run the flask application
CMD ["python", "main.py"]
