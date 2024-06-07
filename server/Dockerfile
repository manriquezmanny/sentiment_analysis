# Defining my runtime and version
FROM python:3.10.14

# Setting the working directory
WORKDIR /app

# Copying the requirements filea nd installing dependencies
COPY requirements.txt /app

# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copying the current directory contents into container at /app
COPY ./app

# Setting up env variables for paths to model and vectorizer_models
ENV MODELS_DIR=/app/models
ENV VECTORIZER_MODELS_DIR=/app/vectorizer_models

# Exposing port
EXPOSE 5000

# Command to run the flask application
CMD ["python", "main.py"]
