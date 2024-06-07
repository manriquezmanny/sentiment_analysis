FROM python:3.10.14

RUN pip install gunicorn

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

# Set environment variables for paths to model and vectorizer_models
ENV MODELS_DIR=/app/models
ENV VECTORIZER_MODELS_DIR=/app/vectorizer_models

CMD [ "gunicorn", "main:app", "-b", "0.0.0.0:8000"]