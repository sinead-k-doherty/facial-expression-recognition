FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y build-essential cmake \
    ffmpeg libsm6 libxext6

RUN pip install poetry==1.1.0rc1

RUN mkdir /fer_model

COPY pyproject.toml poetry.lock /fer_model/

WORKDIR /fer_model

RUN poetry config virtualenvs.create false \
    && poetry install

COPY app/ /fer_model/app/

COPY test_images/ /fer_model/test_images/

EXPOSE 5000

# CMD [".venv/bin/python", "./app/main.py"]
