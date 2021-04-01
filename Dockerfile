FROM ghcr.io/sinead-k-doherty/base-ml-image:latest

RUN apt-get update && \
    apt-get install -y build-essential cmake \
    ffmpeg libsm6 libxext6

RUN pip install poetry

RUN mkdir /fer_model

COPY pyproject.toml poetry.lock /fer_model/

WORKDIR /fer_model

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-root

COPY app/ /fer_model/app/
