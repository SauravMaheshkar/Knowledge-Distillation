# Use an alpine image
FROM ubuntu:22.04 AS builder

# metainformation
LABEL version="0.0.1"
LABEL maintainer="Saurav Maheshkar"
LABEL org.opencontainers.image.source = "https://github.com/SauravMaheshkar/Knowledge-Distillation"

# Helpers
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /code

# Essential Installs
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
		gcc \
		gfortran \
		libopenblas-dev \
		python3 \
		python3.10 \
		python3-pip \
		python3.10-dev \
		python3.10-venv \
		&& apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel isort
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
