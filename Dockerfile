FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
LABEL maintainer="bigscience-workshop"
LABEL repository="drift"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  curl \
  git \
  ca-certificates \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Install uv (manages the Python toolchain and dependencies)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

VOLUME /cache
ENV DRIFT_CACHE=/cache
ENV UV_LINK_MODE=copy

COPY . drift/
WORKDIR /home/drift/
RUN uv sync --extra dev --python 3.11
# Put the project venv on PATH so `python -m drift.cli.*` works directly
ENV PATH="/home/drift/.venv/bin:${PATH}"

CMD bash
