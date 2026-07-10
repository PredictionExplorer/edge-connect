FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_VERSION=1.93.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /build
COPY . .
RUN python3 -m venv /build/venv \
    && /build/venv/bin/pip install --upgrade pip \
    && mkdir /build/wheels \
    && /build/venv/bin/pip wheel --no-deps ./crates/star-py -w /build/wheels \
    && /build/venv/bin/pip wheel --no-deps . -w /build/wheels

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 starserve
COPY --from=builder /build/wheels /tmp/wheels
RUN python3 -m venv /opt/starserve \
    && /opt/starserve/bin/pip install --upgrade pip \
    && /opt/starserve/bin/pip install \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.34" \
    && /opt/starserve/bin/pip install /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels /root/.cache

ENV PATH="/opt/starserve/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STARSERVE_CONFIG=/config/starserve.yaml
USER starserve
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).read()"]
CMD ["sh", "-c", "exec starserve --config \"$STARSERVE_CONFIG\""]
