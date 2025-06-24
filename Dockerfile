FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ARG MAMBA_VERSION=23.1.0-1
ARG TARGETPLATFORM

WORKDIR /opt

RUN chmod 777 -R /tmp && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    curl \
    g++ \
    make \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/ModelTC/light-tts.git
RUN cd light-tts && pip3 install -r requirements.txt
WORKDIR /opt/light-tts
