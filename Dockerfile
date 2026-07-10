ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.28@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa
ARG GENERIC_DEVEL=docker.io/nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@sha256:0230b7f243483cb15969fa3cc724a9459599604427052fc2a0d4291c7c0647dd
ARG GENERIC_RUNTIME=docker.io/nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04@sha256:14f6d08d1cd4a96effbfe3101d0b56326f552c199d05e4979ee0bd616df5811b
ARG FA2_DEVEL=docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed
ARG FA2_RUNTIME=docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc
ARG LLAMA_TAG=b9949
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl


FROM ${UV_IMAGE} AS uvbin


FROM ${FA2_DEVEL} AS llama
ARG LLAMA_TAG
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp /tmp/llama \
    && cmake -S /tmp/llama -B /tmp/llama/build \
         -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/llama/build --target llama-quantize -j "$(nproc)" \
    && mkdir -p /out/bin /out/gguf \
    && cp /tmp/llama/build/bin/llama-quantize /out/bin/ \
    && cp -r /tmp/llama/convert_hf_to_gguf.py /tmp/llama/conversion /out/gguf/ \
    && cp -r /tmp/llama/gguf-py /out/gguf-py


FROM ${GENERIC_DEVEL} AS generic-builder
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uvbin /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON=/usr/bin/python3.12 UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY variants/generic/pyproject.toml variants/generic/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
COPY --from=llama /out/gguf-py /tmp/gguf-py
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /opt/venv /tmp/gguf-py


FROM ${FA2_DEVEL} AS fa2-builder
ARG FLASH_ATTN_WHEEL
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uvbin /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON=/usr/bin/python3.12 UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY variants/fa2/pyproject.toml variants/fa2/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /opt/venv "${FLASH_ATTN_WHEEL}"
COPY --from=llama /out/gguf-py /tmp/gguf-py
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /opt/venv /tmp/gguf-py


FROM ${FA2_RUNTIME} AS fa2
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates tini libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uvbin /uv /uvx /bin/
COPY --chown=1000:1000 --from=fa2-builder /opt/venv /opt/venv
COPY --from=llama /out/gguf /opt/gguf
COPY --from=llama /out/bin/llama-quantize /usr/local/bin/llama-quantize
RUN install -d -o 1000 -g 1000 /home/trainer
ENV HOME=/home/trainer \
    PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    UV_LINK_MODE=copy \
    HF_HOME=/home/trainer/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_OFFLINE=0 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0"
USER 1000
WORKDIR /home/trainer
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]


FROM ${GENERIC_RUNTIME} AS generic
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates tini libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uvbin /uv /uvx /bin/
COPY --chown=1000:1000 --from=generic-builder /opt/venv /opt/venv
COPY --from=llama /out/gguf /opt/gguf
COPY --from=llama /out/bin/llama-quantize /usr/local/bin/llama-quantize
RUN install -d -o 1000 -g 1000 /home/trainer
ENV HOME=/home/trainer \
    PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    UV_LINK_MODE=copy \
    HF_HOME=/home/trainer/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_OFFLINE=0 \
    TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;12.0"
USER 1000
WORKDIR /home/trainer
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
