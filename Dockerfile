ARG CUDA_DEVEL=docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed
ARG CUDA_RUNTIME=docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.14@sha256:1025398289b62de8269e70c45b91ffa37c373f38118d7da036fb8bb8efc85d97
ARG LLAMA_TAG=b9948
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl


FROM ${CUDA_DEVEL} AS builder
ARG UV_IMAGE
ARG LLAMA_TAG

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=${UV_IMAGE} /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

RUN git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp /tmp/llama \
    && cmake -S /tmp/llama -B /tmp/llama/build \
         -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/llama/build --target llama-quantize -j "$(nproc)" \
    && uv pip install --python /opt/venv /tmp/llama/gguf-py \
    && mkdir -p /opt/gguf && cp -r /tmp/llama/convert_hf_to_gguf.py /tmp/llama/conversion /opt/gguf/


FROM ${CUDA_RUNTIME} AS runtime-base
ARG UV_IMAGE

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 ca-certificates tini libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=${UV_IMAGE} /uv /uvx /bin/
COPY --chown=1000:1000 --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/gguf /opt/gguf
COPY --from=builder /tmp/llama/build/bin/llama-quantize /usr/local/bin/llama-quantize

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


FROM runtime-base AS fa2
ARG FLASH_ATTN_WHEEL
RUN uv pip install --python /opt/venv "${FLASH_ATTN_WHEEL}"


FROM runtime-base AS generic
