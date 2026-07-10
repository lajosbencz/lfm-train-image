# lfm-train-image

Optimized base image for training [LiquidAI LFM2.5](https://www.liquid.ai/) models.
It ships a lean, generic HuggingFace training runtime pinned to LFM2.5's requirements, nothing else.
How you deploy it and what you train is up to you.

Published to `ghcr.io/lajosbencz/lfm-train`.

## Variants

Two images per release; pick by attention backend:

| Tag | Attention | Use it for |
|---|---|---|
| `:<tag>` (generic) | PyTorch **SDPA** - uses torch's built-in FlashAttention-2 backend | Default. SFT and any padded/non-packed training. Works on every GPU your torch build supports (incl. consumer Blackwell). No `flash-attn` package. |
| `:<tag>-fa2` | adds the **`flash-attn` package** for `attn_implementation="flash_attention_2"` | Packed / variable-length training (e.g. CPT with sequence packing), where flash-attn's varlen kernels skip pad/cross-doc compute. |

Both give exact FlashAttention-2 math;
for LFM2.5 (hybrid - only the GQA layers use attention) the difference is a modest speedup on packed runs.

When unsure, use generic with `attn_implementation="sdpa"`.

## What's inside (both variants)

- **CUDA 12.8** runtime (`nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`), Python 3.12.
- **PyTorch `2.8.0+cu128`** - spans RTX 4090 (sm_89), A100/L40S/RTX PRO 6000, H100/H200 (sm_90),
  B200 (sm_100).
- **Training stack**: `transformers 5.x`, `trl`, `peft`, `accelerate`, `datasets`, `bitsandbytes`,
  `huggingface-hub`, `hf-transfer`, `safetensors`, `sentencepiece`.
- **GGUF export** (CPU): `llama-quantize` (llama.cpp `b9948`, static) on `PATH`, plus
  `convert_hf_to_gguf.py` + `gguf` - supports `lfm2` and `lfm2moe`.

The `-fa2` variant additionally has `flash-attn 2.8.3.post1` (prebuilt wheel).

Exact versions are pinned in `pyproject.toml` / `uv.lock`; base images and the `uv` image are pinned by digest in the `Dockerfile`.

## Design

- Runs as **non-root** (uid 1000, `HOME=/home/trainer`). Neutral entrypoint (`tini` → `bash`) - no baked services, no SSH, no secrets. Supply your own command.
- `transformers >= 5.0` is required for LFM2.5; no `trust_remote_code` needed for the text models.
- PyTorch is pinned to `2.8.0` so the `-fa2` variant's prebuilt FlashAttention wheel matches (no in-image compile).
- GGUF conversion is CPU work - bundled for convenience, not run on the training GPU by the image.

## Runtime configuration

Defaults ship as environment variables; any value you pass at run time wins.

| Env | Default | Purpose |
|---|---|---|
| `HF_HOME` | `/home/trainer/.cache/huggingface` | Model/dataset/hub cache. Point at a mounted persistent volume to keep weights across runs. |
| `HF_HUB_OFFLINE` | `0` | Set `1` once cached to skip network. |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Fast parallel downloads. |
| `HF_TOKEN` | *(unset)* | Inject at run time for gated models - never baked. |
| `TORCH_CUDA_ARCH_LIST` | `8.0;8.6;8.9;9.0;10.0` | Narrow if targeting fewer GPUs. |

Model names, dataset paths, output dirs, and hyperparameters are your training scripts' concern, the image passes your command and env straight through.

## Usage

Run a project against it (deps are already baked; install only your package):

```
docker run --rm -it --gpus all \
  -v "$PWD":/work -w /work \
  -v hf-cache:/home/trainer/.cache/huggingface \
  -e HF_TOKEN \
  ghcr.io/lajosbencz/lfm-train:v1 \
  bash -lc 'uv pip install -e . --no-deps && python scripts/train.py'
```

Use `:<tag>-fa2` for packed training with `attn_implementation="flash_attention_2"`.

Export a trained checkpoint to GGUF (either variant):

```
python /opt/gguf/convert_hf_to_gguf.py <checkpoint> --outfile f16.gguf --outtype f16
llama-quantize f16.gguf model-Q4_K_M.gguf Q4_K_M
```

## Build

```
docker build --target generic -t ghcr.io/lajosbencz/lfm-train:dev .
docker build --target fa2     -t ghcr.io/lajosbencz/lfm-train:dev-fa2 .
```

`fa2` is `generic` plus the flash-attn wheel, so it reuses every shared layer. Omitting
`--target` builds `generic` (the default final stage).

### Pins

All upstream image and tool pins live in **one `ARG` block at the top of the `Dockerfile`** -
`CUDA_DEVEL`, `CUDA_RUNTIME`, `UV_IMAGE` (each `tag@sha256:…`), `LLAMA_TAG`, `FLASH_ATTN_WHEEL`.
That is the single place to bump them; override at build time with `--build-arg` for a one-off.
GitHub Actions in `.github/workflows/build.yml` are pinned by commit SHA with a version comment.
Resolve a fresh base digest with:

```
skopeo inspect docker://nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 | jq -r .Digest
```

The Python stack is pinned in `pyproject.toml` / `uv.lock` (`uv lock` to refresh).
The **one** cross-file coupling that can't be single-sourced: `torch` (in `pyproject.toml`) and the `FLASH_ATTN_WHEEL` tag `…torch2.8…` (in the `Dockerfile`) must move together.
A prebuilt flash-attn wheel exists only for specific torch minors.
Bump both or the `-fa2` build has no matching wheel.

## Publishing

Push a `v*` tag. `.github/workflows/build.yml` builds both variants, gates each on a Trivy scan (CRITICAL/HIGH, fixed only), then pushes `ghcr.io/lajosbencz/lfm-train:<tag>` (generic) and `:<tag>-fa2`.

## License

MIT — see [LICENSE](LICENSE).

> This repository was substantially authored with Claude Opus 4.8 (Anthropic), under human direction and review.
