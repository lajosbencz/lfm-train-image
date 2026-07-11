# lfm-train-image

Optimized base image for training [LiquidAI LFM2.5](https://www.liquid.ai/) models.
It ships a lean HuggingFace training runtime pinned to LFM2.5's requirements, nothing else.
How you deploy it and what you train is up to you.

Published to `ghcr.io/lajosbencz/lfm-train`.

## Variants

Four purpose-built images per release - pick at deploy time, not at runtime:

| Tag | Torch / CUDA | Attention | Tuned for |
|---|---|---|---|
| `:<tag>` **generic** | latest (torch 2.13 / cu130, CUDA 13) | PyTorch **SDPA** | SFT and general training. Rides the newest torch/CUDA; no `flash-attn` package. |
| `:<tag>-fa2` | locked (torch 2.8 / cu128, CUDA 12.8) | **`flash_attention_2`** | **CPT with sequence packing** - flash-attn's varlen kernels skip pad/cross-doc compute. |
| `:<tag>-ssh` | same as `generic` | PyTorch **SDPA** | `generic` + sshd, root login via a `PUBLIC_KEY` env var, for remote-managed deployments. |
| `:<tag>-ssh-fa2` | same as `fa2` | **`flash_attention_2`** | `fa2` + sshd, root login via a `PUBLIC_KEY` env var, for remote-managed deployments. |

The variants are **version-decoupled on purpose**: only `-fa2` (and `-ssh-fa2`) is pinned to torch 2.8 (the newest torch with a prebuilt flash-attn wheel).
`generic` (and `-ssh`) is free to track the latest torch/CUDA because SDPA's attention backend is built into PyTorch and needs no external package.

## What's inside

Both variants share the training stack; only torch/CUDA and the attention package differ.

- Python 3.12, **non-root** (uid 1000).
- **Training stack**: `transformers 5.x`, `trl`, `peft`, `accelerate`, `datasets`, `bitsandbytes`, `huggingface-hub`, `hf-transfer`, `safetensors`, `sentencepiece`.
- **GGUF export** (CPU): `llama-quantize` (llama.cpp `b9949`, static) on `PATH`, plus `convert_hf_to_gguf.py` + `gguf` - supports `lfm2` and `lfm2moe`.
- `generic`: CUDA 13.0.3 base, `torch 2.13.0+cu130`.
- `fa2`: CUDA 12.8.1 base, `torch 2.8.0+cu128`, `flash-attn 2.8.3.post1` (prebuilt wheel).

Each variant's Python stack is pinned in `variants/<name>/uv.lock`;
base images and the `uv` image are pinned by digest in the `Dockerfile`.

## Design

- Neutral entrypoint (`tini` → `bash`) on `generic`/`fa2` - no baked services, no secrets. Supply your own command.
- `transformers >= 5.0` is required for LFM2.5; no `trust_remote_code` needed for the text models.
- `fa2` torch is pinned to `2.8.0` so its prebuilt FlashAttention wheel matches (no in-image compile).
- GGUF conversion is CPU work - bundled for convenience, not run on the training GPU by the image.
- The `-ssh`/`-ssh-fa2` tags are a deliberate, scoped exception to the uid-1000 non-root design:
  sshd requires root, and root-login-over-SSH is the standard convention for remote-managed pods.
  sshd only starts if `PUBLIC_KEY` is injected at runtime; key-only auth, no password/interactive
  fallback (`sshd-hardening.conf`); host keys are generated fresh per container start, never baked.

## Runtime configuration

Defaults ship as environment variables; any value you pass at run time wins.

| Env | Default | Purpose |
|---|---|---|
| `HF_HOME` | `/home/trainer/.cache/huggingface` | Model/dataset/hub cache. Point at a mounted persistent volume to keep weights across runs. |
| `HF_HUB_OFFLINE` | `0` | Set `1` once cached to skip network. |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Fast parallel downloads. |
| `HF_TOKEN` | *(unset)* | Inject at run time for gated models - never baked. |
| `TORCH_CUDA_ARCH_LIST` | per variant | Narrow if targeting fewer GPUs. |
| `PUBLIC_KEY` | *(unset)* | `-ssh`/`-ssh-fa2` only. SSH public key for root login; sshd doesn't start without it. |

Model names, dataset paths, output dirs, and hyperparameters are your training scripts' concern - the image passes your command and env straight through.

## Usage

Deps are already baked; install only your package:

```
docker run --rm -it --gpus all \
  -v "$PWD":/work -w /work \
  -v hf-cache:/home/trainer/.cache/huggingface \
  -e HF_TOKEN \
  ghcr.io/lajosbencz/lfm-train:v1 \
  bash -lc 'uv pip install -e . --no-deps && python scripts/train.py'
```

Use `:v1-fa2` for packed CPT with `attn_implementation="flash_attention_2"`;
use `:v1` (generic) with `attn_implementation="sdpa"` otherwise.

Export a trained checkpoint to GGUF (either variant):

```
python /opt/gguf/convert_hf_to_gguf.py <checkpoint> --outfile f16.gguf --outtype f16
llama-quantize f16.gguf model-Q4_K_M.gguf Q4_K_M
```

For `:v1-ssh` / `:v1-ssh-fa2`, set `PUBLIC_KEY` and publish port 22:

```
docker run --rm --gpus all -p 2222:22 -e PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
  ghcr.io/lajosbencz/lfm-train:v1-ssh-fa2
```

then `ssh root@<host> -p 2222`.

## Build

```
docker build --target generic     -t ghcr.io/lajosbencz/lfm-train:dev .
docker build --target fa2         -t ghcr.io/lajosbencz/lfm-train:dev-fa2 .
docker build --target generic-ssh -t ghcr.io/lajosbencz/lfm-train:dev-ssh .
docker build --target fa2-ssh     -t ghcr.io/lajosbencz/lfm-train:dev-ssh-fa2 .
```

The CPU llama.cpp tools are built once (the `llama` stage) and copied into both variants.

### Pins

All upstream image and tool pins live in **one `ARG` block at the top of the `Dockerfile`** - `GENERIC_DEVEL` / `GENERIC_RUNTIME`, `FA2_DEVEL` / `FA2_RUNTIME`, `UV_IMAGE` (each `tag@sha256:…`), `LLAMA_TAG`, `FLASH_ATTN_WHEEL`.
That is the single place to bump them; override at build time with `--build-arg` for a one-off.

GitHub Actions in `.github/workflows/build.yml` are pinned by commit SHA with a version comment. Resolve a fresh base digest with:

```
skopeo inspect docker://nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04 | jq -r .Digest
```

The Python stack per variant is pinned in `variants/<name>/uv.lock` (`cd variants/<name> && uv lock` to refresh).
The **one** cross-file coupling:
`fa2`'s `torch==2.8.0` (in `variants/fa2/pyproject.toml`) and the `FLASH_ATTN_WHEEL` tag `…torch2.8…` (in the `Dockerfile`) must move together - a prebuilt flash-attn wheel exists only for specific torch minors.

## Publishing

Push a `v*` tag.
`.github/workflows/build.yml` builds all four variants, gates each on a Trivy scan (CRITICAL/HIGH, fixed only), then pushes `ghcr.io/lajosbencz/lfm-train:<tag>` (generic), `:<tag>-fa2`, `:<tag>-ssh`, and `:<tag>-ssh-fa2`.

## License

MIT - see [LICENSE](LICENSE).

> This repository was substantially authored with Claude Opus 4.8 (Anthropic), under human direction and review.
