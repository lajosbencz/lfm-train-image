#!/usr/bin/env bash
set -e
if [ -n "$PUBLIC_KEY" ]; then
  mkdir -p /root/.ssh
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
  for var in PATH VIRTUAL_ENV UV_LINK_MODE HF_HOME HF_HUB_ENABLE_HF_TRANSFER HF_XET_HIGH_PERFORMANCE HF_HUB_OFFLINE HF_TOKEN TORCH_CUDA_ARCH_LIST; do
    [ -n "${!var+x}" ] && printf '%s="%s"\n' "$var" "${!var}"
  done > /etc/environment
  chmod 600 /etc/environment
  ssh-keygen -A
  /usr/sbin/sshd
fi
exec "${@:-bash}"
