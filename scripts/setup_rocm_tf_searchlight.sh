#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-rocm"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

uv venv --allow-existing --python "${PYTHON_VERSION}" "${VENV_DIR}"
uv pip install --python "${VENV_DIR}/bin/python" -r "${ROOT_DIR}/requirements-rocm-searchlight.txt"
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/rocm_tf_env.sh"

"${VENV_DIR}/bin/python" -c 'import tensorflow as tf; print("TensorFlow:", tf.__version__); print("GPUs:", tf.config.list_physical_devices("GPU"))'
