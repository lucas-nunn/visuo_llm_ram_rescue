#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv-rocm/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
    echo "Missing ${VENV_PYTHON}. Run scripts/setup_rocm_tf_searchlight.sh first." >&2
    exit 1
fi

source "${ROOT_DIR}/scripts/rocm_tf_env.sh"
exec "${VENV_PYTHON}" "$@"
