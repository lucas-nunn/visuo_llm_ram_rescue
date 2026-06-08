#!/usr/bin/env bash

if [ -z "${ROCM_PATH:-}" ]; then
    for candidate in /opt/rocm-7.2.4 /opt/rocm /opt/rocm-*; do
        if [ -d "${candidate}" ]; then
            export ROCM_PATH="${candidate}"
            break
        fi
    done
fi

if [ -z "${ROCM_PATH:-}" ]; then
    echo "Could not find ROCm. Set ROCM_PATH before running." >&2
    return 1 2>/dev/null || exit 1
fi

export HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_cuda_data_dir=${ROCM_PATH}}"
