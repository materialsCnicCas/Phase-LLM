#!/usr/bin/env bash
set -euo pipefail

# One-click SFT launcher for LLaMA-Factory.
# Usage:
#   cd /path/to/LLaMA-Factory
#   bash /path/to/Phase-LLM-Open-Source/2_Training_SFT/run_sft.sh
#   bash /path/to/Phase-LLM-Open-Source/2_Training_SFT/run_sft.sh 3
#   GPU_ID=3 bash /path/to/Phase-LLM-Open-Source/2_Training_SFT/run_sft.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SFT_CONFIG="${SCRIPT_DIR}/sft_config.yaml"
MERGE_CONFIG="${SCRIPT_DIR}/merge_lora.yaml"

# Fixed GPU id in script (edit this line if you want another GPU)
GPU_ID="3"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[Phase-LLM] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "[Phase-LLM] Starting SFT training with config: ${SFT_CONFIG}"
llamafactory-cli train "${SFT_CONFIG}"

echo "[Phase-LLM] Merging LoRA adapter with config: ${MERGE_CONFIG}"
llamafactory-cli export "${MERGE_CONFIG}"

echo "[Phase-LLM] SFT + merge completed successfully."
