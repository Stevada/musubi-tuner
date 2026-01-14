#!/bin/bash

# Z-Image-Edit Training Script with Dual Pathway Control
# This script trains a LoRA for Z-Image-Edit using control/reference images
#
# Usage:
#   bash zimage_edit/train.sh
#
# Before running:
#   1. Prepare dataset with control images
#   2. Cache latents (target and control)
#   3. Update paths below

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Accelerate configuration
ACCELERATE_CONFIG="config/accelerate/default.yaml"

# Dataset
DATASET_CONFIG="config/datasets/zimage_edit_example.toml"

# Model paths
DIT_PATH="models/z-image-dit.safetensors"
VAE_PATH="models/z-image-vae.safetensors"
TEXT_ENCODER="models/Qwen2.5-VL"  # Vision-language model (replaces Qwen3)

# Training parameters
OUTPUT_DIR="output/zimage_edit"
OUTPUT_NAME="zimage_edit_lora"
NETWORK_MODULE="networks.lora"
NETWORK_DIM=32
NETWORK_ALPHA=16

# Optimizer
LEARNING_RATE=1e-4
OPTIMIZER_TYPE="adamw8bit"
LR_SCHEDULER="cosine_with_restarts"
LR_WARMUP_STEPS=100

# Batch and steps
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4
MAX_STEPS=10000
SAVE_EVERY=1000
SAMPLE_EVERY=500

# Memory optimizations
MIXED_PRECISION="bf16"
FP8_VL=true              # Use fp8 for VLM to save memory (recommended)
FP8_SCALED=false         # Use fp8 for DiT (optional, experimental)
GRADIENT_CHECKPOINTING=true
ATTN_MODE="flash"        # Options: flash, sage, xformers, torch

# Sample prompts for validation
SAMPLE_PROMPTS="config/sample_prompts_zimage_edit.txt"

# Logging
LOG_WITH="tensorboard"
LOGGING_DIR="${OUTPUT_DIR}/logs"

# ============================================================================
# Setup
# ============================================================================

echo "===================================="
echo "Z-Image-Edit Training"
echo "===================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Network: ${NETWORK_MODULE} (dim=${NETWORK_DIM}, alpha=${NETWORK_ALPHA})"
echo "Learning rate: ${LEARNING_RATE}"
echo "Batch size: ${BATCH_SIZE} (gradient accumulation: ${GRADIENT_ACCUMULATION})"
echo "Max steps: ${MAX_STEPS}"
echo "Memory optimizations:"
echo "  - Mixed precision: ${MIXED_PRECISION}"
echo "  - FP8 VLM: ${FP8_VL}"
echo "  - FP8 Scaled DiT: ${FP8_SCALED}"
echo "  - Gradient checkpointing: ${GRADIENT_CHECKPOINTING}"
echo "  - Attention mode: ${ATTN_MODE}"
echo "===================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGGING_DIR}"

# Check if dataset config exists
if [ ! -f "${DATASET_CONFIG}" ]; then
    echo "Error: Dataset config not found: ${DATASET_CONFIG}"
    echo "Please create a dataset configuration file."
    exit 1
fi

# Check if models exist
if [ ! -f "${DIT_PATH}" ]; then
    echo "Error: DiT model not found: ${DIT_PATH}"
    exit 1
fi

if [ ! -f "${VAE_PATH}" ]; then
    echo "Error: VAE model not found: ${VAE_PATH}"
    exit 1
fi

if [ ! -d "${TEXT_ENCODER}" ]; then
    echo "Error: Text encoder (Qwen2.5-VL) not found: ${TEXT_ENCODER}"
    exit 1
fi

# ============================================================================
# Training
# ============================================================================

echo ""
echo "Starting training..."
echo ""

# Build optional arguments
OPTIONAL_ARGS=""
[ "${FP8_VL}" = true ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --fp8_vl"
[ "${FP8_SCALED}" = true ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --fp8_scaled"
[ "${GRADIENT_CHECKPOINTING}" = true ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --gradient_checkpointing"

# Run training with accelerate
accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_cpu_threads_per_process 1 \
    --mixed_precision "${MIXED_PRECISION}" \
    src/musubi_tuner/zimage_edit_train_network.py \
    --dit "${DIT_PATH}" \
    --vae "${VAE_PATH}" \
    --text_encoder "${TEXT_ENCODER}" \
    --dataset_config "${DATASET_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "${OUTPUT_NAME}" \
    --network_module "${NETWORK_MODULE}" \
    --network_dim ${NETWORK_DIM} \
    --network_alpha ${NETWORK_ALPHA} \
    --learning_rate ${LEARNING_RATE} \
    --train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --max_train_steps ${MAX_STEPS} \
    --sample_every_n_steps ${SAMPLE_EVERY} \
    --sample_prompts "${SAMPLE_PROMPTS}" \
    --save_every_n_steps ${SAVE_EVERY} \
    --save_model_as safetensors \
    --mixed_precision "${MIXED_PRECISION}" \
    --optimizer_type "${OPTIMIZER_TYPE}" \
    --lr_scheduler "${LR_SCHEDULER}" \
    --lr_warmup_steps ${LR_WARMUP_STEPS} \
    --attn_mode "${ATTN_MODE}" \
    --split_attn \
    --log_with "${LOG_WITH}" \
    --logging_dir "${LOGGING_DIR}" \
    ${OPTIONAL_ARGS}

# ============================================================================
# Completion
# ============================================================================

echo ""
echo "===================================="
echo "Training complete!"
echo "===================================="
echo "LoRA weights saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors"
echo "Logs available in: ${LOGGING_DIR}"
echo ""
echo "To generate images with the trained LoRA, run:"
echo "  bash zimage_edit/inference.sh"
echo "===================================="
