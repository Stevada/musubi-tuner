#!/bin/bash

# Z-Image-Edit Inference Script
# Generate images using Z-Image-Edit with control/reference images
#
# Usage:
#   bash zimage_edit/inference.sh
#
# Configuration:
#   - Edit the variables below to customize generation
#   - Set CONTROL_IMAGE to your reference/control image
#   - Adjust prompt, dimensions, steps, etc. as needed

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Model paths
DIT_PATH="models/z-image-dit.safetensors"
VAE_PATH="models/z-image-vae.safetensors"
TEXT_ENCODER="models/Qwen2.5-VL"
LORA_PATH="output/zimage_edit/zimage_edit_lora.safetensors"  # Optional

# Control/reference image
CONTROL_IMAGE="examples/control_images/reference.png"

# Generation parameters
PROMPT="A beautiful sunset over mountains, highly detailed, 8k"
NEGATIVE_PROMPT="blurry, low quality, distorted"
WIDTH=1024
HEIGHT=1024
STEPS=50
CFG_SCALE=4.0
SEED=42
SHIFT=1.0

# Output
OUTPUT_DIR="output/generated"
OUTPUT_NAME="generated.png"

# Memory optimizations
FP8_VL=true              # Use fp8 for VLM
FP8_SCALED=false         # Use fp8 for DiT
ATTN_MODE="flash"        # Options: flash, sage, xformers, torch

# Device
DEVICE="cuda"

# ============================================================================
# Setup
# ============================================================================

echo "===================================="
echo "Z-Image-Edit Inference"
echo "===================================="
echo "Control image: ${CONTROL_IMAGE}"
echo "Prompt: ${PROMPT}"
echo "Dimensions: ${WIDTH}x${HEIGHT}"
echo "Steps: ${STEPS}"
echo "CFG scale: ${CFG_SCALE}"
echo "Seed: ${SEED}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "===================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

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

if [ ! -f "${CONTROL_IMAGE}" ]; then
    echo "Error: Control image not found: ${CONTROL_IMAGE}"
    echo "Please provide a valid control/reference image path."
    exit 1
fi

# Check if LoRA exists (optional)
if [ -f "${LORA_PATH}" ]; then
    echo "Using LoRA weights: ${LORA_PATH}"
    LORA_ARG="--lora_weight ${LORA_PATH}"
else
    echo "No LoRA weights found (${LORA_PATH}), using base model only"
    LORA_ARG=""
fi

# ============================================================================
# Inference
# ============================================================================

echo ""
echo "Starting generation..."
echo ""

# Build optional arguments
OPTIONAL_ARGS=""
[ "${FP8_VL}" = true ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --fp8_vl"
[ "${FP8_SCALED}" = true ] && OPTIONAL_ARGS="${OPTIONAL_ARGS} --fp8_scaled"

# Run inference
python src/musubi_tuner/zimage_edit_generate_image.py \
    --dit "${DIT_PATH}" \
    --vae "${VAE_PATH}" \
    --text_encoder "${TEXT_ENCODER}" \
    ${LORA_ARG} \
    --control_image "${CONTROL_IMAGE}" \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --width ${WIDTH} \
    --height ${HEIGHT} \
    --num_inference_steps ${STEPS} \
    --guidance_scale ${CFG_SCALE} \
    --seed ${SEED} \
    --shift ${SHIFT} \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "${OUTPUT_NAME}" \
    --attn_mode "${ATTN_MODE}" \
    --device "${DEVICE}" \
    ${OPTIONAL_ARGS}

# ============================================================================
# Completion
# ============================================================================

echo ""
echo "===================================="
echo "Generation complete!"
echo "===================================="
echo "Image saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo ""
echo "To generate with different settings, edit this script or run:"
echo "  python src/musubi_tuner/zimage_edit_generate_image.py --help"
echo "===================================="
