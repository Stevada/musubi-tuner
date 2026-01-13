#!/bin/bash
# FLUX.1 Kontext i2i LoRA Inference/Testing Script
# Generate images using trained FLUX LoRA

# ============================================================================
# Model Paths
# ============================================================================
DIT_PATH="/workspace/runpod-slim/ComfyUI/models/diffusion_models/flux1-kontext-dev.safetensors"
VAE_PATH="/workspace/runpod-slim/ComfyUI/models/vae/ae.sft"
TEXT_ENCODER1="/workspace/runpod-slim/ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors"
TEXT_ENCODER2="/workspace/runpod-slim/ComfyUI/models/text_encoders/clip_l.safetensors"

# ============================================================================
# LoRA Settings
# ============================================================================
# Path to your trained LoRA weight
# Update this to point to your trained LoRA checkpoint
LORA_WEIGHT="./output/flux_i2i_lora_demo_bj.safetensors"

# LoRA strength (0.0 to 1.5+)
# 1.0 = full strength, 0.5 = half strength, 1.5 = overcooked
LORA_MULTIPLIER=1.5

# ============================================================================
# Inference Settings
# ============================================================================
# Control/reference image (REQUIRED for FLUX Kontext)
CONTROL_IMAGE="./dataset/control_images_test/profile.png"

# Prompt for generation
PROMPT="blowjob. The character is sucking the man's dick, looking at the camera."

# Output settings
OUTPUT_DIR="./test_outputs"
OUTPUT_NAME="flux_test_001"

# Image size (height width)
IMAGE_HEIGHT=512
IMAGE_WIDTH=512

# Inference steps (more = higher quality but slower)
# Recommended: 20-30 for FLUX
INFER_STEPS=25

# Random seed (for reproducibility)
# Use -1 for random seed each time
SEED=42

# Embedded CFG scale (distilled guidance)
# Default: 2.5, range typically 1.0-5.0
CFG_SCALE=2.5

# ============================================================================
# Create output directory
# ============================================================================
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Run Inference
# ============================================================================
echo "============================================"
echo "FLUX Kontext i2i LoRA Inference"
echo "============================================"
echo "LoRA: ${LORA_WEIGHT}"
echo "Control Image: ${CONTROL_IMAGE}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}.png"
echo "Prompt: ${PROMPT}"
echo "============================================"
echo ""

python src/musubi_tuner/flux_kontext_generate_image.py \
    --dit "${DIT_PATH}" \
    --vae "${VAE_PATH}" \
    --text_encoder1 "${TEXT_ENCODER1}" \
    --text_encoder2 "${TEXT_ENCODER2}" \
    --lora_weight "${LORA_WEIGHT}" \
    --lora_multiplier ${LORA_MULTIPLIER} \
    --control_image_path "${CONTROL_IMAGE}" \
    --prompt "${PROMPT}" \
    --image_size ${IMAGE_HEIGHT} ${IMAGE_WIDTH} \
    --infer_steps ${INFER_STEPS} \
    --seed ${SEED} \
    --embedded_cfg_scale ${CFG_SCALE} \
    --save_path "${OUTPUT_DIR}/${OUTPUT_NAME}.png" \
    --attn_mode xformers \
    --blocks_to_swap 10

# ============================================================================
# Memory Optimization Options
# ============================================================================
# If you encounter OOM during inference, uncomment these:
#
# For 24GB VRAM (RTX 4090):
# --fp8_scaled --fp8 \
#
# For 16GB VRAM (RTX 4080):
# --fp8_scaled --fp8 \
# --blocks_to_swap 15 \
#
# For 12GB VRAM (RTX 4070):
# --fp8_scaled --fp8 \
# --blocks_to_swap 20 \

# ============================================================================
# Advanced Options
# ============================================================================
# Additional flags you can add:
#
# --no_resize_control        # Don't resize control image (use original size)
# --no_metadata              # Don't save metadata in output image
# --flow_shift 1.0           # Adjust noise schedule (experimental)
# --output_type latent       # Save latent instead of decoded image
# --interactive              # Enable interactive mode for multiple prompts
# --from_file prompts.txt    # Read prompts from file (one per line)

echo ""
echo "============================================"
echo "✓ Inference completed!"
echo "============================================"
echo "Output saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}.png"
echo ""
