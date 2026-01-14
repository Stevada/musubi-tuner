#!/bin/bash
# Qwen-Image Dataset Caching Script
# Run this BEFORE training to cache latents and text encoder outputs
# Supports all 5 Qwen variants: original, edit, edit-2509, edit-2511, layered

# ============================================================================
# Model Version Selection
# ============================================================================
# Valid options: original, edit, edit-2509, edit-2511, layered
# Default: edit-2511 (latest stable version with best features)
MODEL_VERSION="edit-2511"

# ============================================================================
# Model Paths - Update these to match your setup
# ============================================================================
# VAE paths (layered variant uses a different VAE)
# Download from: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI or Qwen-Image-Edit_ComfyUI
VAE_PATH="/workspace/runpod-slim/ComfyUI/models/vae/qwen_image_vae.safetensors"
VAE_PATH_LAYERED="/workspace/runpod-slim/ComfyUI/models/vae/qwen_image_layered_vae.safetensors"

# Text Encoder (Qwen2.5-VL, shared across all variants)
# Download from: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI
TEXT_ENCODER="/workspace/runpod-slim/ComfyUI/models/text_encoders/qwen_2.5_vl_7b.safetensors"

# Dataset configuration
DATASET_CONFIG="/root/musubi-tuner/qwen_dataset_config.toml"

# ============================================================================
# Auto-select VAE based on model version
# ============================================================================
if [ "$MODEL_VERSION" = "layered" ]; then
    SELECTED_VAE="$VAE_PATH_LAYERED"
else
    SELECTED_VAE="$VAE_PATH"
fi

# ============================================================================
# Step 1: Cache Latents
# ============================================================================
echo "============================================"
echo "Step 1: Caching VAE latents..."
echo "Model Version: ${MODEL_VERSION}"
echo "Using VAE: ${SELECTED_VAE}"
echo "============================================"

python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config "${DATASET_CONFIG}" \
    --vae "${SELECTED_VAE}" \
    --model_version "${MODEL_VERSION}"

if [ $? -ne 0 ]; then
    echo "ERROR: Latent caching failed!"
    exit 1
fi

echo ""
echo "✓ Latent caching completed successfully"
echo ""

# ============================================================================
# Step 2: Cache Text Encoder Outputs
# ============================================================================
echo "============================================"
echo "Step 2: Caching text encoder outputs..."
echo "============================================"

python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config "${DATASET_CONFIG}" \
    --text_encoder "${TEXT_ENCODER}" \
    --model_version "${MODEL_VERSION}" \
    --batch_size 1

# Optional: Uncomment the line below for <16GB VRAM
#    --fp8_vl

if [ $? -ne 0 ]; then
    echo "ERROR: Text encoder caching failed!"
    exit 1
fi

echo ""
echo "✓ Text encoder caching completed successfully"
echo ""

# ============================================================================
# Done
# ============================================================================
echo "============================================"
echo "✓ All caching completed successfully!"
echo "============================================"
echo ""
echo "You can now run training with: bash qwen/qwen_train.sh"
echo ""
echo "Note: For edit variants (edit, edit-2509, edit-2511), control images"
echo "are automatically cached. For layered variant, multiple target images"
echo "are cached. Make sure your dataset config is properly configured."
echo ""
