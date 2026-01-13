#!/bin/bash
# FLUX.1 Kontext Dataset Caching Script
# Run this BEFORE training to cache latents and text encoder outputs

# ============================================================================
# Model Paths - Update these to match your setup
# ============================================================================
VAE_PATH="/workspace/runpod-slim/ComfyUI/models/vae/ae.sft"
TEXT_ENCODER1="/workspace/runpod-slim/ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors"
TEXT_ENCODER2="/workspace/runpod-slim/ComfyUI/models/text_encoders/clip_l.safetensors"
DATASET_CONFIG="/root/musubi-tuner/flux_dataset_config.toml"

# ============================================================================
# Step 1: Cache Latents
# ============================================================================
echo "============================================"
echo "Step 1: Caching VAE latents..."
echo "============================================"

python src/musubi_tuner/flux_kontext_cache_latents.py \
    --dataset_config "${DATASET_CONFIG}" \
    --vae "${VAE_PATH}"

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

python src/musubi_tuner/flux_kontext_cache_text_encoder_outputs.py \
    --dataset_config "${DATASET_CONFIG}" \
    --text_encoder1 "${TEXT_ENCODER1}" \
    --text_encoder2 "${TEXT_ENCODER2}" \
    --batch_size 16

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
echo "You can now run training with: bash flux_train.sh"
echo ""
