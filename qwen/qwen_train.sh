#!/bin/bash
# Qwen-Image i2i LoRA Training Script
# Supports all 5 Qwen variants with comprehensive memory optimization

# ============================================================================
# Model Version Selection
# ============================================================================
# Valid options: original, edit, edit-2509, edit-2511, layered
# Default: edit-2511 (latest stable version with best features)
MODEL_VERSION="edit-2511"

# ============================================================================
# Model Paths - Update these to match your setup
# ============================================================================
# DiT paths for each variant
# Download from: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI
DIT_PATH_ORIGINAL="/workspace/runpod-slim/ComfyUI/models/diffusion_models/qwen_image_bf16.safetensors"
# Download from: https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI
DIT_PATH_EDIT="/workspace/runpod-slim/ComfyUI/models/diffusion_models/qwen_image_edit_bf16.safetensors"
DIT_PATH_EDIT_2509="/workspace/runpod-slim/ComfyUI/models/diffusion_models/qwen_image_edit_2509_bf16.safetensors"
DIT_PATH_EDIT_2511="/workspace/runpod-slim/ComfyUI/models/diffusion_models/qwen_image_edit_2511_bf16.safetensors"
# Download from: https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI
DIT_PATH_LAYERED="/workspace/runpod-slim/ComfyUI/models/diffusion_models/qwen_image_layered_bf16.safetensors"

# Auto-select DiT path based on MODEL_VERSION
case "$MODEL_VERSION" in
    original)
        DIT_PATH="$DIT_PATH_ORIGINAL"
        ;;
    edit)
        DIT_PATH="$DIT_PATH_EDIT"
        ;;
    edit-2509)
        DIT_PATH="$DIT_PATH_EDIT_2509"
        ;;
    edit-2511)
        DIT_PATH="$DIT_PATH_EDIT_2511"
        ;;
    layered)
        DIT_PATH="$DIT_PATH_LAYERED"
        ;;
    *)
        echo "ERROR: Invalid MODEL_VERSION: $MODEL_VERSION"
        echo "Valid options: original, edit, edit-2509, edit-2511, layered"
        exit 1
        ;;
esac

# VAE paths (layered variant uses a different VAE)
VAE_PATH="/workspace/runpod-slim/ComfyUI/models/vae/qwen_image_vae.safetensors"
VAE_PATH_LAYERED="/workspace/runpod-slim/ComfyUI/models/vae/qwen_image_layered_vae.safetensors"

# Auto-select VAE based on MODEL_VERSION
if [ "$MODEL_VERSION" = "layered" ]; then
    SELECTED_VAE="$VAE_PATH_LAYERED"
else
    SELECTED_VAE="$VAE_PATH"
fi

# Text Encoder (Qwen2.5-VL, shared across all variants)
TEXT_ENCODER="/workspace/runpod-slim/ComfyUI/models/text_encoders/qwen_2.5_vl_7b.safetensors"

# Dataset configuration
DATASET_CONFIG="/root/musubi-tuner/qwen_dataset_config.toml"

# Output settings
OUTPUT_DIR="./output"
OUTPUT_NAME="qwen_lora_${MODEL_VERSION}"

# ============================================================================
# Network Settings
# ============================================================================
NETWORK_DIM=32
NETWORK_ALPHA=16

# ============================================================================
# Training Parameters
# ============================================================================
LEARNING_RATE=1e-4
MAX_TRAIN_EPOCHS=16
SAVE_EVERY_N_EPOCHS=2

# ============================================================================
# Memory Optimization Settings
# ============================================================================
# Adjust these based on your VRAM (see Memory Optimization Guide below)
# For 24GB VRAM (RTX 4090): Use defaults below
# For 12GB VRAM (RTX 4070 Ti): Set BLOCKS_TO_SWAP=45
FP8_BASE=true
FP8_SCALED=true  # Only works with edit-2509, edit-2511, and layered
BLOCKS_TO_SWAP=0  # 0 = disabled, 16 = 24GB VRAM, 45 = 12GB VRAM

# ============================================================================
# Training Configuration
# ============================================================================
echo "============================================"
echo "Qwen-Image LoRA Training"
echo "============================================"
echo "Model Version: ${MODEL_VERSION}"
echo "DiT: ${DIT_PATH}"
echo "VAE: ${SELECTED_VAE}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "============================================"
echo ""

# Build memory optimization flags
MEMORY_FLAGS=""
if [ "$FP8_BASE" = true ]; then
    MEMORY_FLAGS="$MEMORY_FLAGS --fp8_base"
fi
if [ "$FP8_SCALED" = true ]; then
    # Warn if using fp8_scaled with incompatible variant
    if [ "$MODEL_VERSION" = "original" ] || [ "$MODEL_VERSION" = "edit" ]; then
        echo "WARNING: fp8_scaled only works with edit-2509, edit-2511, and layered"
        echo "Disabling fp8_scaled for $MODEL_VERSION variant"
        MEMORY_FLAGS="${MEMORY_FLAGS/--fp8_scaled/}"
    else
        MEMORY_FLAGS="$MEMORY_FLAGS --fp8_scaled"
    fi
fi
if [ "$BLOCKS_TO_SWAP" -gt 0 ]; then
    MEMORY_FLAGS="$MEMORY_FLAGS --blocks_to_swap $BLOCKS_TO_SWAP"
fi

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/qwen_image_train_network.py \
    --dit "${DIT_PATH}" \
    --vae "${SELECTED_VAE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --model_version "${MODEL_VERSION}" \
    --dataset_config "${DATASET_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "${OUTPUT_NAME}" \
    --network_module networks.lora_qwen_image \
    --network_dim ${NETWORK_DIM} \
    --network_alpha ${NETWORK_ALPHA} \
    --learning_rate ${LEARNING_RATE} \
    --optimizer_type adamw8bit \
    --max_train_epochs ${MAX_TRAIN_EPOCHS} \
    --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --xformers \
    --split_attn \
    --timestep_sampling shift \
    --discrete_flow_shift 2.2 \
    --weighting_scheme none \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --seed 42 \
    --logging_dir ./logs \
    --log_with tensorboard \
    $MEMORY_FLAGS

# ============================================================================
# Memory Optimization Guide
# ============================================================================
# Based on 1024x1024 training, batch size 1, bf16 + gradient_checkpointing + xformers
#
# Tier 1 (42GB+ VRAM - High-end workstation):
# - FP8_BASE=false
# - FP8_SCALED=false
# - BLOCKS_TO_SWAP=0
#
# Tier 2 (30GB VRAM - RTX A6000):
# - FP8_BASE=true
# - FP8_SCALED=true
# - BLOCKS_TO_SWAP=0
#
# Tier 3 (24GB VRAM - RTX 4090):
# - FP8_BASE=true
# - FP8_SCALED=true
# - BLOCKS_TO_SWAP=16
#
# Tier 4 (12GB VRAM - RTX 4070 Ti):
# - FP8_BASE=true
# - FP8_SCALED=true
# - BLOCKS_TO_SWAP=45
# - Requires 64GB+ system RAM
#
# IMPORTANT NOTES:
# - fp8_scaled ONLY works with edit-2509, edit-2511, and layered variants
# - fp8_scaled does NOT work with original or edit variants (will be auto-disabled)
# - Edit variants require additional VRAM for control images
# - Layered variant requires qwen_image_layered_vae.safetensors (different VAE)
# - For layered dataset, set multiple_target=true in your dataset config TOML
# - If blocks_to_swap > 45, main RAM usage will increase significantly
#
# ============================================================================
# Notes
# ============================================================================
# 1. Before training, you MUST run caching scripts:
#
#    bash qwen/qwen_cache.sh
#
#    Or manually run:
#    python src/musubi_tuner/qwen_image_cache_latents.py \
#        --dataset_config "${DATASET_CONFIG}" \
#        --vae "${SELECTED_VAE}" \
#        --model_version "${MODEL_VERSION}"
#
#    python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
#        --dataset_config "${DATASET_CONFIG}" \
#        --text_encoder "${TEXT_ENCODER}" \
#        --model_version "${MODEL_VERSION}" \
#        --batch_size 1
#
# 2. Network module MUST be 'networks.lora_qwen_image' for Qwen models
#
# 3. Timestep sampling 'shift' with discrete_flow_shift 2.2 is recommended
#    Alternatively, use 'qwen_shift' for dynamic resolution-based shift
#
# 4. For edit variants, control images must be specified in dataset config
#    and will be cached automatically during the caching step
#
# 5. Monitor training with: tensorboard --logdir ./logs
#
