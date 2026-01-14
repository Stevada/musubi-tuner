#!/bin/bash
# Qwen-Image i2i LoRA Inference/Testing Script
# Generate images using trained Qwen LoRA
# Supports all 5 variants with edit-specific advanced features

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

# ============================================================================
# LoRA Settings
# ============================================================================
# Path to your trained LoRA weight
# Update this to point to your trained LoRA checkpoint
LORA_WEIGHT="./output/qwen_lora_${MODEL_VERSION}.safetensors"

# LoRA strength (0.0 to 1.5+)
# 0.5 = half strength, 1.0 = full strength, 1.5 = overcooked
LORA_MULTIPLIER=1.0

# ============================================================================
# Inference Settings
# ============================================================================
# Control/reference image
# REQUIRED for edit variants (edit, edit-2509, edit-2511)
# Optional for layered variant (used for consistency)
# Not used for original variant
CONTROL_IMAGE="./dataset/control_images_test/sample.png"

# Prompt for generation
PROMPT="A beautiful landscape with mountains and a lake"

# Negative prompt (use space, not empty string)
NEGATIVE_PROMPT=" "

# Output settings
OUTPUT_DIR="./test_outputs"
OUTPUT_NAME="qwen_test_001"

# Image size (height width)
IMAGE_HEIGHT=1024
IMAGE_WIDTH=1024

# Inference steps (more = higher quality but slower)
# Recommended: 20-30 for Qwen
INFER_STEPS=25

# Random seed (for reproducibility)
# Use -1 for random seed each time
SEED=42

# Guidance scale (classifier-free guidance)
# Typical: 3.0-5.0 for Qwen
GUIDANCE_SCALE=4.0

# ============================================================================
# Advanced Options - Edit Variants Only
# ============================================================================
# Control image resize options (mutually exclusive)
# Recommended: Use resize_to_official for edit variants
RESIZE_CONTROL_TO_OFFICIAL=true  # Resize to 1M pixels keeping aspect ratio
RESIZE_CONTROL_TO_IMAGE_SIZE=false  # Resize to match output size

# Reference Consistency Mask (RCM) - Edit variants only
# Prevents unintended background/face changes
# Leave RCM_THRESHOLD empty to disable
RCM_THRESHOLD=""  # Example: "0.2" for relative, "0.05" for absolute
RCM_RELATIVE_THRESHOLD=true  # true = relative (0.1-0.5), false = absolute (0.01-0.1)
RCM_KERNEL_SIZE=3  # Gaussian blur kernel size for smoother masks
RCM_DILATE_SIZE=0  # Expand inpainting region by N pixels
RCM_DEBUG_SAVE=false  # Save per-step masks for debugging

# Inpainting mask (mutually exclusive with RCM)
# Black/white mask: white = edit region, black = preserve
MASK_PATH=""  # Example: "./masks/inpaint_mask.png"

# ============================================================================
# Advanced Options - Layered Variant Only
# ============================================================================
# Number of layers to output (original + N layers will be generated)
OUTPUT_LAYERS=4

# ============================================================================
# Memory Optimization Options
# ============================================================================
FP8_SCALED=false
TEXT_ENCODER_CPU=false  # Recommended for <16GB VRAM
BLOCKS_TO_SWAP=0  # 0 = disabled, 10-15 for lower VRAM

# ============================================================================
# Create output directory
# ============================================================================
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Build Command Line Arguments
# ============================================================================
# Build control image arguments
CONTROL_ARGS=""
if [ -n "$CONTROL_IMAGE" ]; then
    CONTROL_ARGS="--control_image_path ${CONTROL_IMAGE}"
fi

# Build resize arguments
RESIZE_ARGS=""
if [ "$RESIZE_CONTROL_TO_OFFICIAL" = true ]; then
    RESIZE_ARGS="--resize_control_to_official_size"
elif [ "$RESIZE_CONTROL_TO_IMAGE_SIZE" = true ]; then
    RESIZE_ARGS="--resize_control_to_image_size"
fi

# Build RCM arguments
RCM_ARGS=""
if [ -n "$RCM_THRESHOLD" ]; then
    RCM_ARGS="--rcm_threshold ${RCM_THRESHOLD}"
    if [ "$RCM_RELATIVE_THRESHOLD" = true ]; then
        RCM_ARGS="$RCM_ARGS --rcm_relative_threshold"
    fi
    RCM_ARGS="$RCM_ARGS --rcm_kernel_size ${RCM_KERNEL_SIZE}"
    if [ "$RCM_DILATE_SIZE" -gt 0 ]; then
        RCM_ARGS="$RCM_ARGS --rcm_dilate_size ${RCM_DILATE_SIZE}"
    fi
    if [ "$RCM_DEBUG_SAVE" = true ]; then
        RCM_ARGS="$RCM_ARGS --rcm_debug_save"
    fi
fi

# Build mask arguments
MASK_ARGS=""
if [ -n "$MASK_PATH" ]; then
    MASK_ARGS="--mask_path ${MASK_PATH}"
fi

# Build layered arguments
LAYERED_ARGS=""
if [ "$MODEL_VERSION" = "layered" ]; then
    LAYERED_ARGS="--output_layers ${OUTPUT_LAYERS}"
fi

# Build memory optimization arguments
MEMORY_ARGS=""
if [ "$FP8_SCALED" = true ]; then
    MEMORY_ARGS="$MEMORY_ARGS --fp8_scaled"
fi
if [ "$TEXT_ENCODER_CPU" = true ]; then
    MEMORY_ARGS="$MEMORY_ARGS --text_encoder_cpu"
fi
if [ "$BLOCKS_TO_SWAP" -gt 0 ]; then
    MEMORY_ARGS="$MEMORY_ARGS --blocks_to_swap ${BLOCKS_TO_SWAP}"
fi

# ============================================================================
# Run Inference
# ============================================================================
echo "============================================"
echo "Qwen-Image i2i LoRA Inference"
echo "============================================"
echo "Model Version: ${MODEL_VERSION}"
echo "LoRA: ${LORA_WEIGHT}"
echo "Control Image: ${CONTROL_IMAGE}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}.png"
echo "Prompt: ${PROMPT}"
echo "============================================"
echo ""

python src/musubi_tuner/qwen_image_generate_image.py \
    --dit "${DIT_PATH}" \
    --vae "${SELECTED_VAE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --model_version "${MODEL_VERSION}" \
    --lora_weight "${LORA_WEIGHT}" \
    --lora_multiplier ${LORA_MULTIPLIER} \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --image_size ${IMAGE_HEIGHT} ${IMAGE_WIDTH} \
    --infer_steps ${INFER_STEPS} \
    --seed ${SEED} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --save_path "${OUTPUT_DIR}/${OUTPUT_NAME}.png" \
    --attn_mode xformers \
    $CONTROL_ARGS \
    $RESIZE_ARGS \
    $RCM_ARGS \
    $MASK_ARGS \
    $LAYERED_ARGS \
    $MEMORY_ARGS

echo ""
echo "============================================"
echo "✓ Inference completed!"
echo "============================================"
echo "Output saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}.png"
echo ""

# ============================================================================
# Memory Optimization Guide
# ============================================================================
# For 24GB VRAM (RTX 4090):
# - FP8_SCALED=true
# - TEXT_ENCODER_CPU=false
# - BLOCKS_TO_SWAP=0
#
# For 16GB VRAM (RTX 4080):
# - FP8_SCALED=true
# - TEXT_ENCODER_CPU=false
# - BLOCKS_TO_SWAP=10
#
# For 12GB VRAM (RTX 4070):
# - FP8_SCALED=true
# - TEXT_ENCODER_CPU=true
# - BLOCKS_TO_SWAP=15
#
# ============================================================================
# Advanced Edit Features Guide (edit, edit-2509, edit-2511 only)
# ============================================================================
# Reference Consistency Mask (RCM):
# - Prevents unintended changes to background/unchanged areas
# - Dynamically creates mask during denoising
# - Lower threshold = larger editing region
# - Typical values:
#   - Relative: 0.1 to 0.5 (RCM_RELATIVE_THRESHOLD=true)
#   - Absolute: 0.01 to 0.1 (RCM_RELATIVE_THRESHOLD=false)
# - Enable with: RCM_THRESHOLD="0.2"
# - Use RCM_DEBUG_SAVE=true to visualize masks
# - REQUIRES: Control image same size as output image
# - MUTUALLY EXCLUSIVE with inpainting mask
#
# Inpainting Mask:
# - Specify regions to edit with black/white mask
# - White = edit region, Black = preserve
# - Set MASK_PATH="/path/to/mask.png"
# - REQUIRES: Control image same size as output image
# - MUTUALLY EXCLUSIVE with RCM
#
# Control Image Resize Options:
# - resize_to_official_size: Recommended for edit variants (resizes to 1M pixels)
# - resize_to_image_size: Alternative for layered variant
# - Mutually exclusive options
# - Edit-2511 requires official size resizing for best results
#
# ============================================================================
# Layered Variant Features (layered only)
# ============================================================================
# - Automatically segments image into layers
# - Set OUTPUT_LAYERS=4 (number of layers to generate)
# - Control image can be provided for consistency
# - Automatic captioning if prompt is empty
# - Will generate OUTPUT_LAYERS + 1 images (original + layers)
#
# ============================================================================
# Additional Advanced Options
# ============================================================================
# You can add these flags to the python command above:
#
# --flow_shift 2.0           # Adjust noise schedule (default: auto)
# --output_type latent       # Save latent instead of decoded image
# --no_metadata              # Don't save metadata in output image
# --interactive              # Enable interactive mode for multiple prompts
# --from_file prompts.txt    # Read prompts from file (one per line)
# --append_original_name     # Append control image name to output filename
#
