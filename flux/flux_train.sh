#!/bin/bash
# FLUX.1 Kontext i2i LoRA Training Script
# Image-to-image training with control/reference images

# ============================================================================
# Model Paths - Update these to match your setup
# ============================================================================
# Download from: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
DIT_PATH="/workspace/runpod-slim/ComfyUI/models/diffusion_models/flux1-kontext-dev.safetensors"
VAE_PATH="/workspace/runpod-slim/ComfyUI/models/vae/ae.sft"

# Download from: https://huggingface.co/comfyanonymous/flux_text_encoders
TEXT_ENCODER1="/workspace/runpod-slim/ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors"
TEXT_ENCODER2="/workspace/runpod-slim/ComfyUI/models/text_encoders/clip_l.safetensors"

# Dataset configuration
DATASET_CONFIG="/root/musubi-tuner/flux_dataset_config.toml"

# Output settings
OUTPUT_DIR="./output"
OUTPUT_NAME="flux_i2i_lora_demo_bj"

# ============================================================================
# Training Configuration
# ============================================================================
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/flux_kontext_train_network.py \
    --dit "${DIT_PATH}" \
    --vae "${VAE_PATH}" \
    --text_encoder1 "${TEXT_ENCODER1}" \
    --text_encoder2 "${TEXT_ENCODER2}" \
    --dataset_config "${DATASET_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "${OUTPUT_NAME}" \
    --network_module networks.lora_flux \
    --network_dim 32 \
    --network_alpha 32 \
    --learning_rate 1e-4 \
    --optimizer_type adamw8bit \
    --max_train_epochs 16 \
    --save_every_n_epochs 2 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --blocks_to_swap 10 \
    --sdpa \
    --timestep_sampling flux_shift \
    --weighting_scheme none \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --seed 42 \
    --logging_dir ./logs \
    --log_with tensorboard

# ============================================================================
# Memory Optimization Options (for limited VRAM)
# ============================================================================
# Uncomment these lines if you encounter OOM errors:
#
# For 24GB VRAM (RTX 4090):
# --fp8_scaled --fp8 \
# --fp8_t5 \
#
# For 16GB VRAM (RTX 4080):
# --fp8_scaled --fp8 \
# --fp8_t5 \
# --blocks_to_swap 10 \
#
# For 12GB VRAM (RTX 4070):
# --fp8_scaled --fp8 \
# --fp8_t5 \
# --blocks_to_swap 20 \

# ============================================================================
# Notes
# ============================================================================
# 1. Before training, you MUST run caching scripts:
#
#    # Cache latents
#    python src/musubi_tuner/flux_kontext_cache_latents.py \
#        --dataset_config "${DATASET_CONFIG}" \
#        --vae "${VAE_PATH}"
#
#    # Cache text encoder outputs
#    python src/musubi_tuner/flux_kontext_cache_text_encoder_outputs.py \
#        --dataset_config "${DATASET_CONFIG}" \
#        --text_encoder1 "${TEXT_ENCODER1}" \
#        --text_encoder2 "${TEXT_ENCODER2}" \
#        --batch_size 16
#
# 2. Network module MUST be 'networks.lora_flux' for FLUX models
#
# 3. Timestep sampling 'flux_shift' is recommended for FLUX
#
# 4. Control images must have matching filenames with target images
#
# 5. Monitor training with: tensorboard --logdir ./logs
