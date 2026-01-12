# Claude Code Guidance for Musubi Tuner

This document provides guidance for Claude Code when working with the Musubi Tuner codebase.

## Project Overview

Musubi Tuner is a Python-based training framework for LoRA (Low-Rank Adaptation) models supporting multiple video and image generation architectures:

- **Video Models**: HunyuanVideo, HunyuanVideo 1.5, Wan2.1/2.2, FramePack
- **Image Models**: FLUX.1 Kontext, Z-Image, Qwen-Image/Qwen-Image-Edit/Qwen-Image-Layered

The project emphasizes memory-efficient training and inference for generative models.

## Environment Setup

### Requirements
- **Python**: 3.10 or later (verified with 3.10)
- **PyTorch**: 2.5.1 or later with CUDA support
- **Installation methods**: pip or uv (experimental)

### Installation Commands
```bash
# Install PyTorch with CUDA first
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install project with pip
pip install -e .

# Or use uv (experimental)
uv run --extra cu124  # or cu128, cu130
```

### Optional Dependencies
- `ascii-magic`: ASCII art visualization
- `matplotlib`: Plotting utilities
- `tensorboard`: Training monitoring
- `prompt-toolkit`: Interactive CLI features

## Project Structure

### Core Directories
- `src/musubi_tuner/`: Main package with training/inference scripts
- `src/musubi_tuner/dataset/`: Dataset configuration and loading
- `src/musubi_tuner/modules/`: Model architectures and components
- `src/musubi_tuner/networks/`: LoRA network implementations
- `src/musubi_tuner/utils/`: Common utilities (model handling, device management)
- `docs/`: Comprehensive documentation

### Architecture-Specific Modules
- `hunyuan_model/`: HunyuanVideo implementation
- `hunyuan_video_1_5/`: HunyuanVideo 1.5 configurations
- `wan/`: Wan2.1/2.2 modules
- `frame_pack/`: FramePack implementation
- `flux/`: FLUX model utilities
- `zimage/`: Z-Image utilities
- `qwen_image/`: Qwen-Image utilities

## Common Workflows

### 1. Dataset Preparation (Required Before Training)

```bash
# Cache latents
python src/musubi_tuner/cache_latents.py \
  --dataset_config path/to/dataset.toml \
  --vae path/to/vae \
  --vae_chunk_size 32 \
  --vae_tiling

# Cache text encoder outputs
python src/musubi_tuner/cache_text_encoder_outputs.py \
  --dataset_config path/to/dataset.toml \
  --text_encoder1 path/to/te1 \
  --text_encoder2 path/to/te2 \
  --batch_size 16
```

### 2. Training

Each architecture has its own training script following the pattern `{architecture}_train_network.py`:

```bash
# General pattern
accelerate launch \
  --num_cpu_threads_per_process 1 \
  --mixed_precision bf16 \
  src/musubi_tuner/{architecture}_train_network.py \
  --dit path/to/dit \
  --dataset_config path/to/dataset.toml \
  --network_module networks.lora \
  --network_dim 32

# Examples:
# HunyuanVideo: hv_train_network.py
# HunyuanVideo 1.5: hv_1_5_train_network.py
# Wan2.1: wan_train_network.py
# FramePack: fpack_train_network.py
# FLUX.1 Kontext: flux_kontext_train_network.py
# Z-Image: zimage_train_network.py
# Qwen-Image: qwen_image_train_network.py
```

**Note**: Qwen-Image series also supports full fine-tuning via `qwen_image_train.py`.

### 3. Inference

Each architecture has its own inference script:

```bash
# Video generation (HunyuanVideo example)
python src/musubi_tuner/hv_generate_video.py \
  --fp8 \
  --video_size 544 960 \
  --video_length 5 \
  --prompt "your prompt here" \
  --dit path/to/dit \
  --vae path/to/vae

# Image generation (FLUX.1 Kontext example)
python src/musubi_tuner/flux_kontext_generate_image.py \
  --control_image_path path/to/control.png \
  --prompt "your prompt here" \
  [similar args]
```

### 4. LoRA Utilities

```bash
# Merge LoRA weights into base model
python src/musubi_tuner/merge_lora.py \
  --dit path/to/dit \
  --lora_weight path/to/lora.safetensors \
  --save_merged_model path/to/output

# Convert LoRA formats
python src/musubi_tuner/convert_lora.py \
  --input path/to/lora.safetensors \
  --output path/to/converted.safetensors \
  --target other

# Post-hoc EMA for LoRA
python src/musubi_tuner/lora_post_hoc_ema.py [args]
```

## Key Technical Concepts

### Dataset Configuration
- Uses **TOML format** for dataset specification
- Supports images, videos, control images, and metadata JSONL files
- Features bucketing for variable resolutions/lengths
- Architecture-specific settings in dataset config

### Memory Optimization
The project prioritizes memory efficiency with multiple strategies:
- **Precision**: `--fp8_base`, `--fp8_llm` for 8-bit precision
- **Block swapping**: `--blocks_to_swap` to offload model blocks
- **Attention mechanisms**: SDPA, FlashAttention, SageAttention, xformers
- **VAE optimization**: Tiling or chunking for large resolutions
- **Gradient checkpointing**: Reduce memory during backpropagation
- **Mixed precision**: bf16 training via accelerate

### LoRA Networks
- Modular implementations for different architectures
- Configurable via `--network_module`, `--network_dim`, `--network_alpha`
- Support for different target modules and rank configurations
- Network-specific parameters for fine-grained control

### Training Features
- **Timestep sampling**: Control which diffusion steps to train on
- **Discrete flow shift**: Adjust noise schedules
- **Distributed training**: Via accelerate for multi-GPU setups
- **Interactive/batch modes**: Flexible inference workflows

## Development Guidelines

### When Working on This Project

1. **Architecture-specific changes**: Each architecture (HunyuanVideo, Wan, FramePack, etc.) has separate modules. Changes to one architecture typically don't affect others.

2. **Dataset modifications**: Dataset handling is centralized in `src/musubi_tuner/dataset/`. Changes here may affect multiple architectures.

3. **Memory optimization**: Be extremely cautious with memory usage. This project is designed for large models where memory is critical.

4. **Testing**: No formal test suite exists. Manual testing via training/inference scripts is required.

5. **Configuration**: Prefer command-line arguments over hardcoded values. Follow existing patterns for new parameters.

6. **Documentation**: Update relevant files in `docs/` for significant changes.

### Common Debugging Steps

1. Check dataset TOML configuration
2. Verify latents and text encodings are cached
3. Check CUDA memory usage (`nvidia-smi`)
4. Try fp8 precision or block swapping if OOM
5. Review accelerate configuration
6. Check model paths and file accessibility

### File Naming Conventions

- Training scripts: `{architecture}_train_network.py`
- Inference scripts: `{architecture}_generate_{video|image}.py`
- Models: `{architecture}_model/` or `{architecture}/`
- Networks: `networks/lora_{architecture}.py` or similar

## Important Notes

- **Active development**: Project has experimental features
- **No CI/CD**: Manual validation required
- **Accelerate**: Used for distributed training setup
- **Documentation**: Comprehensive guides in `docs/` directory
- **Python version**: Verified with 3.10, may work with newer versions
- **GPU required**: Training and inference require CUDA-capable GPUs

## When to Ask Questions

If you're uncertain about:
- Which architecture the user is working with
- Dataset format expectations
- Memory constraints of the user's hardware
- Intended use case (training vs inference)
- Whether full fine-tuning or LoRA is desired

Ask the user for clarification before making assumptions.

## References

- Project documentation: `docs/` directory
- Dataset examples: Look for `.toml` files in the repository
- Architecture guides: `docs/` contains architecture-specific documentation
- Installation guide: `README.md` (if present) or `pyproject.toml`
