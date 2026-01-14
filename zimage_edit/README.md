# Z-Image-Edit: Dual Pathway Control Image Support for Z-Image

Z-Image-Edit extends Z-Image with dual pathway control/reference image conditioning, following the proven Qwen-Image-Edit architecture pattern.

## Features

- **Dual Pathway Architecture**: Control images flow through two pathways for rich conditioning
  - **Pathway 1 (Latent)**: VAE-encoded control latents concatenated to noisy target
  - **Pathway 2 (Embedding)**: ViT-encoded visual features embedded in text stream via VLM
- **Qwen2.5-VL Integration**: Vision-language model replaces Qwen3 for multimodal understanding
- **Memory Efficient**: Support for fp8 quantization, gradient checkpointing, and advanced attention backends
- **LoRA Training**: Efficient fine-tuning with Low-Rank Adaptation
- **Classifier-Free Guidance**: High-quality generation with CFG support

## Architecture

### Data Flow

```
Control Image(s) ───┬──> VAE Encoder ──> Control Latents ──┐
                    │                                       ├─> Concatenate ─> DiT Model
Target Image ─> VAE ──> Noisy Latents ─────────────────────┘
                    │
                    └──> Qwen2.5-VL (ViT + LLM) ──> Text Embeddings
Text Prompt ────────────┘                            (with visual features)
```

### Key Components

1. **Utility Module** (`zimage_edit_utils.py`): Wraps Qwen VLM and Z-Image functions
2. **Training Script** (`zimage_edit_train_network.py`): LoRA training with dual pathway
3. **Inference Script** (`zimage_edit_generate_image.py`): Standalone image generation
4. **Bash Scripts**: `train.sh` and `inference.sh` for easy usage

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- transformers >= 4.40 (for Qwen2.5-VL)
- accelerate
- safetensors
- einops
- PIL

### Model Checkpoints

Download the required models:

- **Z-Image DiT**: Base diffusion transformer checkpoint
- **Z-Image VAE**: Variational autoencoder for latent encoding
- **Qwen2.5-VL**: Vision-language model (NEW - replaces Qwen3)

```bash
# Example directory structure
models/
├── z-image-dit.safetensors
├── z-image-vae.safetensors
└── Qwen2.5-VL/
    ├── model.safetensors
    ├── config.json
    └── ...
```

## Quick Start

### 1. Prepare Dataset

Create a dataset configuration file (TOML format):

```toml
[general]
architecture = "zimage-edit"
resolution = [1024, 1024]

[[datasets]]
image_dir = "path/to/target/images"
control_image_dir = "path/to/control/images"
latents_cache_dir = "path/to/latents/cache"
latents_cache_dir_control = "path/to/control/latents/cache"
metadata_file = "path/to/metadata.jsonl"
```

### 2. Cache Latents

Cache both target and control image latents:

```bash
# Cache target image latents
python src/musubi_tuner/cache_latents.py \
    --dataset_config config/datasets/zimage_edit_example.toml \
    --vae models/z-image-vae.safetensors \
    --architecture zimage-edit

# Control latents are cached automatically during training preparation
```

### 3. Train LoRA

Edit `zimage_edit/train.sh` to set your paths and parameters, then run:

```bash
bash zimage_edit/train.sh
```

Key training parameters:
- `NETWORK_DIM=32`: LoRA rank (higher = more capacity, more memory)
- `LEARNING_RATE=1e-4`: Base learning rate
- `MAX_STEPS=10000`: Total training steps
- `FP8_VL=true`: Use fp8 for VLM to save memory (recommended)

### 4. Generate Images

Edit `zimage_edit/inference.sh` to set your control image and prompt, then run:

```bash
bash zimage_edit/inference.sh
```

Or use the Python script directly:

```bash
python src/musubi_tuner/zimage_edit_generate_image.py \
    --dit models/z-image-dit.safetensors \
    --vae models/z-image-vae.safetensors \
    --text_encoder models/Qwen2.5-VL \
    --control_image examples/reference.png \
    --prompt "A beautiful sunset over mountains" \
    --output_dir output/generated \
    --width 1024 --height 1024 \
    --num_inference_steps 50 \
    --guidance_scale 4.0 \
    --seed 42
```

## Advanced Usage

### Memory Optimization

For GPUs with limited VRAM:

1. **Enable fp8 for VLM**: `--fp8_vl` (saves ~50% memory)
2. **Use gradient checkpointing**: `--gradient_checkpointing`
3. **Lower batch size**: `--train_batch_size 1` with higher `--gradient_accumulation_steps`
4. **Try different attention backends**: `--attn_mode flash` (fastest) or `--attn_mode sage` (most memory efficient)

### Multiple Control Images

Z-Image-Edit supports multiple control images (comma-separated):

```bash
python src/musubi_tuner/zimage_edit_generate_image.py \
    --control_image "image1.png,image2.png,image3.png" \
    ...
```

### Fine-tuning Tips

1. **Start with base model validation**: Test inference with base model before training
2. **Monitor sample generations**: Use `--sample_every_n_steps 500` to track progress
3. **Adjust LoRA rank**: Higher `network_dim` for complex tasks, lower for simpler edits
4. **Learning rate**: Start with `1e-4`, reduce if training is unstable
5. **CFG scale**: 3.5-5.0 works well for most cases, adjust based on output quality

## Comparison with Other Architectures

| Feature | Z-Image | Z-Image-Edit | Qwen-Image-Edit | FLUX Kontext |
|---------|---------|--------------|-----------------|--------------|
| Control images | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| Dual pathway | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| Text encoder | Qwen3 (LLM) | Qwen2.5-VL (VLM) | Qwen2.5-VL (VLM) | T5 + CLIP |
| Architecture | Unified stream | Unified stream | Double-stream | Double-stream |
| Model changes | - | None (compatible!) | Yes (RoPE) | No |

**Key Advantage**: Z-Image-Edit requires NO changes to the base Z-Image model architecture due to its unified stream design!

## Technical Details

### Why Dual Pathway?

The dual pathway approach allows the model to leverage control images at multiple abstraction levels:

- **Low-level (Pathway 1)**: Spatial structure, colors, textures via VAE latents
- **High-level (Pathway 2)**: Semantic concepts, objects, composition via ViT features

This results in more faithful control compared to single pathway approaches.

### Model Compatibility

Z-Image-Edit uses the same base DiT architecture as Z-Image:
- Can potentially load same DiT weights (untested)
- Requires Qwen2.5-VL instead of Qwen3
- Training data must include control images

### Flow Matching

Z-Image-Edit uses inverted flow matching:
- Target: `latents - noise` (opposite of standard flow matching)
- Supports discrete flow shift for noise schedule adjustment

## Troubleshooting

### Out of Memory (OOM)

1. Enable fp8 for VLM: `--fp8_vl`
2. Reduce batch size: `--train_batch_size 1`
3. Enable gradient checkpointing: `--gradient_checkpointing`
4. Try SageAttention: `--attn_mode sage`
5. Lower image resolution temporarily

### Poor Generation Quality

1. Check control image quality and resolution
2. Increase CFG scale: `--guidance_scale 5.0`
3. Increase sampling steps: `--num_inference_steps 75`
4. Try different seeds
5. Adjust negative prompt

### Training Not Converging

1. Reduce learning rate: `--learning_rate 5e-5`
2. Increase warmup steps: `--lr_warmup_steps 200`
3. Check dataset quality (control images must align with targets)
4. Monitor sample generations to diagnose issues

## File Structure

```
musubi-tuner/
├── src/musubi_tuner/
│   ├── zimage/
│   │   ├── zimage_edit_utils.py      # NEW: VLM utilities
│   │   ├── zimage_model.py           # Existing (no changes)
│   │   ├── zimage_utils.py           # Existing (no changes)
│   │   └── ...
│   ├── zimage_edit_train_network.py  # NEW: Training script
│   ├── zimage_edit_generate_image.py # NEW: Inference script
│   └── dataset/
│       └── image_video_dataset.py    # Updated with constants
│
├── zimage_edit/                      # NEW: Scripts and docs
│   ├── README.md                     # This file
│   ├── train.sh                      # Training script
│   └── inference.sh                  # Inference script
│
└── config/
    └── datasets/
        └── zimage_edit_example.toml  # Example config
```

## Citation

If you use Z-Image-Edit in your research, please cite:

```bibtex
@software{zimage_edit_2025,
  title={Z-Image-Edit: Dual Pathway Control Image Support for Z-Image},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/musubi-tuner}
}
```

## License

This implementation follows the same license as the Musubi Tuner project.

## Acknowledgments

- Z-Image team for the base architecture
- Qwen team for Qwen2.5-VL vision-language model
- Qwen-Image-Edit for the dual pathway inspiration
- Musubi Tuner project for the training framework

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review the CLAUDE.md file for project overview

---

Happy generating! 🎨
