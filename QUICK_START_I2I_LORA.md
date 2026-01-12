# Quick Start Guide: Character Consistency i2i LoRA Training

This guide shows you how to train a character consistency image-to-image LoRA using Qwen-Image-Edit.

## What You'll Need

- Qwen-Image-Edit model files from [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI):
  - DiT: `qwen_image_edit_bf16.safetensors` (or edit-2509/edit-2511)
  - VAE: `qwen_image_vae.safetensors`
  - Text Encoder: `qwen_2.5_vl_7b.safetensors`
- GPU with at least 12GB VRAM (24GB+ recommended)
- Your dataset: character images + desired outputs + captions

## Step 1: Organize Your Dataset

Create this directory structure:

```
my_character_dataset/
├── target_images/              # Your goal outputs
│   ├── alice_wave.png
│   ├── alice_wave.txt         # "A person waving hello"
│   ├── alice_sit.png
│   ├── alice_sit.txt          # "A person sitting on a chair"
│   ├── bob_jump.png
│   ├── bob_jump.txt           # "A person jumping in the air"
│   └── ...
├── control_images/             # Reference character images
│   ├── alice_wave.png         # MUST match target filename!
│   ├── alice_sit.png
│   ├── bob_jump.png
│   └── ...
└── cache/                      # Auto-created during caching
```

**Important Notes:**
- Control image filenames MUST match target image filenames (ignoring extension)
- Captions should describe the action/pose, NOT the character appearance
- Character information comes from the control image

## Step 2: Create Dataset Config

Copy `sample_dataset_config.toml` and update the paths to point to your dataset:

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/absolute/path/to/my_character_dataset/target_images"
control_directory = "/absolute/path/to/my_character_dataset/control_images"
cache_directory = "/absolute/path/to/my_character_dataset/cache"
num_repeats = 1  # Increase to 2-3 for small datasets (<50 images)
```

**CRITICAL: Use absolute paths, not relative paths!**

## Step 3: Cache Latents

Pre-encode your images through the VAE:

```bash
python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config /absolute/path/to/dataset_config.toml \
    --vae /path/to/qwen_image_vae.safetensors \
    --model_version edit \
    --batch_size 1 \
    --vae_batch_size 1 \
    --num_workers 2
```

**Model Versions:**
- Use `edit` for qwen_image_edit_bf16.safetensors
- Use `edit-2509` for qwen_image_edit_2509_bf16.safetensors
- Use `edit-2511` for qwen_image_edit_2511_bf16.safetensors

This creates `*_qie.safetensors` files in your cache directory.

## Step 4: Cache Text Encoder Outputs

Pre-encode prompts with control images through Qwen2.5-VL:

```bash
python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config /absolute/path/to/dataset_config.toml \
    --text_encoder /path/to/qwen_2.5_vl_7b.safetensors \
    --model_version edit \
    --batch_size 1 \
    --num_workers 2 \
    --fp8_vl
```

**Important:** Add `--fp8_vl` if you have less than 16GB VRAM (saves ~8GB).

This creates `*_qie_te.safetensors` files in your cache directory.

**Verify:** Each training image should have 2 cache files (latents + text encoder).

## Step 5: Train the LoRA

Start training with this command:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/qwen_image_train_network.py \
    --dit /path/to/qwen_image_edit_bf16.safetensors \
    --vae /path/to/qwen_image_vae.safetensors \
    --text_encoder /path/to/qwen_2.5_vl_7b.safetensors \
    --model_version edit \
    --dataset_config /absolute/path/to/dataset_config.toml \
    --output_dir ./output \
    --output_name character_lora \
    --network_module networks.lora_qwen_image \
    --network_dim 32 \
    --network_alpha 16 \
    --learning_rate 1e-4 \
    --optimizer_type adamw8bit \
    --max_train_epochs 20 \
    --save_every_n_epochs 2 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --sdpa \
    --timestep_sampling shift \
    --discrete_flow_shift 2.2 \
    --weighting_scheme none \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --seed 42 \
    --fp8_vl \
    --logging_dir ./logs \
    --log_with tensorboard
```

**Monitor Training:**
```bash
tensorboard --logdir ./logs
```

Loss should decrease to 0.01-0.05. Checkpoints are saved every 2 epochs.

### Hyperparameter Recommendations by Dataset Size

| Dataset Size | Epochs | Learning Rate | network_dim | num_repeats |
|--------------|--------|---------------|-------------|-------------|
| <50 images   | 20-30  | 1e-4          | 32          | 2-3         |
| 50-200 images| 15-20  | 1e-4          | 32-64       | 1-2         |
| >200 images  | 10-15  | 8e-5          | 64-128      | 1           |

### Memory Optimization (if needed)

If you get Out of Memory errors, add these flags:

```bash
# Reduce from 42GB to 30GB VRAM
--fp8_base --fp8_scaled

# Reduce to 24GB VRAM (requires 64GB+ RAM)
--fp8_base --fp8_scaled --blocks_to_swap 16

# Reduce to 12GB VRAM (requires 64GB+ RAM)
--fp8_base --fp8_scaled --blocks_to_swap 45
```

## Step 6: Test Your LoRA

Generate test images with your trained LoRA:

```bash
python src/musubi_tuner/qwen_image_generate_image.py \
    --dit /path/to/qwen_image_edit_bf16.safetensors \
    --vae /path/to/qwen_image_vae.safetensors \
    --text_encoder /path/to/qwen_2.5_vl_7b.safetensors \
    --model_version edit \
    --control_image_path /path/to/reference_character.png \
    --prompt "A person waving hello" \
    --negative_prompt " " \
    --image_size 1024 1024 \
    --infer_steps 25 \
    --guidance_scale 4.0 \
    --attn_mode sdpa \
    --lora_weight ./output/character_lora_000010.safetensors \
    --lora_multiplier 1.0 \
    --save_path ./output_images \
    --output_type images \
    --seed 12345 \
    --resize_control_to_official_size
```

**Key Parameters:**
- `--control_image_path`: Your reference character image
- `--prompt`: Describe the desired action/pose
- `--lora_multiplier`: LoRA strength (0.5-1.5, try 0.7-1.2)
- `--resize_control_to_official_size`: Highly recommended for best results

### Testing Multiple Checkpoints

Compare different training epochs to find the best:

```bash
for epoch in 2 4 6 8 10 12 14 16 18 20; do
    python src/musubi_tuner/qwen_image_generate_image.py \
        --dit /path/to/qwen_image_edit_bf16.safetensors \
        --vae /path/to/qwen_image_vae.safetensors \
        --text_encoder /path/to/qwen_2.5_vl_7b.safetensors \
        --model_version edit \
        --control_image_path /path/to/reference_character.png \
        --prompt "A person waving hello" \
        --image_size 1024 1024 \
        --infer_steps 25 \
        --guidance_scale 4.0 \
        --attn_mode sdpa \
        --lora_weight ./output/character_lora_$(printf "%06d" $epoch).safetensors \
        --lora_multiplier 1.0 \
        --save_path ./test/epoch_$epoch \
        --seed 42 \
        --resize_control_to_official_size
done
```

## Troubleshooting

**"Cache file not found"**
- Solution: Re-run both caching steps, verify `.safetensors` files exist in cache directory

**Training loss doesn't decrease**
- Solution: Increase learning rate to 2e-4, increase `num_repeats` in TOML config

**Character not preserved in output**
- Solution: Use high-quality control images, verify `model_version` matches everywhere, increase `network_dim` to 64

**Out of Memory during training**
- Solution: Add `--fp8_base --fp8_scaled --blocks_to_swap 16`, or reduce resolution

**Generated images have artifacts**
- Solution: Verify all model files from same source, try earlier checkpoint, reduce `lora_multiplier`

**Inference OOM (Out of Memory)**
- Solution: Add `--text_encoder_cpu` to run text encoder on CPU (saves ~8GB VRAM)

## Critical Reminders

1. **Model Version Consistency**: Use the same `--model_version` (edit/edit-2509/edit-2511) in all steps
2. **Absolute Paths**: All TOML config paths must be absolute
3. **Filename Matching**: Control and target images must have matching filenames
4. **FP8 Consistency**: If you cache with `--fp8_vl`, use it in training too
5. **Same Model Files**: Use the same DiT/VAE/Text Encoder throughout

## Expected Results

With proper training:
- Character appearance (face, clothing, style) preserved across different poses
- Generated images follow the prompt while maintaining character identity
- Clean backgrounds and consistent proportions
- Best results typically at epochs 10-20 depending on dataset size

## How It Works

Qwen-Image-Edit achieves character consistency through:
1. Control images encoded to latent space via VAE
2. Control latents concatenated with noisy target latents during denoising
3. Vision-Language Model (Qwen2.5-VL) processes control image WITH text prompt
4. Creates image-conditioned text embeddings that understand both character and desired action

This approach provides superior character consistency compared to ControlNet or IP-Adapter.

## Next Steps

- Experiment with different `lora_multiplier` values (0.5-1.5)
- Try different prompts to test versatility
- Test on control images NOT in your training set
- Adjust hyperparameters based on your specific dataset

For complete details, see the full implementation plan at `/home/stevexu/.claude/plans/composed-yawning-church.md`
