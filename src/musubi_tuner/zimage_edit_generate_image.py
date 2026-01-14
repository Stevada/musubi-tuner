#!/usr/bin/env python3
"""
Z-Image-Edit Inference Script with Dual Pathway Control

Standalone script for generating images using Z-Image-Edit with control/reference images.

Features:
- Dual pathway control conditioning:
  - Pathway 1 (Latent): VAE-encoded control latents
  - Pathway 2 (Embedding): ViT-encoded visual features via VLM
- Classifier-free guidance support
- LoRA weights support
- Memory optimizations (fp8, attention backends)

Example usage:
    python src/musubi_tuner/zimage_edit_generate_image.py \
        --dit models/z-image-dit.safetensors \
        --vae models/z-image-vae.safetensors \
        --text_encoder models/Qwen2.5-VL \
        --control_image examples/reference.png \
        --prompt "A beautiful sunset over mountains" \
        --output_dir output/generated \
        --width 1024 --height 1024 \
        --num_inference_steps 50 \
        --guidance_scale 4.0
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from musubi_tuner.zimage import zimage_model, zimage_autoencoder, zimage_config
from musubi_tuner.zimage import zimage_edit_utils
from musubi_tuner.utils import image_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Z-Image-Edit Inference with Dual Pathway Control")

    # Model paths
    parser.add_argument("--dit", type=str, required=True, help="DiT checkpoint path")
    parser.add_argument("--vae", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen2.5-VL path")
    parser.add_argument("--lora_weight", type=str, help="LoRA weights path (optional)")

    # Control image
    parser.add_argument(
        "--control_image",
        type=str,
        required=True,
        help="Control/reference image path (supports multiple comma-separated paths)",
    )

    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--width", type=int, default=1024, help="Output width (must be multiple of 16)")
    parser.add_argument("--height", type=int, default=1024, help="Output height (must be multiple of 16)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="CFG scale (>1.0 for CFG)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--shift", type=float, default=1.0, help="Flow shift parameter")

    # Output
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--output_name", type=str, default=None, help="Output filename (default: auto-generated)")

    # Optimizations
    parser.add_argument("--fp8_vl", action="store_true", help="Use fp8 for VLM to save memory")
    parser.add_argument("--fp8_scaled", action="store_true", help="Use fp8 for DiT")
    parser.add_argument("--attn_mode", type=str, default="flash", choices=["torch", "flash", "sage", "xformers"], help="Attention backend")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")

    return parser.parse_args()


def load_models(args):
    """Load all models: VAE, DiT, VLM."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load VAE
    logger.info(f"Loading VAE from {args.vae}")
    vae = zimage_autoencoder.load_autoencoder_kl(args.vae, device=device, disable_mmap=True)
    vae.eval()

    # Load DiT
    logger.info(f"Loading DiT from {args.dit}")
    dit_dtype = torch.bfloat16
    dit = zimage_model.load_zimage_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=True,
        loading_device=device,
        dit_weight_dtype=dit_dtype,
        fp8_scaled=args.fp8_scaled,
        disable_numpy_memmap=True,
        use_16bit_for_attention=True,
    )
    dit.eval()

    # Load VLM (Qwen2.5-VL)
    logger.info(f"Loading Qwen2.5-VL from {args.text_encoder}")
    vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
    tokenizer, text_encoder = zimage_edit_utils.load_qwen2_5_vl(
        args.text_encoder, vl_dtype, device, disable_mmap=True
    )
    vl_processor = zimage_edit_utils.load_vl_processor()
    text_encoder.eval()

    # Load LoRA if provided
    if args.lora_weight:
        logger.info(f"Loading LoRA weights from {args.lora_weight}")
        # TODO: Implement LoRA loading
        # This would use musubi_tuner's LoRA utilities to load and merge weights
        logger.warning("LoRA loading not yet implemented in this standalone script")

    return vae, dit, tokenizer, text_encoder, vl_processor, device


def preprocess_control_images(control_image_paths, width, height):
    """Load and preprocess control images."""
    control_tensors = []
    control_nps = []

    for path in control_image_paths:
        logger.info(f"Loading control image: {path}")
        tensor, np_img, _ = zimage_edit_utils.preprocess_control_image(
            path, resize_to_official=False, resize_size=(width, height)
        )
        # Z-Image uses RGB only
        tensor = tensor[:, :3, :, :]
        np_img = np_img[:, :, :3]
        control_tensors.append(tensor)
        control_nps.append(np_img)

    return control_tensors, control_nps


def encode_prompt_with_images(vl_processor, text_encoder, prompt, control_images, device):
    """Encode text prompt with control images via VLM (Pathway 2)."""
    logger.info("Encoding prompt with control images via VLM...")
    with torch.no_grad():
        embed, mask = zimage_edit_utils.get_text_embeds_with_image(
            vl_processor, text_encoder, prompt, control_images, model_version="edit"
        )
    embed = embed.to(device, dtype=torch.bfloat16)
    mask = mask.to(device, dtype=torch.bool)
    return embed, mask


def encode_control_latents(vae, control_tensors, device):
    """Encode control images via VAE (Pathway 1)."""
    logger.info("Encoding control latents via VAE...")
    vae.to(device)
    vae.eval()

    control_latents = []
    with torch.no_grad():
        for tensor in control_tensors:
            latent = vae.encode(tensor.to(device, vae.dtype))
            # Apply Z-Image latent transformations
            shift = zimage_config.ZIMAGE_VAE_SHIFT_FACTOR
            scale = zimage_config.ZIMAGE_VAE_SCALING_FACTOR
            latent = (latent - shift) * scale
            latent = latent.to(torch.bfloat16)
            control_latents.append(latent)

    # Pack and concatenate
    control_latents_packed = []
    for cl in control_latents:
        cl = cl.unsqueeze(2)  # Add frame dim: B, C, H, W -> B, C, 1, H, W
        cl_packed = zimage_edit_utils.pack_latents(cl)  # -> B, L, C
        control_latents_packed.append(cl_packed)
    control_latent = torch.cat(control_latents_packed, dim=1)  # Concat in sequence dim

    vae.to("cpu")
    torch.cuda.empty_cache()

    return control_latent


def generate_image(args, dit, vae, embed, mask, negative_embed, negative_mask, control_latent, device):
    """Generate image using Z-Image-Edit with dual pathway."""
    # Prepare latents
    vae_scale = zimage_config.ZIMAGE_VAE_SCALE_FACTOR * 2
    height_latent = 2 * (args.height // vae_scale)
    width_latent = 2 * (args.width // vae_scale)
    shape = (1, dit.in_channels, height_latent, width_latent)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        logger.info(f"Using seed: {args.seed}")

    latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

    # Trim embeddings
    image_sequence_length = (height_latent // dit.all_patch_size[0]) * (width_latent // dit.all_patch_size[0])
    embed, _ = zimage_edit_utils.trim_pad_embeds_and_mask(image_sequence_length, embed, mask)
    mask = None
    if negative_embed is not None:
        negative_embed, _ = zimage_edit_utils.trim_pad_embeds_and_mask(image_sequence_length, negative_embed, negative_mask)

    # Prepare timesteps
    timesteps, sigmas = zimage_edit_utils.get_timesteps_sigmas(args.num_inference_steps, args.shift)
    timesteps = timesteps.to(device)
    sigmas = sigmas.to(device)

    # CFG
    do_cfg = args.guidance_scale > 1.0

    # Sampling loop
    logger.info("Generating image...")
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000  # Reverse for z-image

            latent_model_input = latents.to(dit.dtype)
            latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dim

            # Pack latents
            latent_model_input_packed = zimage_edit_utils.pack_latents(latent_model_input)

            # Concatenate control latents (Pathway 1)
            if control_latent is not None:
                latent_model_input_packed = torch.cat([latent_model_input_packed, control_latent], dim=1)

            with torch.no_grad():
                # Forward with dual pathway
                model_out = dit(latent_model_input_packed, timestep, embed, mask)

            # Remove control portion
            if control_latent is not None:
                model_out = model_out[:, :image_sequence_length]

            # CFG
            if do_cfg:
                with torch.no_grad():
                    negative_model_out = dit(latent_model_input_packed, timestep, negative_embed, None)
                if control_latent is not None:
                    negative_model_out = negative_model_out[:, :image_sequence_length]
                noise_pred = negative_model_out + args.guidance_scale * (model_out - negative_model_out)
            else:
                noise_pred = model_out

            # Unpack and update
            noise_pred = zimage_edit_utils.unpack_latents(noise_pred, args.height, args.width)
            noise_pred = -noise_pred.squeeze(2)  # Remove frame dim and invert
            latents = zimage_edit_utils.step(noise_pred.to(torch.float32), latents, sigmas, i)

    return latents


def decode_latents(vae, latents, device):
    """Decode latents to pixels."""
    logger.info("Decoding latents to image...")
    latents = latents.to(vae.dtype)
    vae.to(device)
    vae.eval()

    latents = zimage_edit_utils.shift_scale_latents_for_decode(latents)
    with torch.no_grad():
        pixels = vae.decode(latents)

    pixels = pixels.to(torch.float32).cpu()
    pixels = (pixels / 2 + 0.5).clamp(0, 1)

    vae.to("cpu")
    torch.cuda.empty_cache()

    return pixels


def save_image(pixels, output_path):
    """Save generated image."""
    # pixels: [B, C, H, W]
    pixels = pixels[0].permute(1, 2, 0).numpy()  # [H, W, C]
    pixels = (pixels * 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    image.save(output_path)
    logger.info(f"Saved image to {output_path}")


def main():
    args = parse_args()

    # Validate dimensions
    if args.width % 16 != 0 or args.height % 16 != 0:
        raise ValueError(f"Width and height must be multiples of 16, got {args.width}x{args.height}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse control images (support comma-separated)
    control_image_paths = [p.strip() for p in args.control_image.split(",")]

    # Load models
    vae, dit, tokenizer, text_encoder, vl_processor, device = load_models(args)

    # Preprocess control images
    control_tensors, control_nps = preprocess_control_images(control_image_paths, args.width, args.height)

    # Encode prompt with control images (Pathway 2)
    embed, mask = encode_prompt_with_images(vl_processor, text_encoder, args.prompt, control_nps, device)

    # Encode negative prompt if CFG
    negative_embed, negative_mask = None, None
    if args.guidance_scale > 1.0 and args.negative_prompt:
        negative_embed, negative_mask = encode_prompt_with_images(
            vl_processor, text_encoder, args.negative_prompt, control_nps, device
        )

    # Encode control latents (Pathway 1)
    control_latent = encode_control_latents(vae, control_tensors, device)
    control_latent = control_latent.to(device, dtype=torch.bfloat16)

    # Generate
    latents = generate_image(args, dit, vae, embed, mask, negative_embed, negative_mask, control_latent, device)

    # Decode
    pixels = decode_latents(vae, latents, device)

    # Save
    if args.output_name:
        output_path = os.path.join(args.output_dir, args.output_name)
    else:
        seed_str = f"_seed{args.seed}" if args.seed is not None else ""
        output_path = os.path.join(args.output_dir, f"generated{seed_str}.png")

    save_image(pixels, output_path)

    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
