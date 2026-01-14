"""
Z-Image-Edit Utilities Module

This module provides utility functions for Z-Image-Edit with dual pathway control image support.
It wraps functions from qwen_image_utils (for VLM and control image handling) and zimage_utils
(for Z-Image specific operations).

Architecture:
- Pathway 1 (Latent): VAE-encoded control latents concatenated to noisy target
- Pathway 2 (Embedding): ViT-encoded control features embedded in text stream via VLM
"""

import logging
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor
from transformers.image_utils import ImageInput

# Import from qwen_image for VLM and control image handling
from musubi_tuner.qwen_image import qwen_image_utils

# Import from zimage for Z-Image specific operations
from musubi_tuner.zimage import zimage_utils

logger = logging.getLogger(__name__)

# ============================================================================
# VLM Functions (wrapped from qwen_image_utils)
# ============================================================================


def load_qwen2_5_vl(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> Tuple[Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration]:
    """
    Load Qwen2.5-VL vision-language model for Z-Image-Edit.

    This is the dual pathway text encoder that replaces Qwen3 in Z-Image.
    It provides:
    - ViT encoding for control/reference images
    - Text encoding via language model
    - Merged visual+text embeddings

    Args:
        ckpt_path: Path to Qwen2.5-VL checkpoint
        dtype: Model dtype (torch.bfloat16 or torch.float8_e4m3fn for fp8)
        device: Device to load model on
        disable_mmap: Disable numpy memmap for loading
        state_dict: Optional pre-loaded state dict

    Returns:
        Tuple of (tokenizer, VLM model)
    """
    logger.info(f"Loading Qwen2.5-VL for Z-Image-Edit from {ckpt_path}")
    return qwen_image_utils.load_qwen2_5_vl(ckpt_path, dtype, device, disable_mmap, state_dict)


def load_vl_processor() -> Qwen2VLProcessor:
    """
    Load Qwen2VL processor for image preprocessing.

    The processor handles:
    - Image resizing and normalization
    - Vision token preparation
    - Integration with VLM input format

    Returns:
        Qwen2VL processor instance
    """
    return qwen_image_utils.load_vl_processor()


def get_text_embeds_with_image(
    vl_processor: Qwen2VLProcessor,
    vlm: Qwen2_5_VLForConditionalGeneration,
    prompt: Union[str, List[str]],
    images: Union[List[ImageInput], ImageInput] = None,
    model_version: str = "edit",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text prompt with control images via VLM (Pathway 2).

    This function processes control images through the ViT in Qwen2.5-VL
    and merges visual features with text tokens. The resulting embeddings
    contain both textual and visual information.

    Args:
        vl_processor: Qwen2VL processor
        vlm: Qwen2.5-VL model
        prompt: Text prompt(s)
        images: Control image(s) as PIL Images or numpy arrays
        model_version: Model version ("edit" for Z-Image-Edit)

    Returns:
        Tuple of (embeddings, attention_mask)
        - embeddings: [batch, seq_len, hidden_dim] with visual+text features
        - attention_mask: [batch, seq_len] boolean mask
    """
    logger.debug(f"Encoding text with control images via VLM (model_version={model_version})")
    return qwen_image_utils.get_qwen_prompt_embeds_with_image(
        vl_processor, vlm, prompt, images, model_version=model_version
    )


def preprocess_control_image(
    image_path: str,
    resize_to_official: bool = True,
    resize_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Load and preprocess control/reference image.

    Handles:
    - Loading image from path
    - Resizing to target resolution
    - Conversion to tensor format
    - Normalization

    Args:
        image_path: Path to control image
        resize_to_official: Use official resizing (True) or custom size (False)
        resize_size: Optional (width, height) for custom resize

    Returns:
        Tuple of (image_tensor, image_np, original_size)
        - image_tensor: [1, C, H, W] tensor for VAE encoding
        - image_np: [H, W, C] numpy array for VLM encoding
        - original_size: (width, height) of original image
    """
    return qwen_image_utils.preprocess_control_image(image_path, resize_to_official, resize_size)


# ============================================================================
# Latent Manipulation Functions (wrapped from qwen_image_utils)
# ============================================================================


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Pack latents from spatial format to sequence format.

    Converts: [B, C, F, H, W] -> [B, F*H*W, C]

    This is required for concatenating control latents to noisy target latents
    in the sequence dimension (Pathway 1).

    Args:
        latents: [B, C, F, H, W] spatial latents

    Returns:
        [B, seq_len, C] packed latents where seq_len = F*H*W
    """
    return qwen_image_utils.pack_latents(latents)


def unpack_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int = qwen_image_utils.VAE_SCALE_FACTOR,
) -> torch.Tensor:
    """
    Unpack latents from sequence format to spatial format.

    Converts: [B, seq_len, C] -> [B, C, F, H, W]

    This is the inverse of pack_latents, used after the DiT forward pass
    to convert predictions back to spatial format.

    Args:
        latents: [B, seq_len, C] packed latents
        height: Target height in pixels
        width: Target width in pixels
        vae_scale_factor: VAE downscaling factor (default: 8)

    Returns:
        [B, C, F, H, W] unpacked latents
    """
    return qwen_image_utils.unpack_latents(latents, height, width, vae_scale_factor)


# ============================================================================
# Z-Image Specific Functions (wrapped from zimage_utils)
# ============================================================================


def get_timesteps_sigmas(
    num_inference_steps: int,
    shift: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get timesteps and sigmas for Z-Image sampling.

    Z-Image uses a specific flow matching schedule with adjustable shift.

    Args:
        num_inference_steps: Number of sampling steps
        shift: Flow shift parameter for adjusting noise schedule

    Returns:
        Tuple of (timesteps, sigmas)
        - timesteps: [num_steps] timestep values
        - sigmas: [num_steps] noise level values
    """
    return zimage_utils.get_timesteps_sigmas(num_inference_steps, shift)


def step(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """
    Perform one step of Z-Image denoising.

    Uses Euler method for flow matching ODE integration.

    Args:
        model_output: Model prediction (velocity field)
        sample: Current sample (noisy latent)
        sigmas: Noise schedule
        step_index: Current step index

    Returns:
        Updated sample for next step
    """
    return zimage_utils.step(model_output, sample, sigmas, step_index)


def shift_scale_latents_for_decode(latents: torch.Tensor) -> torch.Tensor:
    """
    Shift and scale latents for VAE decoding.

    Z-Image uses specific latent space transformations:
    - Shift by ZIMAGE_VAE_SHIFT_FACTOR
    - Scale by ZIMAGE_VAE_SCALING_FACTOR

    Args:
        latents: Model latents to decode

    Returns:
        VAE-ready latents
    """
    return zimage_utils.shift_scale_latents_for_decode(latents)


def trim_pad_embeds_and_mask(
    image_sequence_length: int,
    embed: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trim or pad embeddings to match image sequence length.

    Z-Image requires caption embeddings to be appropriately sized
    relative to the image sequence length.

    Args:
        image_sequence_length: Target sequence length
        embed: Caption embeddings [B, seq_len, dim]
        mask: Attention mask [B, seq_len]

    Returns:
        Tuple of (trimmed_embed, trimmed_mask)
    """
    return zimage_utils.trim_pad_embeds_and_mask(image_sequence_length, embed, mask)


# ============================================================================
# Z-Image-Edit Specific Utilities
# ============================================================================


def calculate_image_sequence_length(latents: torch.Tensor, patch_size: int = 2) -> int:
    """
    Calculate the sequence length for image latents after packing.

    For Z-Image-Edit with control images, we need to track the target
    image sequence length separately from control image sequence length.

    Args:
        latents: [B, C, F, H, W] latents
        patch_size: Spatial patch size (default: 2 for Z-Image)

    Returns:
        Sequence length (F * H * W) where H, W are in patch units
    """
    B, C, F, H, W = latents.shape
    # Z-Image uses patch_size=2, so each spatial dimension is divided by 2
    H_patches = H // patch_size
    W_patches = W // patch_size
    return F * H_patches * W_patches


def validate_control_latents(
    control_latents: torch.Tensor,
    target_latents: torch.Tensor,
) -> None:
    """
    Validate that control latents have compatible shape with target latents.

    Args:
        control_latents: [B, C, F, H, W] control latents
        target_latents: [B, C, F, H, W] target latents

    Raises:
        ValueError: If shapes are incompatible
    """
    if control_latents.shape[1] != target_latents.shape[1]:
        raise ValueError(
            f"Control latents channel dimension {control_latents.shape[1]} "
            f"does not match target latents {target_latents.shape[1]}"
        )

    logger.debug(
        f"Control latents shape: {control_latents.shape}, "
        f"Target latents shape: {target_latents.shape}"
    )


# ============================================================================
# Constants and Configuration
# ============================================================================

# Re-export VAE scale factor for convenience
VAE_SCALE_FACTOR = qwen_image_utils.VAE_SCALE_FACTOR

# Z-Image uses patch_size=2 for spatial patching
DEFAULT_PATCH_SIZE = 2

# Default model version for Z-Image-Edit
DEFAULT_MODEL_VERSION = "edit"
