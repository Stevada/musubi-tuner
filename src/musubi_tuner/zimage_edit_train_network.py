"""
Z-Image-Edit Training Script with Dual Pathway Control Image Support

This script implements training for Z-Image-Edit, which extends Z-Image with
dual pathway control/reference image conditioning:
- Pathway 1 (Latent): VAE-encoded control latents concatenated to noisy target
- Pathway 2 (Embedding): ViT-encoded control features embedded in text stream via VLM

Based on zimage_train_network.py with adaptations from qwen_image_train_network.py
"""

import argparse
from typing import Optional
import math

import torch
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_Z_IMAGE_EDIT,
    ARCHITECTURE_Z_IMAGE_EDIT_FULL,
)
from musubi_tuner.zimage import zimage_model, zimage_utils, zimage_autoencoder, zimage_config
from musubi_tuner.zimage import zimage_edit_utils
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.utils import model_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ZImageEditNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_Z_IMAGE_EDIT

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_Z_IMAGE_EDIT_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        )
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        self._i2v_training = False
        self._control_training = True  # Z-Image-Edit supports control images
        self.default_guidance_scale = 4.0  # Default CFG scale for edit mode

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        """
        Process sample prompts with control images for validation sampling.

        This replaces Qwen3 with Qwen2.5-VL and handles control image encoding
        through both pathways.
        """
        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # Load Qwen2.5-VL instead of Qwen3
        vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
        logger.info(f"Loading Qwen2.5-VL for Z-Image-Edit with dtype={vl_dtype}")
        tokenizer, text_encoder = zimage_edit_utils.load_qwen2_5_vl(
            args.text_encoder, vl_dtype, device, disable_mmap=True
        )
        vl_processor = zimage_edit_utils.load_vl_processor()
        text_encoder.eval()

        # Encode prompts with control images
        logger.info("Encoding with Qwen2.5-VL (VLM)")

        sample_prompts_te_outputs = {}  # (prompt, control_image_paths) -> (embed, mask)
        control_image_nps = {}  # control_image_path -> numpy array

        def embed_key_fn(p, ctrl_img_paths):
            """Create unique key for cached embeddings."""
            return p if ctrl_img_paths is None else (p, tuple(ctrl_img_paths))

        with torch.amp.autocast(device_type=device.type, dtype=vl_dtype), torch.no_grad():
            for prompt_dict in prompts:
                width, height = prompt_dict.get("width", 1024), prompt_dict.get("height", 1024)
                width = (width // 16) * 16
                height = (height // 16) * 16

                # Load control images if provided
                control_image_paths = None
                control_image_tensors = None
                if "control_image_path" in prompt_dict and len(prompt_dict["control_image_path"]) > 0:
                    control_image_paths = prompt_dict["control_image_path"]
                    control_image_tensors = []
                    for path in control_image_paths:
                        tensor, np_img, _ = zimage_edit_utils.preprocess_control_image(
                            path, resize_to_official=True, resize_size=(width, height)
                        )
                        # Z-Image uses RGB only (no alpha channel)
                        tensor = tensor[:, :3, :, :]
                        np_img = np_img[:, :, :3]
                        control_image_tensors.append(tensor)
                        control_image_nps[path] = np_img
                    prompt_dict["control_image_tensors"] = control_image_tensors

                if "negative_prompt" not in prompt_dict:
                    prompt_dict["negative_prompt"] = ""

                # Encode prompts (with or without control images)
                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                    embed_key = embed_key_fn(p, control_image_paths)
                    if p is None or embed_key in sample_prompts_te_outputs:
                        continue

                    logger.info(f"cache Text Encoder outputs for prompt: {p} with images: {control_image_paths}")

                    if control_image_paths is None or len(control_image_paths) == 0:
                        # Text-only: fallback to pure text encoding
                        # Note: Qwen2.5-VL can encode text without images
                        control_images = None
                    else:
                        control_images = [control_image_nps[c] for c in control_image_paths]

                    # Encode with VLM (Pathway 2: ViT + text)
                    embed, mask = zimage_edit_utils.get_text_embeds_with_image(
                        vl_processor, text_encoder, p, control_images, model_version="edit"
                    )
                    embed = embed.cpu()
                    mask = mask.cpu()
                    sample_prompts_te_outputs[embed_key] = (embed, mask)

        del tokenizer, text_encoder
        clean_memory_on_device(device)

        # Prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            control_image_paths = None
            if "control_image_path" in prompt_dict and len(prompt_dict["control_image_path"]) > 0:
                control_image_paths = prompt_dict["control_image_path"]

            # Get embeddings
            prompt = prompt_dict.get("prompt", "")
            embed_key = embed_key_fn(prompt, control_image_paths)
            embed, mask = sample_prompts_te_outputs[embed_key]
            prompt_dict_copy["cap_feats"] = embed
            prompt_dict_copy["cap_mask"] = mask

            negative_prompt = prompt_dict.get("negative_prompt", "")
            embed_key = embed_key_fn(negative_prompt, control_image_paths)
            negative_embed, negative_mask = sample_prompts_te_outputs[embed_key]
            prompt_dict_copy["negative_cap_feats"] = negative_embed
            prompt_dict_copy["negative_cap_mask"] = negative_mask

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """
        Z-Image-Edit inference with dual pathway control.

        This extends Z-Image inference to handle:
        1. Control latents (Pathway 1): VAE-encoded, concatenated to noisy target
        2. Control features (Pathway 2): ViT-encoded, embedded in text stream
        """
        model: zimage_model.ZImageTransformer2DModel = accelerator.unwrap_model(transformer)
        device = accelerator.device

        if cfg_scale is None:
            cfg_scale = 4.0

        # Get embeddings (already contain visual features from VLM if control images provided)
        embed = sample_parameter["cap_feats"].to(device=device, dtype=torch.bfloat16)
        mask = sample_parameter["cap_mask"].to(device=device, dtype=torch.bool)

        do_cfg = cfg_scale > 1.0
        if do_cfg:
            negative_embed = sample_parameter["negative_cap_feats"].to(device=device, dtype=torch.bfloat16)
            negative_mask = sample_parameter["negative_cap_mask"].to(device=device, dtype=torch.bool)
        else:
            negative_embed = None
            negative_mask = None

        # Prepare control latents (Pathway 1)
        control_latent = None
        if "control_image_tensors" in sample_parameter and len(sample_parameter["control_image_tensors"]) > 0:
            logger.info("Encoding control images via VAE (Pathway 1)")
            control_image_tensors = sample_parameter["control_image_tensors"]
            vae.to(device)
            vae.eval()

            with torch.no_grad():
                control_latents = [vae.encode(t.to(device, vae.dtype)) for t in control_image_tensors]
            # Scale/shift control latents same as target latents
            shift = zimage_config.ZIMAGE_VAE_SHIFT_FACTOR
            scale = zimage_config.ZIMAGE_VAE_SCALING_FACTOR
            control_latents = [(cl - shift) * scale for cl in control_latents]
            control_latents = [cl.to(torch.bfloat16) for cl in control_latents]

            # Pack and concatenate control latents
            control_latents_packed = []
            for cl in control_latents:
                cl = cl.unsqueeze(2)  # Add frame dim: B, C, H, W -> B, C, 1, H, W
                cl_packed = zimage_edit_utils.pack_latents(cl)  # B, C, 1, H, W -> B, L, C
                control_latents_packed.append(cl_packed)
            control_latent = torch.cat(control_latents_packed, dim=1)  # Concat in sequence dim
            control_latent = control_latent.to(device, dtype=torch.bfloat16)

            vae.to("cpu")
            clean_memory_on_device(device)

        # Prepare noise latents
        vae_scale = zimage_config.ZIMAGE_VAE_SCALE_FACTOR * 2
        height_latent = 2 * (int(height) // vae_scale)
        width_latent = 2 * (int(width) // vae_scale)
        shape = (1, model.in_channels, height_latent, width_latent)
        latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

        # Trim embeddings to match image sequence length
        image_sequence_length = (height_latent // model.all_patch_size[0]) * (width_latent // model.all_patch_size[0])
        embed, _ = zimage_edit_utils.trim_pad_embeds_and_mask(image_sequence_length, embed, mask)
        mask = None  # No attention mask needed after trimming
        if do_cfg and negative_embed is not None:
            negative_embed, _ = zimage_edit_utils.trim_pad_embeds_and_mask(image_sequence_length, negative_embed, negative_mask)

        # Prepare timesteps
        timesteps, sigmas = zimage_edit_utils.get_timesteps_sigmas(sample_steps, discrete_flow_shift)
        timesteps = timesteps.to(device)
        sigmas = sigmas.to(device)

        # Z-Image-Edit inference loop
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000  # Reverse for z-image

            latent_model_input = latents.to(model.dtype)
            latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dimension [B, C, 1, H, W]

            # Pack latents for sequence processing
            latent_model_input_packed = zimage_edit_utils.pack_latents(latent_model_input)

            # Concatenate control latents if available (Pathway 1)
            if control_latent is not None:
                latent_model_input_packed = torch.cat([latent_model_input_packed, control_latent], dim=1)

            with accelerator.autocast(), torch.no_grad():
                # Forward pass with dual pathway:
                # - latent_model_input_packed contains noisy target + control latents (Pathway 1)
                # - embed contains text + visual features from VLM (Pathway 2)
                model_out = transformer(latent_model_input_packed, timestep, embed, mask)

            # Remove control portion from output
            if control_latent is not None:
                img_seq_len = image_sequence_length
                model_out = model_out[:, :img_seq_len]

            # CFG
            if do_cfg:
                # Prepare negative input
                latent_model_input_neg = latent_model_input_packed
                if control_latent is not None:
                    # Use same control latents for negative
                    pass  # Already concatenated

                with accelerator.autocast(), torch.no_grad():
                    negative_model_out = transformer(latent_model_input_neg, timestep, negative_embed, None)

                if control_latent is not None:
                    negative_model_out = negative_model_out[:, :img_seq_len]

                noise_pred = negative_model_out + cfg_scale * (model_out - negative_model_out)
            else:
                noise_pred = model_out

            # Unpack and update latents
            noise_pred = zimage_edit_utils.unpack_latents(noise_pred, height, width)
            noise_pred = -noise_pred.squeeze(2)  # Remove frame dimension and invert sign
            latents = zimage_edit_utils.step(noise_pred.to(torch.float32), latents, sigmas, i)

        # Decode
        latents = latents.to(vae.dtype)
        vae.to(device)
        vae.eval()

        logger.info(f"Decoding image from latents: {latents.shape}")
        latents = zimage_edit_utils.shift_scale_latents_for_decode(latents)
        with torch.no_grad():
            pixels = vae.decode(latents)

        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)

        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.unsqueeze(2)  # B C F H W. F=1.
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae
        logger.info(f"Loading VAE model from {vae_path}")
        vae = zimage_autoencoder.load_autoencoder_kl(vae_path, device="cpu", disable_mmap=True)
        return vae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        # Load Z-Image model (no changes needed - compatible with dual pathway!)
        model = zimage_model.load_zimage_model(
            device=loading_device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
            use_16bit_for_attention=not args.use_32bit_attention,
        )
        return model

    def compile_transformer(self, args, transformer):
        model: zimage_model.ZImageTransformer2DModel = transformer
        # Compile blocks
        return model_utils.compile_transformer(
            args, model, [model.noise_refiner, model.context_refiner, model.layers], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        # Transform VAE latents to Model latents
        # Model Latents = (VAE Latents - shift) * scale
        shift = zimage_config.ZIMAGE_VAE_SHIFT_FACTOR
        scale = zimage_config.ZIMAGE_VAE_SCALING_FACTOR
        latents = (latents - shift) * scale
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        """
        DiT forward pass with dual pathway control.

        Handles:
        1. Pathway 1: VAE-encoded control latents concatenated to noisy target
        2. Pathway 2: VLM-encoded text+visual features
        """
        model: zimage_model.ZImageTransformer2DModel = accelerator.unwrap_model(transformer)
        bsize = latents.shape[0]

        # latents: [B, C, H, W]
        # noisy_model_input: [B, C, H, W]
        image_sequence_length = (latents.shape[2] // model.all_patch_size[0]) * (latents.shape[3] // model.all_patch_size[0])

        # Add frame dimension F=1
        noisy_model_input = noisy_model_input.unsqueeze(2)  # [B, C, 1, H, W]

        # PATHWAY 1: VAE-encoded control latents
        num_control_images = 0
        while f"latents_control_{num_control_images}" in batch:
            num_control_images += 1

        if num_control_images > 0:
            latents_control = []
            for i in range(num_control_images):
                lc = batch[f"latents_control_{i}"]  # B, C, 1, H, W (F=1 for images)
                # No need to add frame dim - already has it
                lc_packed = zimage_edit_utils.pack_latents(lc)  # B, L, C
                latents_control.append(lc_packed)
            latents_control = torch.cat(latents_control, dim=1)  # Concat all controls

            # Pack noisy input and concatenate with control
            noisy_model_input_packed = zimage_edit_utils.pack_latents(noisy_model_input)
            noisy_model_input_packed = torch.cat([noisy_model_input_packed, latents_control], dim=1)
        else:
            noisy_model_input_packed = zimage_edit_utils.pack_latents(noisy_model_input)

        # PATHWAY 2: VLM-encoded text with visual features
        # The embeddings already contain visual features from Qwen2.5-VL
        vl_embed = batch["vl_embed"]  # list[torch.Tensor] - each is [Seq, Dim]
        txt_seq_lens = [x.shape[0] for x in vl_embed]

        max_len = max(txt_seq_lens)
        # if not split_attn, we need to make attention mask
        if not args.split_attn and bsize > 1:
            padded_len = math.ceil((max_len + image_sequence_length) / zimage_config.SEQ_MULTI_OF) * zimage_config.SEQ_MULTI_OF
            max_len = int(padded_len) - image_sequence_length
            vl_mask = torch.zeros(bsize, max_len, dtype=torch.bool, device=vl_embed[0].device)
            for i, x in enumerate(txt_seq_lens):
                vl_mask[i, :x] = True
        else:
            vl_mask = None  # if split_attn, mask is not used

        vl_embed = [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in vl_embed]
        vl_embed = torch.stack(vl_embed, dim=0)  # B, L, D

        # Timesteps
        t_input = (1000.0 - timesteps) / 1000.0

        # Prepare inputs on device
        noisy_model_input_packed = noisy_model_input_packed.to(device=accelerator.device, dtype=network_dtype)
        vl_embed = vl_embed.to(device=accelerator.device, dtype=network_dtype)
        vl_mask = vl_mask.to(device=accelerator.device) if vl_mask is not None else None
        t_input = t_input.to(device=accelerator.device, dtype=network_dtype)

        # Enable grad for gradient checkpointing
        if args.gradient_checkpointing:
            noisy_model_input_packed.requires_grad_(True)
            vl_embed.requires_grad_(True)

        # Forward pass with dual pathway
        # - noisy_model_input_packed: [noisy_target_latent, control_latent] (Pathway 1)
        # - vl_embed: text + visual features from VLM (Pathway 2)
        with accelerator.autocast():
            model_pred = transformer(x=noisy_model_input_packed, t=t_input, cap_feats=vl_embed, cap_mask=vl_mask)

        # Remove control portion from predictions
        if num_control_images > 0:
            img_seq_len = image_sequence_length
            model_pred = model_pred[:, :img_seq_len]

        # Unpack predictions to spatial format
        lat_h = latents.shape[2]
        lat_w = latents.shape[3]
        model_pred = zimage_edit_utils.unpack_latents(
            model_pred,
            lat_h * zimage_config.ZIMAGE_VAE_SCALE_FACTOR,
            lat_w * zimage_config.ZIMAGE_VAE_SCALE_FACTOR,
            vae_scale_factor=zimage_config.ZIMAGE_VAE_SCALE_FACTOR,
        )

        # model_pred: [B, C, 1, H, W]
        model_pred = model_pred.squeeze(2)  # [B, C, H, W]

        # Target: Z-Image uses inverted flow matching
        target = latents - noise

        return model_pred, target

    # endregion model specific


def zimage_edit_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Z-Image-Edit specific parser setup"""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT")
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen2.5-VL text encoder checkpoint path")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder (VLM) model")
    parser.add_argument(
        "--use_32bit_attention",
        action="store_true",
        help="use 32-bit precision for attention computations in DiT model even when using mixed precision (original behavior)",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = zimage_edit_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is not None:
        logger.warning("vae_dtype is not used in Z-Image-Edit architecture (always float32)")

    trainer = ZImageEditNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
