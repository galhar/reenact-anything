# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Apply Learned Motion Embeddings to Target Images.

This script applies previously learned motion embeddings (from train_reenact.py) to target images
to generate videos with transferred motion. The motion embeddings are loaded from a checkpoint
and used to condition the Stable Video Diffusion model during inference.

Usage:
    python inference_only.py \\
        --motion_embedding_path <path_to_learned_embeddings> \\
        --validation_images_path <path_to_target_images> \\
        --output_dir <output_directory>

The script will:
1. Load the learned motion embedding from the specified path
2. Generate videos for each image in the validation directory
3. Save the generated videos in the output directory
"""
import argparse
import logging
import os
from pathlib import Path
from pprint import pprint
from urllib.parse import urlparse

import cv2
import diffusers
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import check_min_version, is_wandb_available
from einops import rearrange
from PIL import Image
from tqdm.auto import tqdm

import wandb  # Can avoid it here if we really want to
from utils.embds_inversion_utils import (
    ImageEmbeddingWrapper,
    allow_motion_embedding,
    override_pipeline_call,
)
from utils.video_utils import SimpleImagesDataset, log_decoded_video

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply learned motion embeddings to target images for video generation."
    )
    parser.add_argument(
        "--motion_embedding_path",
        required=True,
        type=str,
        help="Path to the learned motion embedding. Can be either:\n"
             "  - A directory containing checkpoints (e.g., ./outputs/jumping_jacks/): "
             "    The script will automatically find the most recent checkpoint\n"
             "  - A specific .pt file (e.g., ./outputs/jumping_jacks/motion_embedding_3000.pt): "
             "    Direct path to a motion embedding file",
    )
    parser.add_argument(
        "--validation_images_path",
        type=str,
        default=None,
        help="Directory containing target images to apply the motion to. "
             "A video will be generated for each image in this directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where generated videos will be saved. "
             "If not specified, videos will be saved in <motion_embedding_path>/inference_images/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible video generation. Default: 0",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help="Optional logging backend: 'wandb' for Weights & Biases or 'tensorboard' for TensorBoard. "
             "If not specified, no logging will be performed.",
    )
    args = parser.parse_args()

    return args


def main(
    motion_embedding_path,
    validation_images_path,
    seed=0,
    report_to=None,
    output_dir=None,
    pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid",
    num_frames=14,
    num_validation_images=30,
):
    """
    Apply learned motion embeddings to target images.
    
    Args:
        motion_embedding_path: Path to learned motion embedding (directory or .pt file)
        validation_images_path: Directory containing target images
        seed: Random seed for reproducibility
        report_to: Logging backend ('wandb' or 'tensorboard')
        output_dir: Output directory for generated videos (defaults to motion_embedding_path if not specified)
        pretrained_model_name_or_path: HuggingFace model identifier for SVD
        num_frames: Number of frames to generate (default: 14)
        num_validation_images: Maximum number of images to process (default: 30)
    """
    # Use motion_embedding_path as output_dir if not specified
    if output_dir is None:
        output_dir = motion_embedding_path
    logging_dir = os.path.join(output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with=report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    if report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    val_dataset = SimpleImagesDataset(
        validation_images_path,
        max_images_n=num_validation_images,
        device=accelerator.device,
    )

    # Get the most recent checkpoint
    if os.path.isfile(motion_embedding_path):
        path = motion_embedding_path
    else:
        dirs = os.listdir(motion_embedding_path)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        path = os.path.join(motion_embedding_path, path)

    if path is None:
        accelerator.print(
            f"Checkpoint '{motion_embedding_path}' does not exist. Starting a new training run."
        )
        raise ValueError("resume from checkpoint does not exists")

    accelerator.print(f"Resuming from checkpoint {path}")

    image_embds_wrap = ImageEmbeddingWrapper(torch.Tensor([0]))
    image_embds_wrap.requires_grad_(False)
    image_embds_wrap.to(device=accelerator.device, dtype=torch.float32)
    image_embds_wrap.load_tensor(path)
    global_step = int(path.split("-")[1])

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("reenact_inference", config=locals())

    # sample images!
    logger.info(f"Running validation... \n Generating {num_validation_images} videos.")

    # The models need unwrapping because for compatibility in distributed training mode.
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        revision="main",
        torch_dtype=torch.float32,
    ).to(accelerator.device)

    allow_motion_embedding(pipeline.unet)

    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.image_encoder.requires_grad_(False)

    # run inference
    inference_save_dir = os.path.join(output_dir, "inference_images")

    if not os.path.exists(inference_save_dir):
        os.makedirs(inference_save_dir)

    with torch.autocast(
        str(accelerator.device).replace(":0", ""),
        # enabled=accelerator.mixed_precision == "fp16",
    ):
        with torch.no_grad():
            for val_img_idx in tqdm(range(len(val_dataset))):
                sample = val_dataset[val_img_idx]
                cond_name = sample["name"]
                cond_img = sample["frame"]
                print("[*] Running inference for image {}".format(cond_name))

                video_frames = override_pipeline_call(
                    pipeline,
                    cond_img,
                    num_frames=num_frames,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.02,
                    motion_features=image_embds_wrap(),
                    max_guidance_scale=1,  # Dont do classifeir free guidance for now
                    generator=generator,
                ).frames[0]

                log_decoded_video(
                    global_step,
                    num_frames,
                    val_img_idx,
                    inference_save_dir,
                    video_frames,
                    video_desc="val_" + cond_name,
                    report_to=report_to,
                )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()

    main(
        motion_embedding_path=args.motion_embedding_path,
        validation_images_path=args.validation_images_path,
        seed=args.seed,
        report_to=args.report_to,
        output_dir=args.output_dir,
    )
