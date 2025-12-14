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
Motion-Textual Inversion for Semantic Video Motion Transfer.

This script performs motion-textual inversion on a reference video to learn motion embeddings.
The learned embeddings capture semantic motion patterns that can be applied to different target images.

Usage:
    python train_reenact.py \\
        --video_to_inverse <path_to_reference_video> \\
        --validation_images_path <path_to_validation_images> \\
        --output_dir <output_directory>

The script will:
1. Learn motion embeddings from the reference video using diffusion model loss
2. Save checkpoints periodically during training
3. Run validation on provided images to visualize motion transfer
4. Save the final motion embedding for use with inference_only.py
"""
import argparse
import datetime
import inspect
import logging
import math
import os

import accelerate
import diffusers
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKLTemporalDecoder,
    StableVideoDiffusionPipeline,
    UNetSpatioTemporalConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from torch.utils.data import RandomSampler
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

try:
    import wandb  # Can avoid it here if we really want to
except ImportError:
    pass

from utils.embds_inversion_utils import (
    ImageEmbeddingWrapper,
    allow_motion_embedding,
    initialize_image_embedding,
    override_pipeline_call,
    pipeline_decode_latents,
)
from utils.train_utils import (
    _get_add_time_ids,
    encode_image,
    gpu_stats,
    log_grads,
    rand_log_normal,
    tensor_to_vae_latent,
)
from utils.video_utils import (
    MotionSingleVideoDataset,
    SimpleImagesDataset,
    log_decoded_video,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Motion-Textual Inversion: Learn motion embeddings from a reference video."
    )
    parser.add_argument(
        "--video_to_inverse",
        required=True,
        type=str,
        help="Path to the reference motion video file (e.g., ./reference_motion_video/jumping_jacks/jumping_jacks.mp4). "
             "This video will be used to learn the motion embeddings.",
    )
    parser.add_argument(
        "--num_tokens_in_motion_features",
        type=int,
        default=5,
        help="Number of tokens in the motion features embedding. Higher values may capture more detailed motion "
             "but require more training. Default: 5",
    )
    parser.add_argument(
        "--validation_images_path",
        type=str,
        default=None,
        help="Directory containing validation images to test motion transfer during training. "
             "These images will be used to generate videos at validation steps to visualize how well the learned "
             "motion transfers to different targets.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l1_weighted", "l1", "l2"],
        help="Loss function type: 'l2' (MSE, default), 'l1' (MAE), or 'l1_weighted' (MAE with motion-weighted pixels).",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every N training steps. During validation, videos are generated for the validation images "
             "to monitor training progress. Default: 500",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where motion embeddings and checkpoints will be saved. "
             "If not specified, a timestamped directory will be created in ./outputs/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Default: 0",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform. More steps may improve quality but take longer. Default: 5000",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every N training steps. Checkpoints can be used later for inference. Default: 500",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard"],
        help="Logging backend: 'wandb' for Weights & Biases or 'tensorboard' for TensorBoard. Default: wandb",
    )

    args = parser.parse_args()

    return args


def load_models(
    pretrained_model_name_or_path,
    output_dir,
    gradient_accumulation_steps,
    mixed_precision,
    report_to,
    enable_xformers_memory_efficient_attention,
    seed,
    allow_tf32,
    gradient_checkpointing,
):
    logging_dir = os.path.join(output_dir, "/logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

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

    # Load img encoder, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="feature_extractor",
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="image_encoder",
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        variant="fp16",
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Override calls in unet:
    allow_motion_embedding(unet)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):

            for i, model in enumerate(models):
                if isinstance(model, ImageEmbeddingWrapper):
                    model.save_pretrained(
                        os.path.join(output_dir, "image_embedding_wrapper")
                    )
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, ImageEmbeddingWrapper):
                    model.load_tensor(
                        os.path.join(input_dir, "image_embedding_wrapper")
                    )

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    return accelerator, image_encoder, feature_extractor, vae, unet, weight_dtype


def init_optimizer_and_scheduler(
    accelerator,
    train_dataloader,
    use_8bit_adam,
    learning_rate,
    adam_beta1,
    adam_beta2,
    adam_weight_decay,
    adam_epsilon,
    gradient_accumulation_steps,
    max_train_steps,
    num_train_epochs,
    lr_scheduler,
    lr_warmup_steps,
    image_embds_wrap,
):
    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        image_embds_wrap.get_parameters(),
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    return optimizer, lr_scheduler, overrode_max_train_steps


def train_step(
    batch,
    weight_dtype,
    image_embds_wrap,
    unet,
    vae,
    feature_extractor,
    image_encoder,
    optimizer,
    accelerator,
    per_gpu_batch_size,
    gradient_accumulation_steps,
    global_step,
    lr_scheduler,
    loss_type,
):
    with accelerator.accumulate(image_embds_wrap):
        # first, convert images to latent space.
        pixel_values = (
            batch["frames"].to(weight_dtype).to(accelerator.device, non_blocking=True)
        )
        conditional_pixel_values = pixel_values[:, 0:1, :, :, :]

        latents = tensor_to_vae_latent(pixel_values, vae)

        bsz = latents.shape[0]

        cond_noise = torch.randn_like(conditional_pixel_values)
        cond_sigmas = rand_log_normal(
            shape=[
                bsz,
            ],
            loc=-3.0,
            scale=0.5,
        ).to(latents)
        noise_aug_strength = cond_sigmas[0]  # TODO: support batch > 1
        cond_sigmas = cond_sigmas[:, None, None, None, None]

        conditional_pixel_values = cond_noise * cond_sigmas + conditional_pixel_values
        conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[
            :, 0, :, :, :
        ]
        conditional_latents = conditional_latents / vae.config.scaling_factor

        # Sample a random timestep for each image
        # P_mean=0.7 P_std=1.6
        noise = torch.randn_like(latents)
        sigmas = rand_log_normal(
            shape=[
                bsz,
            ],
            loc=2.8,
            scale=1.6,
        ).to(latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        sigmas = sigmas[:, None, None, None, None]
        noisy_latents = latents + noise * sigmas
        timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(
            accelerator.device
        )

        inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

        # Get the text embedding for conditioning.
        encoder_hidden_states = encode_image(
            pixel_values[:, 0, :, :, :].to(
                device=accelerator.device, dtype=weight_dtype
            ),
            feature_extractor,
            image_encoder,
        )

        # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
        # However, I am unable to fully align with the calculation method of the motion score,
        # so I adopted this approach. The same applies to the 'fps' (frames per second).
        added_time_ids = _get_add_time_ids(
            unet,
            7,  # fixed
            127,  # motion_bucket_id = 127, fixed
            noise_aug_strength,  # noise_aug_strength == cond_sigmas
            encoder_hidden_states.dtype,
            bsz,
        )
        added_time_ids = added_time_ids.to(latents.device)

        # Concatenate the `conditional_latents` with the `noisy_latents`.
        conditional_latents = conditional_latents.unsqueeze(1).repeat(
            1, noisy_latents.shape[1], 1, 1, 1
        )
        inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

        # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
        target = latents
        model_pred = unet(  # CHANGED HERE "encoder_hidden_states" to image_embds_wrap() # removed () since now we activate at input
            inp_noisy_latents,
            timesteps,
            image_embds_wrap,
            added_time_ids=added_time_ids,
        ).sample

        # Denoise the latents
        c_out = -sigmas / ((sigmas**2 + 1) ** 0.5)
        c_skip = 1 / (sigmas**2 + 1)
        denoised_latents = model_pred * c_out + c_skip * noisy_latents
        weighing = (1 + sigmas**2) * (sigmas**-2.0)

        # loss
        diff_func = None
        if loss_type == "l1_weighted":
            # Calculate std per pixel across frames
            # target shape is [batch, frames, channels, height, width]
            pixel_std = torch.std(target, dim=1).mean(
                dim=1
            )  # Get std across frames dimension, mean of that over the channels per spatial location

            # Normalize std to [0,1] range
            min_std = torch.min(pixel_std)
            max_std = torch.max(pixel_std)
            normalized_std = (pixel_std - min_std) / (max_std - min_std + 1e-8)

            # Scale to [1,11] range
            pixel_weights = 1.0 + 1000.0 * normalized_std

            # Expand weights to match target shape
            pixel_weights = pixel_weights.expand_as(target)

            diff_func = lambda x, y: pixel_weights * torch.nn.functional.l1_loss(x, y)
        elif loss_type == "l1":
            diff_func = torch.nn.functional.l1_loss
        else:
            diff_func = torch.nn.functional.mse_loss

        loss = torch.mean(
            (
                weighing.float() * diff_func(denoised_latents.float(), target.float())
            ).reshape(target.shape[0], -1),
            dim=1,
        )
        loss = loss.mean()

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(per_gpu_batch_size)).mean()
        train_loss_addition = avg_loss.item() / gradient_accumulation_steps

        motion_embds_before_update = (
            image_embds_wrap.clone().detach()
        )  # For later debugging
        # Backpropagate
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        # log_grads(
        #     image_embds_wrap - motion_embds_before_update,
        #     optimizer,
        #     global_step,
        #     accelerator,
        # )
        optimizer.zero_grad()

        return loss, train_loss_addition, denoised_latents


def offload_training(
    image_encoder,
    vae,
    unet,
    pretrained_model_name_or_path,
    weight_dtype,
    accelerator,
):
    # Move to CPU to reduce GPU memory during validation
    print("Offloading training models from GPU to CPU during validation")
    image_encoder.to("cpu")
    vae.to("cpu")
    unet.to("cpu")
    torch.cuda.empty_cache()

    # The models need unwrapping because for compatibility in distributed training mode.
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    allow_motion_embedding(pipeline.unet)

    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.image_encoder.requires_grad_(False)

    return pipeline


def reload_training(pipeline, image_encoder, vae, unet, accelerator):
    del pipeline
    torch.cuda.empty_cache()

    # Reload from cpu to GPU after validation
    print("Reload training models from CPU to GPU after validation")
    image_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    unet.to(accelerator.device)


def run_validation(
    pipeline,
    generator,
    val_dataset,
    batch,
    image_embds_wrap,
    denoised_latents,
    accelerator,
    output_dir,
    global_step,
    num_frames,
    report_to,
    height,
    width,
):
    # run inference
    val_save_dir = os.path.join(output_dir, "validation_images")
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)

    with torch.autocast(
        str(accelerator.device).replace(":0", ""),
        enabled=accelerator.mixed_precision == "fp16",
    ):
        with torch.no_grad():
            # See the training sample:
            training_decoded = pipeline_decode_latents(
                pipeline,
                denoised_latents.detach(),
                num_frames,
                decode_chunk_size=num_frames // 2 + 1,
            )[0]
            log_decoded_video(
                global_step,
                num_frames,
                0,
                val_save_dir,
                training_decoded,
                video_desc="train_img",
                report_to=report_to,
            )

            # Now validation samples:
            for val_img_idx in range(len(val_dataset) + 1):
                if val_img_idx == 0:
                    # Always start with the training img itself
                    cond_name = "train_cond"
                    cond_img = batch["frame_0"]
                else:
                    img_idx = val_img_idx - 1
                    sample = val_dataset[img_idx]
                    cond_name = sample["name"]
                    cond_img = sample["frame"]

                num_frames = num_frames
                video_frames = override_pipeline_call(
                    pipeline,
                    cond_img,
                    height=height,
                    width=width,
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
                    val_save_dir,
                    video_frames,
                    video_desc="val_" + cond_name,
                    report_to=report_to,
                )

    del video_frames
    del training_decoded


def gather_function_args(function, locals):
    # Get function arguments using inspect
    args = inspect.signature(function).parameters
    params = {name: value for name, value in locals.items() if name in args}
    return params


def inverse_motion(
    video_to_inverse,
    output_dir=None,
    validation_steps=500,
    checkpointing_steps=500,
    max_train_steps=5000,
    validation_images_path=None,
    num_tokens_in_motion_features=5,
    pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid",
    num_frames=14,
    width=1024,
    height=576,
    max_num_validation_images=12,
    seed=0,
    per_gpu_batch_size=1,
    num_train_epochs=100,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    enable_xformers_memory_efficient_attention=True,
    report_to=None,
    allow_tf32=False,
    mixed_precision="fp16",
    num_workers=0,
    use_8bit_adam=False,
    learning_rate=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_weight_decay=0.01,
    adam_epsilon=1e-8,
    lr_warmup_steps=500,
    scale_lr=False,
    lr_scheduler="constant",
    loss_type="l2",
):
    """
    Perform motion-textual inversion on a reference video.
    
    This function learns motion embeddings from a reference video by optimizing motion-text tokens
    using diffusion model loss. The learned embeddings capture semantic motion patterns that can
    be applied to different target images.
    
    Args:
        video_to_inverse: Path to the reference motion video
        output_dir: Directory to save checkpoints and motion embeddings
        validation_steps: Run validation every N steps
        checkpointing_steps: Save checkpoint every N steps
        max_train_steps: Total number of training steps
        validation_images_path: Directory with images for validation during training
        num_tokens_in_motion_features: Number of tokens in motion embedding
        pretrained_model_name_or_path: HuggingFace model identifier for SVD
        num_frames: Number of frames in generated videos
        width: Video width
        height: Video height
        max_num_validation_images: Maximum number of validation images to process
        seed: Random seed
        loss_type: Loss function type ('l2', 'l1', or 'l1_weighted')
        report_to: Logging backend ('wandb' or 'tensorboard')
        ... (other training hyperparameters)
    
    Returns:
        None. Saves motion embeddings to output_dir.
    """

    # Hack to add it to args with minimal changes to the script
    if output_dir is None:
        video_name = os.path.basename(video_to_inverse)
        output_dir = add_timestamp_suff(
            os.path.join("./outputs", video_name.split(".")[0])
        )
    os.makedirs(output_dir, exist_ok=True)

    accelerator, image_encoder, feature_extractor, vae, unet, weight_dtype = (
        load_models(
            pretrained_model_name_or_path,
            output_dir,
            gradient_accumulation_steps,
            mixed_precision,
            report_to,
            enable_xformers_memory_efficient_attention,
            seed,
            allow_tf32,
            gradient_checkpointing,
        )
    )

    # ============ DataLoaders ============
    global_batch_size = per_gpu_batch_size * accelerator.num_processes

    val_dataset = SimpleImagesDataset(
        validation_images_path,
        width=width,
        height=height,
        max_images_n=max_num_validation_images,
        device=accelerator.device,
    )
    train_dataset = MotionSingleVideoDataset(
        video_to_inverse,
        size=(num_frames, width, height),
        device=accelerator.device,
    )
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=per_gpu_batch_size,
        num_workers=num_workers,
    )

    # Initialize motion token:
    print("[*] Initializing the motion features ")
    image_embedding = initialize_image_embedding(
        train_dataset[0]["frames"],
        pretrained_model_name_or_path,
        num_tokens_in_motion_features,
    )
    image_embds_wrap = ImageEmbeddingWrapper(image_embedding)
    image_embds_wrap.requires_grad_(True)
    image_embds_wrap.to(dtype=torch.float32)

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * per_gpu_batch_size
            * accelerator.num_processes
        )

    # ============ Optimizer and Scheduler ============
    optimizer, lr_scheduler, overrode_max_train_steps = init_optimizer_and_scheduler(
        accelerator,
        train_dataloader,
        use_8bit_adam,
        learning_rate,
        adam_beta1,
        adam_beta2,
        adam_weight_decay,
        adam_epsilon,
        gradient_accumulation_steps,
        max_train_steps,
        num_train_epochs,
        lr_scheduler,
        lr_warmup_steps,
        image_embds_wrap,
    )

    # ============ Training setup ============

    # Prepare everything with our `accelerator`
    image_embds_wrap, unet, optimizer, lr_scheduler, train_dataloader = (
        accelerator.prepare(
            image_embds_wrap, unet, optimizer, lr_scheduler, train_dataloader
        )
    )

    # attribute handling for models using DDP
    if isinstance(
        unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        unet = unet.module

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "SVDXtend", config=gather_function_args(inverse_motion, locals())
        )

    # Train!
    total_batch_size = (
        per_gpu_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    gpu_stats(logger)
    global_step = 0
    first_epoch = 0

    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            loss, train_loss_addition, denoised_latents = train_step(
                batch,
                weight_dtype,
                image_embds_wrap(),
                unet,
                vae,
                feature_extractor,
                image_encoder,
                optimizer,
                accelerator,
                per_gpu_batch_size,
                gradient_accumulation_steps,
                global_step,
                lr_scheduler,
                loss_type,
            )
            train_loss += train_loss_addition

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)

                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % checkpointing_steps == 0:
                        image_embds_wrap.save_pretrained(
                            os.path.join(
                                output_dir, f"motion_embedding_{global_step}.pt"
                            )
                        )

                    # sample images!
                    if (global_step % validation_steps == 0) or (global_step == 1):
                        logger.info(
                            f"Running validation... \n Generating {max_num_validation_images} videos."
                        )

                        pipeline = offload_training(
                            image_encoder,
                            vae,
                            unet,
                            pretrained_model_name_or_path,
                            weight_dtype,
                            accelerator,
                        )

                        run_validation(
                            pipeline,
                            generator,
                            val_dataset,
                            batch,
                            image_embds_wrap,
                            denoised_latents,
                            accelerator,
                            output_dir,
                            global_step,
                            num_frames,
                            report_to,
                            height,
                            width,
                        )

                        reload_training(
                            pipeline,
                            image_encoder,
                            vae,
                            unet,
                            accelerator,
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        image_embds_wrap = accelerator.unwrap_model(image_embds_wrap)
        image_embds_wrap.save_pretrained(output_dir)

    accelerator.end_training()


def add_timestamp_suff(dirname):
    now = datetime.datetime.now()
    return dirname + "_" + now.strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == "__main__":
    args = parse_args()

    # NOTE - normalized to [-1, 1] input
    inverse_motion(
        video_to_inverse=args.video_to_inverse,
        output_dir=args.output_dir,
        validation_steps=args.validation_steps,
        checkpointing_steps=args.checkpointing_steps,
        max_train_steps=args.max_train_steps,
        validation_images_path=args.validation_images_path,
        num_tokens_in_motion_features=args.num_tokens_in_motion_features,
        seed=args.seed,
        loss_type=args.loss_type,
        report_to=args.report_to,
    )
