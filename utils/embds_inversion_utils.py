import gc
import os
import sys
import types  # For overriding bound methods of given instances
from typing import Dict, Optional, Tuple, Union

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.transformers.transformer_temporal import (
    TransformerSpatioTemporalModel,
    TransformerTemporalModelOutput,
)
from diffusers.models.unets.unet_3d_blocks import (
    CrossAttnDownBlockSpatioTemporal,
    CrossAttnUpBlockSpatioTemporal,
    UNetMidBlockSpatioTemporal,
)
from diffusers.models.unets.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionModel,
    UNetSpatioTemporalConditionOutput,
)
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import *  # for overriding easier
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _append_dims,  # for overriding easier
)
from diffusers.utils import load_image

from utils.video_utils import pt_to_pil


class ImageEmbeddingWrapper(torch.nn.Module):
    SAVE_NAME = "img_embds_wrapper.pt"

    def __init__(self, tensor):
        super(ImageEmbeddingWrapper, self).__init__()
        self.tensor = torch.nn.Parameter(tensor)

    def forward(self, *inputs):
        # Dummy forward pass that just returns the tensor
        return self.tensor

    def get_parameters(self):
        # Provide access to the tensor as if it were a model parameter
        return [self.tensor]

    def save_pretrained(self, save_path):
        if os.path.isdir(save_path):
            file_path = os.path.join(
                save_path, self.SAVE_NAME
            )  # Save inside the directory
        else:
            file_path = save_path  # Save directly to the specified file path
            save_dir = os.path.dirname(file_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)  # Create directories if they don’t exist

        torch.save(self.tensor, file_path)
        print(f"[ImageEmbeddingWrapper] Tensor saved to {file_path}")

    def load_tensor(self, load_path):
        if os.path.isfile(load_path):
            file_path = load_path  # Use the provided file
        else:
            file_path = os.path.join(
                load_path, self.SAVE_NAME
            )  # Use SAVE_NAME in the given directory

        loaded_tensor = torch.load(file_path, weights_only=True)
        self.tensor = torch.nn.Parameter(loaded_tensor.to(self.tensor.device))
        print(f"[ImageEmbeddingWrapper] Tensor loaded from {file_path}")
        return self


def initialize_image_embedding(
    video, pretrained_model_name_or_path, N, do_classifier_free_guidance=False
):
    """
    @params N: N from the Reenact Anything paper, number of tokens for the cross attention
    """
    # we get torch.Size([2, 1, 1024]), of 2-size batch of conditional and unconditional
    # Or torch.Size([1, 1, 1024]) when not doing classifier free guidance

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path
    )
    pipeline._guidance_scale = 3  # done here to allow further inference: https://github.com/huggingface/diffusers/blob/v0.30.0/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L492

    # The motion embedding using clip is done here - https://github.com/huggingface/diffusers/blob/v0.30.0/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L495
    # Get the origin encoder output, and initialize using this information to ensure reasonable convergence
    with torch.no_grad():
        encoding_all_frames = pipeline._encode_image(
            pt_to_pil(video),
            device=None,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        ).to(video.device)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    encoding_all_frames_multiple_tokens = encoding_all_frames.repeat(
        1, N, 1
    )  # [num_frames, 1, channels] -> [num_frames, N, channels]
    encoding_with_temporal_enc_addition = torch.cat(
        [
            encoding_all_frames_multiple_tokens,
            encoding_all_frames_multiple_tokens[0].unsqueeze(0),
        ],
        dim=0,
    )  # [num_frames, N, channels] -> [num_frames + 1, N, channels]
    encoding_single_batch = encoding_with_temporal_enc_addition.unsqueeze(0)

    return (
        encoding_single_batch
        + encoding_single_batch.std() * torch.randn_like(encoding_single_batch) * 0.1
    )  # Added a little noise, since each N tokens are completely the same so to encourage diffrence between the weights


def pipeline_decode_latents(pipeline, latents, num_frames, decode_chunk_size=None):
    """
    For my usage to decode the latents during training before we start validation, for debugging
    """
    decode_chunk_size = (
        decode_chunk_size if decode_chunk_size is not None else num_frames
    )

    frames = pipeline.decode_latents(latents, num_frames, decode_chunk_size)
    frames = pipeline.video_processor.postprocess_video(video=frames, output_type="pil")
    return frames


# Overrides of functions
def override_pipeline_call(
    self_override,
    image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
    height: int = 576,
    width: int = 1024,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 25,
    sigmas: Optional[List[float]] = None,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    fps: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: Optional[int] = None,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    return_dict: bool = True,
    motion_features=None,  # OVERRIDEN HERE
):
    r"""
    The call function to the pipeline for generation.

    Args:
        image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.Tensor`):
            Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0,
            1]`.
        height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The width in pixels of the generated image.
        num_frames (`int`, *optional*):
            The number of video frames to generate. Defaults to `self.unet.config.num_frames` (14 for
            `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
        num_inference_steps (`int`, *optional*, defaults to 25):
            The number of denoising steps. More denoising steps usually lead to a higher quality video at the
            expense of slower inference. This parameter is modulated by `strength`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        min_guidance_scale (`float`, *optional*, defaults to 1.0):
            The minimum guidance scale. Used for the classifier free guidance with first frame.
        max_guidance_scale (`float`, *optional*, defaults to 3.0):
            The maximum guidance scale. Used for the classifier free guidance with last frame.
        fps (`int`, *optional*, defaults to 7):
            Frames per second. The rate at which the generated images shall be exported to a video after
            generation. Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
        motion_bucket_id (`int`, *optional*, defaults to 127):
            Used for conditioning the amount of motion for the generation. The higher the number the more motion
            will be in the video.
        noise_aug_strength (`float`, *optional*, defaults to 0.02):
            The amount of noise added to the init image, the higher it is the less the video will look like the
            init image. Increase it for more motion.
        decode_chunk_size (`int`, *optional*):
            The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the
            expense of more memory usage. By default, the decoder decodes all frames at once for maximal quality.
            For lower memory usage, reduce `decode_chunk_size`.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of videos to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `pil`, `np` or `pt`.
        callback_on_step_end (`Callable`, *optional*):
            A function that is called at the end of each denoising step during inference. The function is called
            with the following arguments:
                `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
            `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is
            returned, otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.Tensor`) is
            returned.
    """
    self = self_override

    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
    decode_chunk_size = (
        decode_chunk_size if decode_chunk_size is not None else num_frames
    )

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(image, height, width)

    # 2. Define call parameters
    if isinstance(image, PIL.Image.Image):
        batch_size = 1
    elif isinstance(image, list):
        batch_size = len(image)
    else:
        batch_size = image.shape[0]
    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    self._guidance_scale = max_guidance_scale

    # 3. Encode input image
    # OVERRIDEN HERE
    image_embeddings = (
        motion_features
        if motion_features is not None
        else self._encode_image(
            image, device, num_videos_per_prompt, self.do_classifier_free_guidance
        )
    )

    # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
    # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
    fps = fps - 1

    # 4. Encode input image using VAE
    image = self.video_processor.preprocess(image, height=height, width=width).to(
        device
    )
    noise = randn_tensor(
        image.shape, generator=generator, device=device, dtype=image.dtype
    )
    image = image + noise_aug_strength * noise

    needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
    if needs_upcasting:
        self.vae.to(dtype=torch.float32)

    image_latents = self._encode_vae_image(
        image,
        device=device,
        num_videos_per_prompt=num_videos_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
    )
    image_latents = image_latents.to(image_embeddings.dtype)

    # cast back to fp16 if needed
    if needs_upcasting:
        self.vae.to(dtype=torch.float16)

    # Repeat the image latents for each frame so we can concatenate them with the noise
    # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
    image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

    # 5. Get Added Time IDs
    added_time_ids = self._get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        image_embeddings.dtype,
        batch_size,
        num_videos_per_prompt,
        self.do_classifier_free_guidance,
    )
    added_time_ids = added_time_ids.to(device)

    # 6. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, None, sigmas
    )

    # 7. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_frames,
        num_channels_latents,
        height,
        width,
        image_embeddings.dtype,
        device,
        generator,
        latents,
    )

    # 8. Prepare guidance scale
    guidance_scale = torch.linspace(
        min_guidance_scale, max_guidance_scale, num_frames
    ).unsqueeze(0)
    guidance_scale = guidance_scale.to(device, latents.dtype)
    guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
    guidance_scale = _append_dims(guidance_scale, latents.ndim)

    self._guidance_scale = guidance_scale

    # 9. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimension
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if not output_type == "latent":
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = self.video_processor.postprocess_video(
            video=frames, output_type=output_type
        )
    else:
        frames = latents

    self.maybe_free_model_hooks()

    if not return_dict:
        return frames

    return StableVideoDiffusionPipelineOutput(frames=frames)


def forward_UNetSpatioTemporalConditionModel(
    self,
    sample: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    added_time_ids: torch.Tensor,
    return_dict: bool = True,
) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
    r"""
    The [`UNetSpatioTemporalConditionModel`] forward method.

    Args:
        sample (`torch.Tensor`):
            The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
        timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.Tensor`):
            The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
        added_time_ids: (`torch.Tensor`):
            The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
            embeddings and added to the time embeddings.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
            of a plain tuple.
    Returns:
        [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
            If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
            returned, otherwise a `tuple` is returned where the first element is the sample tensor.
    """
    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    batch_size, num_frames = sample.shape[:2]
    timesteps = timesteps.expand(batch_size)

    t_emb = self.time_proj(timesteps)

    # `Timesteps` does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=sample.dtype)

    emb = self.time_embedding(t_emb)

    time_embeds = self.add_time_proj(added_time_ids.flatten())
    time_embeds = time_embeds.reshape((batch_size, -1))
    time_embeds = time_embeds.to(emb.dtype)
    aug_emb = self.add_embedding(time_embeds)
    emb = emb + aug_emb

    # Flatten the batch and frames dimensions
    # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
    sample = sample.flatten(0, 1)
    # Repeat the embeddings num_video_frames times
    # emb: [batch, channels] -> [batch * frames, channels]
    emb = emb.repeat_interleave(num_frames, dim=0)

    # Changed here from this:
    # # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
    # encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
    # To this:
    # encoder_hidden_states: [batch, frames+1, N, channels] -> [batch * (frames+1), N, channels]
    _, _, N, chn = encoder_hidden_states.shape
    encoder_hidden_states = encoder_hidden_states.reshape(-1, N, chn)
    # End Change

    # 2. pre-process
    sample = self.conv_in(sample)

    image_only_indicator = torch.zeros(
        batch_size, num_frames, dtype=sample.dtype, device=sample.device
    )

    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if (
            hasattr(downsample_block, "has_cross_attention")
            and downsample_block.has_cross_attention
        ):
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )
        else:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                image_only_indicator=image_only_indicator,
            )

        down_block_res_samples += res_samples

    # 4. mid
    sample = self.mid_block(
        hidden_states=sample,
        temb=emb,
        encoder_hidden_states=encoder_hidden_states,
        image_only_indicator=image_only_indicator,
    )

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if (
            hasattr(upsample_block, "has_cross_attention")
            and upsample_block.has_cross_attention
        ):
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                image_only_indicator=image_only_indicator,
            )

    # 6. post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    # 7. Reshape back to original shape
    sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

    if not return_dict:
        return (sample,)

    return UNetSpatioTemporalConditionOutput(sample=sample)


def forward_downUpMidBlock_TransformerSpatioTemporalModel(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    image_only_indicator: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
            Input hidden_states.
        num_frames (`int`):
            The number of frames to be processed per batch. This is used to reshape the hidden states.
        encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
            Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
            self-attention.
        image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
            A tensor indicating whether the input contains only images. 1 indicates that the input contains only
            images, 0 indicates that the input contains video frames.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`]
            instead of a plain tuple.

    Returns:
        [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
            If `return_dict` is True, an
            [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
    """
    # 1. Input
    batch_frames, _, height, width = hidden_states.shape
    num_frames = image_only_indicator.shape[-1]
    batch_size = batch_frames // num_frames

    # CHANGED HERE:
    # [batch, frames + 1, N, channels] -> [batch * (frames + 1), N, channels]

    # time_context = encoder_hidden_states
    # time_context_first_timestep = time_context[None, :].reshape(
    #     batch_size, num_frames, -1, time_context.shape[-1]
    # )[:, 0]
    # time_context = time_context_first_timestep[:, None].broadcast_to(
    #     batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
    # )
    # time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1])
    time_context = encoder_hidden_states  # [batch * (frames+1), N, channels]
    time_context_first_timestep = time_context[None, :].reshape(
        batch_size, num_frames + 1, -1, time_context.shape[-1]
    )[
        :, -1
    ]  # Take the f+1 embeddings we initialized for the temporal attention. [batch, 1, N, channels]
    time_context = time_context_first_timestep[:, None].broadcast_to(
        batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
    )  # [batch, height * width, N, channels]
    time_context = time_context.reshape(
        batch_size * height * width, -1, time_context.shape[-1]
    )  # [batch * height * width, N, channels]

    # And the spatial attention condition:
    encoder_hidden_states = encoder_hidden_states[None, :].reshape(
        batch_size, num_frames + 1, -1, encoder_hidden_states.shape[-1]
    )[
        :, :num_frames
    ]  # Take the f embeddings we initialized for the spatial attention. [batch, F, N, channels]
    encoder_hidden_states = encoder_hidden_states.reshape(
        batch_size * num_frames, -1, encoder_hidden_states.shape[-1]
    )

    # End Change

    residual = hidden_states

    hidden_states = self.norm(hidden_states)
    inner_dim = hidden_states.shape[1]
    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
        batch_frames, height * width, inner_dim
    )
    hidden_states = self.proj_in(hidden_states)

    num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
    num_frames_emb = num_frames_emb.repeat(batch_size, 1)
    num_frames_emb = num_frames_emb.reshape(-1)
    t_emb = self.time_proj(num_frames_emb)

    # `Timesteps` does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)

    emb = self.time_pos_embed(t_emb)
    emb = emb[:, None, :]

    # 2. Blocks
    for block, temporal_block in zip(
        self.transformer_blocks, self.temporal_transformer_blocks
    ):
        if self.training and self.gradient_checkpointing:
            hidden_states = torch.utils.checkpoint.checkpoint(
                block,
                hidden_states,
                None,
                encoder_hidden_states,
                None,
                use_reentrant=False,
            )
        else:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states_mix = hidden_states
        hidden_states_mix = hidden_states_mix + emb

        hidden_states_mix = temporal_block(
            hidden_states_mix,
            num_frames=num_frames,
            encoder_hidden_states=time_context,
        )
        hidden_states = self.time_mixer(
            x_spatial=hidden_states,
            x_temporal=hidden_states_mix,
            image_only_indicator=image_only_indicator,
        )

    # 3. Output
    hidden_states = self.proj_out(hidden_states)
    hidden_states = (
        hidden_states.reshape(batch_frames, height, width, inner_dim)
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    output = hidden_states + residual

    if not return_dict:
        return (output,)

    return TransformerTemporalModelOutput(sample=output)


def allow_motion_embedding(unet):
    unet.forward = types.MethodType(forward_UNetSpatioTemporalConditionModel, unet)

    for name, child in unet.named_children():
        # Downblocks, upblocks and midblock:
        if name in ["down_blocks", "up_blocks", "mid_block"]:
            modules_list = child if isinstance(child, torch.nn.ModuleList) else [child]
            for cur_module in modules_list:
                if (
                    isinstance(cur_module, CrossAttnDownBlockSpatioTemporal)
                    or isinstance(cur_module, CrossAttnUpBlockSpatioTemporal)
                    or isinstance(cur_module, UNetMidBlockSpatioTemporal)
                ):
                    # Replace with the custom block
                    for attn in cur_module.attentions:
                        assert isinstance(attn, TransformerSpatioTemporalModel), (
                            "This overriding assumes all of the attentions of"
                            " the blocks are of type TransformerSpatioTemporalModel. "
                            "If it isn't the case, should rewrite it."
                        )

                        attn.forward = types.MethodType(
                            forward_downUpMidBlock_TransformerSpatioTemporalModel, attn
                        )

    return unet
    return unet
