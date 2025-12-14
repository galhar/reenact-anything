import torch
from einops import rearrange

def log_grads(update_itself, optimizer, step, accelerator):
    # We only have a single tensor to optimize upon
    group = optimizer.param_groups[0]
    p = group["params"][0]

    # The update is done here https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py#L432
    # state = optimizer.state[p]
    # adam_update = (
    #     optimizer.lr
    #     * state["exp_avg"]
    #     / (state["exp_avg_sq"] + optimizer.epsilon)
    # )
    # adam_update_debug_dict = {
    #     "update_vs_data_median": torch.median(adam_update)
    #     / torch.median(p.data).log10(),
    #     "update_vs_data_mean": torch.mean(adam_update)
    #     / torch.mean(p.data).log10(),
    #     "update_vs_data_std": torch.std(adam_update)
    #     / torch.std(p.data).log10(),
    # }
    if p.grad is None:
        print(f"Warning: grad was none on step {step}, skipp logging gradients")
        return

    log_dict = {
        "step": step,
        # Weights themselves:
        "data_median": torch.median(p.data),
        "data_mean": torch.mean(p.data),
        "data_std": torch.std(p.data),
        # Grads:
        "grad_median": torch.median(p.grad),
        "grad_mean": torch.mean(p.grad),
        "grad_std": torch.std(p.grad),
        # Update of weights:
        "update_median": torch.median(update_itself),
        "update_mean": torch.mean(update_itself),
        "update_std": torch.std(update_itself),
        # Update vs Weights:
        "update_vs_data_median": (
            torch.median(update_itself) / torch.median(p.data)
        ).log10(),
        "update_vs_data_mean": (torch.mean(update_itself) / torch.mean(p.data)).log10(),
        "update_vs_data_std": (torch.std(update_itself) / torch.std(p.data)).log10(),
    }

    accelerator.log(log_dict, step=step)




# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners
    )
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (
        torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)
        - window_size // 2
    ).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def gpu_stats(logger):
    logger.info(
        f"torch.cuda.memory_allocated: {(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)} GB"
    )
    logger.info(
        f"torch.cuda.memory_reserved: {(torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)} GB"
    )
    logger.info(
        f"torch.cuda.max_memory_reserved: {(torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)} GB"
    )


def encode_image(pixel_values, feature_extractor, image_encoder):
    pixel_values_cpy = pixel_values.clone()
    # pixel: [-1, 1]
    pixel_values_cpy = _resize_with_antialiasing(pixel_values_cpy.float(), (224, 224))
    # We unnormalize it after resizing.
    pixel_values_cpy = (pixel_values_cpy + 1.0) / 2.0

    # Normalize the image with for CLIP input
    pixel_values_cpy = feature_extractor(
        images=pixel_values_cpy,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values

    pixel_values_cpy = pixel_values_cpy.to(device=pixel_values.device, dtype=pixel_values.dtype)
    image_embeddings = image_encoder(pixel_values_cpy).image_embeds
    return image_embeddings

def _get_add_time_ids(
    unet,
    fps,
    motion_bucket_id,
    noise_aug_strength,
    dtype,
    batch_size,
):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids