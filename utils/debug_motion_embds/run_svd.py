# NOTE: This file was used for debugging during implementation.
# It's a bit of a mess, but I'm keeping it here in case anyone wants to debug things.

import torch
from diffusers import DiffusionPipeline

from utils.embds_inversion_utils import (
    ImageEmbeddingWrapper,
    allow_motion_embedding,
    override_pipeline_call,
)
from utils.video_utils import MotionSingleVideoDataset

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt"
)

allow_motion_embedding(pipeline.unet)

num_frames, width, height = 14, 1024, 576

dataset = MotionSingleVideoDataset(
    "/reference_motion_video/realistic_horse_jumping/realistic_horse_jumping.mp4",
    size=(num_frames, width, height),
)
img_embds_wrap = ImageEmbeddingWrapper(torch.Tensor([1]))
img_embds_wrap.load_tensor(
    "/outputs/motion_inversion_gym/checkpoint-2900/image_embedding_wrapper"
)  # Just to take anything, doesnt matter what is inside here


motion_embds = img_embds_wrap().to("cpu")  # (1,1,1024) (batch, 1 tokens n, channels)
frame_0 = dataset[0]["frame_0"].to("cpu").unsqueeze(0)  # [batch=1, 3, height, width]

N = 5
motion_embds_expanded_tokens = motion_embds.repeat(
    1, N, 1
)  # (1, 1, 1024) -> (1, N, 1024) (batch, N, chn)
motion_embds_expanded_frames = motion_embds_expanded_tokens.unsqueeze(0).repeat(
    1, num_frames + 1, 1, 1
)  # (1, N, 1024) -> (1, F+1, N, 1024)

# For easier later debugging, define the tokens here:
base_embd = torch.ones(motion_embds.shape)  # [1, 1, 1024]
inflate_N = torch.cat(
    [base_embd * (i + 1) for i in range(N)], dim=1
)  # [1, N, 1024], 1,2,3,4,5 for each of the tokens
debug_motino_embd = torch.cat(
    [inflate_N.unsqueeze(0) * 10 * (i + 1) for i in range(num_frames + 1)], dim=1
)  # [1, 15, 5, 1024] (batch, frames+1, N, chn)


# Defined such debug motion features to make it easy for us to verify the correct embedding got in to the correct place later in the pipeline
# The debugging motion features:
"""
[[       
         # Frame 1 tokens:
         [ 10.], token 1 ([10,10,10,...], (1,1024))
         [ 20.], token 2  ([20,20,20,...], (1,1024))
         [ 30.], token 3
         [ 40.], token 4
         [ 50.]], token 5 ([50,50,50,...], (1,1024))

         # Frame 2 tokens:
        [[ 20.],
         [ 40.],
         [ 60.],
         [ 80.],
         [100.]],

         # Frame 3 tokens:
        [[ 30.],
         [ 60.],
         [ 90.],
         [120.],
         [150.]],

         # Frame 4 tokens:
        [[ 40.],
         [ 80.],
         [120.],
         [160.],
         [200.]],

        [[ 50.],
         [100.],
         [150.],
         [200.],
         [250.]],

        [[ 60.],
         [120.],
         [180.],
         [240.],
         [300.]],

        [[ 70.],
         [140.],
         [210.],
         [280.],
         [350.]],

        [[ 80.],
         [160.],
         [240.],
         [320.],
         [400.]],

        [[ 90.],
         [180.],
         [270.],
         [360.],
         [450.]],

        [[100.],
         [200.],
         [300.],
         [400.],
         [500.]],

        [[110.],
         [220.],
         [330.],
         [440.],
         [550.]],

        [[120.],
         [240.],
         [360.],
         [480.],
         [600.]],

        [[130.],
         [260.],
         [390.],
         [520.],
         [650.]],

        # Frame 14 tokens:
        [[140.],
         [280.],
         [420.],
         [560.],
         [700.]],

         # Temporal tokens, to be spatially broadcasted:
        [[150.],
         [300.],
         [450.],
         [600.],
         [750.]
"""


video = override_pipeline_call(
    pipeline,
    frame_0,
    height=height,
    width=width,
    num_frames=num_frames,
    decode_chunk_size=8,
    motion_bucket_id=127,
    fps=7,
    noise_aug_strength=0.02,
    motion_features=debug_motino_embd,#motion_embds_expanded_frames,
    max_guidance_scale=1,  # Dont do classifeir free guidance for now
    # generator=generator,
).frames[0]
