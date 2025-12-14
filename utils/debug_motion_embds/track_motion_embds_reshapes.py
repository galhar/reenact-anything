import torch



# ======================= Create the embeddings: ===========================
N = 5
num_frames = 14

# For easier later debugging, define the tokens here:
base_embd = torch.ones((1, 1, 1024))#[1, 1, 1024]
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












# ====================== Now follow the transformations it goes: ==========================

encoder_hidden_states = debug_motino_embd.clone()
# forward_UNetSpatioTemporalConditionModel:
# Original:
# encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
# encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
# Rewritten:

# encoder_hidden_states: [batch, frames+1, N, channels] -> [batch * (frames+1), N, channels]
_, _, N, chn = encoder_hidden_states.shape
encoder_hidden_states = encoder_hidden_states.reshape(-1, N, chn)




# Downblock:
# https://github.com/huggingface/diffusers/blob/b52684c3edfa4aa33d72add90385c7b76c968b24/src/diffusers/models/unets/unet_3d_blocks.py#L1306
# Simply inserted into "TransformerSpatioTemporalModel"

# Upblock:
# https://github.com/huggingface/diffusers/blob/b52684c3edfa4aa33d72add90385c7b76c968b24/src/diffusers/models/unets/unet_3d_blocks.py#L1517
# Simply inserted into "TransformerSpatioTemporalModel"

# Midblock:
# https://github.com/huggingface/diffusers/blob/b52684c3edfa4aa33d72add90385c7b76c968b24/src/diffusers/models/unets/unet_3d_blocks.py#L1109
# Simply inserted into "TransformerSpatioTemporalModel"





# forward_downUpMidBlock_TransformerSpatioTemporalModel:

# Time context is what goes into the temporal attention.
# Original:
#time_context = encoder_hidden_states
# time_context_first_timestep = time_context[None, :].reshape(
#     batch_size, num_frames, -1, time_context.shape[-1]
# )[:, 0]
# time_context = time_context_first_timestep[:, None].broadcast_to(
#     batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
# )
# time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1])
#rewritten:
batch_size = 1 # Just adding, exists in the overriden function
height = 100
width = 100

time_context = encoder_hidden_states #[batch * (frames+1), N, channels]
time_context_first_timestep = time_context[None, :].reshape(
    batch_size, num_frames + 1, -1, time_context.shape[-1]
)[:, -1] # Take the f+1 embeddings we initialized for the temporal attention. [batch, 1, N, channels]
time_context = time_context_first_timestep[:, None].broadcast_to(
    batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
) #[batch, height * width, N, channels]
time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1]) #[batch * height * width, N, channels]

# And the spatial attention condition:
encoder_hidden_states = encoder_hidden_states[None, :].reshape(
        batch_size, num_frames + 1, -1, encoder_hidden_states.shape[-1]
    )[:, :num_frames]  # Take the f embeddings we initialized for the spatial attention. [batch, F, N, channels]
encoder_hidden_states = encoder_hidden_states.reshape(batch_size * num_frames, -1,
                                                      encoder_hidden_states.shape[-1])


# Verified V! The embedding looks good, it contains the numbers I would expect it to from the debugging_embedding.




# Basic attention blocks, spatial and temporal, just feed it forward and the dimensions should be ok:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py#L466
# Spatial: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py#L506
# Temporal: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py#L701







