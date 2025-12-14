import os
from typing import List, Union

import cv2
import imageio
import numpy as np
import PIL.Image
import PIL.ImageOps
import torch
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import Dataset
from torchvision import transforms

try:
    import wandb
except ImportError:
    wandb = None


def export_to_video(
    video_frames: Union[List[np.ndarray], torch.Tensor, List[PIL.Image.Image]],
    output_path: str,
    fps: int = 7,
) -> str:
    """
    Args:
        video_frames: Video frames of shape (frames, height, width, channels) e.g (25, 576, 768, 3)
        output_path: Path to save video (.mp4 or .gif)
    """
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.numpy()

    writer = imageio.get_writer(output_path, fps=fps)
    for frame in video_frames:
        writer.append_data(np.array(frame))
    writer.close()
    return output_path


def load_video(video_path: str = None) -> List[PIL.Image.Image]:
    reader = imageio.get_reader(video_path)
    return [PIL.Image.fromarray(f) for f in reader]


def pil_to_pt(images: List[PIL.Image.Image]) -> torch.Tensor:
    vid_frames_np = VaeImageProcessor.pil_to_numpy(images)
    return VaeImageProcessor.numpy_to_pt(vid_frames_np)


def pt_to_pil(images: torch.Tensor) -> List[PIL.Image.Image]:
    vid_frames_np = VaeImageProcessor.pt_to_numpy(images)
    return VaeImageProcessor.numpy_to_pil(vid_frames_np)


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [
        PIL.Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        for frame in frames
    ]

    pil_frames[0].save(
        output_gif_path.replace(".mp4", ".gif"),
        format="GIF",
        append_images=pil_frames[1:],
        save_all=True,
        duration=500,
        loop=0,
    )


def log_decoded_video(
    global_step,
    num_frames,
    val_img_idx,
    val_save_dir,
    video_frames,
    video_desc="val_img",
    report_to=None,
    out_file_name=None,
):
    if out_file_name is None:
        out_suff = f"{video_desc}_{val_img_idx}"
        filename = f"step_{global_step}_{out_suff}.gif"
    else:
        filename = out_file_name
    out_file = os.path.join(
        val_save_dir,
        filename,
    )

    for i in range(num_frames):
        img = video_frames[i]
        video_frames[i] = np.array(img)
    export_to_gif(video_frames, out_file, 8)

    if report_to == "wandb":
        downsampled_out_file = os.path.join(
            val_save_dir,
            "downsampled_" + filename,
        )
        # resize to save space in wandb
        orig_h, orig_w = video_frames[0].shape[-3:-1]
        downscale_by = 0.2
        new_size = (int(downscale_by * orig_w), int(downscale_by * orig_h))
        resized_video = np.array(
            [
                cv2.resize(frame, dsize=new_size, interpolation=cv2.INTER_CUBIC)
                for frame in video_frames
            ]
        )
        export_to_gif(resized_video, downsampled_out_file, 8)
        wandb.log(
            {out_suff: wandb.Video(downsampled_out_file, fps=8)}, step=global_step
        )
        os.remove(downsampled_out_file)


class MotionSingleVideoDataset(Dataset):
    def __init__(
        self,
        video_path,
        size=(25, 1024, 576),  # (n_frames, width, height)
        repeats=100,
        flip_p=0.5,
        set="train",
        device="cpu",
    ):
        self.video_path = video_path
        self.size = size
        self.n_frames, self.width, self.height = size
        self.flip_p = flip_p

        assert os.path.exists(self.video_path) and self.video_path.endswith(
            ".mp4"
        ), f"Video path {self.video_path} does not exist or is not an mp4 file"

        self.motion_vid_path = self.video_path

        if set == "train":
            self._length = repeats

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # Now initialize the video
        frames = load_video(self.motion_vid_path)
        resized_frames = [
            f.resize((self.width, self.height)) for f in frames
        ]  # resize(width, height)

        # Ensure frame count is n_frames, up\downsample if necessary
        if len(resized_frames) > self.n_frames:
            print(
                f"Warning: Video has {len(resized_frames)} frames. Taking the first {self.n_frames}"
            )
            resized_frames = resized_frames[: self.n_frames]
        elif len(resized_frames) < self.n_frames:
            print(
                f"Warning: Video has {len(resized_frames)} frames. Upsampling to {self.n_frames}"
            )
            resized_frames = resized_frames + [resized_frames[-1]] * (
                self.n_frames - len(resized_frames)
            )

        self.vid_frames_pil = resized_frames
        self.vid_frames = (
            pil_to_pt(self.vid_frames_pil).to(device) * 2.0 - 1.0
        )  # normalize to [-1, 1]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        sample = {}

        # TODO: Possible - Add augmentations here
        augmented_frames = self.vid_frames
        sample["frame_0"] = augmented_frames[0]
        sample["frames"] = augmented_frames

        return sample

    @staticmethod
    def save_batch(batch):
        export_to_video(pt_to_pil(batch["frames"]), "./debug_frames.mp4")

        pt_to_pil(batch["frame_0"].unsqueeze(0))[0].save(
            "./debug_frame_0.png",
        )


class SimpleImagesDataset(Dataset):
    def __init__(
        self,
        data_root,
        width=1024,
        height=576,
        max_images_n=None,
        device="cpu",
    ):
        self.data_root = data_root
        self._max_images_n = max_images_n
        self.width = width
        self.height = height

        if self.data_root is not None:
            self.img_paths = [
                os.path.join(self.data_root, file_path)
                for file_path in os.listdir(self.data_root)
                if file_path.endswith((".png", "jpg"))
            ]
        else:
            self.img_paths = []

        self.img_names = [
            os.path.splitext(os.path.basename(filename))[0]
            for filename in self.img_paths
        ]
        self.orig_images_pil = [PIL.Image.open(f) for f in self.img_paths]
        self.images_pil = [
            f.resize((self.width, self.height)) for f in self.orig_images_pil
        ]

        self.images = [  # pil_to_pt returns [1,chn,w,h] so squeeze to [3,w,h] (remove alpha channel)
            pil_to_pt(img).to(device)[:, :3, :, :] for img in self.images_pil
        ]

        self.imgs_n = (
            min(len(self.images), self._max_images_n)
            if self._max_images_n is not None
            else len(self.images)
        )
        self.images = self.images[: self.imgs_n]
        self.img_names = self.img_names[: self.imgs_n]
        self._length = self.imgs_n

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        return {"name": self.img_names[i], "frame": self.images[i]}

    @staticmethod
    def save_sample(sample: torch.Tensor, savename=None):
        if savename is None:
            savename = "./debug_frame_0.png"
        pt_to_pil(sample)[0].save(savename)