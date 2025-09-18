import os
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TFv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchcodec.decoders import VideoDecoder

class FencingDataset(Dataset):
    def __init__(self, path_clips, num_frames=30, resize=(112, 112), transform=None, device="cpu"):
        self.path_clips = path_clips
        self.metadata = pd.read_csv(os.path.join(path_clips, "metadata.csv"))
        self.label_map = {"Left": 0, "Right": 1}

        self.num_frames = num_frames
        self.resize = resize
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        clip_id = row["new_clip_id"]
        label = self.label_map[row["label"]]

        video_path = os.path.join(self.path_clips, f"{clip_id}.mp4")
        # Initialize torchcodec decoder
        decoder = VideoDecoder(video_path, device=self.device, dimension_order="NCHW", seek_mode="exact")

        # video metadata: number of frames
        total_frames = decoder.metadata.num_frames

        # get last num_frames frames; pad if too few
        if total_frames >= self.num_frames:
            # slice from end
            frames = decoder[-self.num_frames:]  # returns Tensor of shape [num_frames, C, H, W]
        else:
            # get all frames, then pad at front
            frames = decoder[:]  # all frames
            # frames is [total_frames, C, H, W]
            # pad with first frame (or black) at front
            pad_amt = self.num_frames - total_frames
            first_frame = frames[0].unsqueeze(0).repeat(pad_amt, 1, 1, 1)
            frames = torch.cat([first_frame, frames], dim=0)

        # Now `frames` is [num_frames, C, H, W]

        # Resize each frame
        # Using torchvision transforms v2 (functional) on each frame
        # If you want to do it more efficiently, can vectorize or batch
        # Here example per-frame loop
        resized = []
        for f in frames:
            # f is uint8, shape (C, H, W)
            # Resize: TFv2.resize expects (C, H, W) or maybe (H, W, C), but since dimension_order="NCHW", we use (C, H, W)
            f_resized = TFv2.resize(f, self.resize)  # this resizes spatially
            resized.append(f_resized)
        video_tensor = torch.stack(resized, dim=1)  # shape [C, T, H, W] with T = num_frames

        # Normalize to float [0,1]
        video_tensor = video_tensor.float() / 255.0

        if self.transform:
            video_tensor = self.transform(video_tensor)

        return video_tensor, label

def load_data(path_clips, batch_size=8, val_split=0.2, shuffle=True, num_workers=2):
    dataset = FencingDataset(path_clips)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader