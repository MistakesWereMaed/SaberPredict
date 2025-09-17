import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io  # video reader

class FencingDataset(Dataset):
    def __init__(self, path_clips):
        self.path_clips = path_clips

        # Read CSV
        self.metadata = pd.read_csv(path_clips + "/metadata.csv")
        self.label_map = {"Left": 0, "Right": 1}  # convert labels to numbers

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        clip_id = row["new_clip_id"]
        label = self.label_map[row["label"]]

        video_path = self.path_clips, + f"{clip_id}.mp4"

        # Read video (T, H, W, C)
        video, _, _ = io.read_video(video_path, pts_unit="sec")

        # Convert to [C, T, H, W], normalize to [0,1]
        video = video.permute(3, 0, 1, 2).float() / 255.0

        if self.transform:
            video = self.transform(video)

        return video, label

def load_data(path_clips, batch_size=16):
    dataset = FencingDataset(path_clips)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    return dataloader