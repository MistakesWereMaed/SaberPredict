import pytorch_lightning as pl

from torch.utils.data import DataLoader
from Training.dataset import SkeletonDataset

class SkeletonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv,
        test_csv,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Build train first to define label space
        self.train_set = SkeletonDataset(self.train_csv)
        self.label_dict = self.train_set.id_to_label
        self.num_classes = len(self.label_dict)

        # Share label mapping across splits
        self.test_set = SkeletonDataset(self.test_csv)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
