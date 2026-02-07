import pytorch_lightning as pl

from torch.utils.data import DataLoader

from Training.dataset import SkeletonSequenceDataset
from Training.sampler import BoundaryAwareSequenceSampler

class SkeletonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_frame_csv,
        train_sequence_csv,
        val_frame_csv,
        val_sequence_csv,
        batch_size=32,
        num_workers=2,
    ):
        super().__init__()
        self.train_frame_csv = train_frame_csv
        self.train_sequence_csv = train_sequence_csv
        self.val_frame_csv = val_frame_csv
        self.val_sequence_csv = val_sequence_csv

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_set = SkeletonSequenceDataset(
            self.train_frame_csv,
            self.train_sequence_csv
        )

        self.val_set = SkeletonSequenceDataset(
            self.val_frame_csv,
            self.val_sequence_csv
        )

        self.num_classes = len(self.train_set.label_to_id)
        self.label_dict = self.train_set.id_to_label

        self.train_sampler = BoundaryAwareSequenceSampler(
            sequence_df=self.train_set.seq_df,
            #label_map=self.train_set.label_map
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
