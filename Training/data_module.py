import pytorch_lightning as pl

from torch.utils.data import DataLoader
from Training.dataset import SkeletonDataset
from Training.sampler import ClassWeightedSampler

class SkeletonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        test_df,
        batch_size=32,
        num_workers=4,
        use_weighted_sampling=False
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_weighted_sampling = use_weighted_sampling

    def setup(self, stage=None):
        # Build train first to define label space
        self.train_set = SkeletonDataset(self.train_df)
        self.label_dict = self.train_set.id_to_label
        self.class_weights = self.train_set.class_weights
        self.num_classes = len(self.label_dict)

        # Share label mapping across splits
        self.test_set = SkeletonDataset(self.test_df)

    def train_dataloader(self):
        if self.use_weighted_sampling:
            labels = self.train_set.labels
            weights = self.train_set.sampler_weights
            sampler = ClassWeightedSampler(labels=labels, weights=weights).get_sampler()

            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
            )
        else:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
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
