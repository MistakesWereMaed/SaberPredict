import pytorch_lightning as pl
from dataloader import SkeletonDataModule
from model import GNN

from pytorch_lightning.tuner.tuning import Tuner

PATH_DATA           = "../../../Dataset/Data/Processed/data.csv"

BATCH_SIZE          = 32
MAX_EPOCHS          = 100

def main():
    data = SkeletonDataModule(
        csv_path=PATH_DATA,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    data.setup()

    model = GNN(
        num_classes=data.num_classes,
        label_dict=data.label_dict,
        lr=5e-5
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        logger=False,
        callbacks=None,
        gradient_clip_val=1.0
    )

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=data)

    new_lr = lr_finder.suggestion()
    print("Suggested LR:", new_lr)

if __name__ == "__main__":
    main()