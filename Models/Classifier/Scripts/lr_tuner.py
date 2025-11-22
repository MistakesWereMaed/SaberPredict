import pytorch_lightning as pl
import argparse

from dataloader import SkeletonDataModule
from Models import TCN, GNN, MLP

from pytorch_lightning.tuner import Tuner

PATH_DATA           = "../../../Dataset/Data/Processed/data.csv"
PATH_LOGS           = "../Models/Logs"

BATCH_SIZE          = 32
MAX_EPOCHS          = 25

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TCN")
    args = parser.parse_args()

    model_name = args.model

    match(model_name):
        case "TCN":
            model_class = TCN.model
        case "GNN":
            model_class = GNN.model
        case "MLP":
            model_class = MLP.model

    data = SkeletonDataModule(
        csv_path=PATH_DATA,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    data.setup()

    model = model_class(
        num_classes=data.num_classes,
        label_dict=data.label_dict,
        lr=5e-4
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=False
    )

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=data, min_lr=1e-4, max_lr=1e-2)

    new_lr = lr_finder.suggestion()
    print("Suggested LR:", new_lr)

if __name__ == "__main__":
    main()