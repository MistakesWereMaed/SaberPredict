import pytorch_lightning as pl
import argparse
import wandb

from Pipeline.Models.dataloader import SkeletonDataModule
from Pipeline.Models.classifier import TCN

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner import Tuner

PATH_TRAIN          = "Dataset/Data/Processed/train.csv"
PATH_TEST           = "Dataset/Data/Processed/test.csv"

PATH_LOGS           = "Pipeline/Logs"
PATH_CHECKPOINTS    = "Pipeline/Models/Checkpoints"

PROJECT_NAME        = "SaberPredict"

BATCH_SIZE          = 32
MAX_EPOCHS          = 75
TUNED_LR            = 1.659586907437561e-05

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", default=False)
    args = parser.parse_args()

    name = "TCN"
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=name,
        save_dir=PATH_LOGS
    )

    data = SkeletonDataModule(
        train_csv=PATH_TRAIN,
        test_csv=PATH_TEST,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    data.setup()

    model = TCN(
        num_classes=data.num_classes,
        label_dict=data.label_dict,
        max_epochs=MAX_EPOCHS,
        lr=TUNED_LR
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=PATH_CHECKPOINTS,
        filename=f"{name}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=model.hparams.max_epochs,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback]
    )

    if args.tune:
        tuner = Tuner(trainer)

        lr_finder = tuner.lr_find(model, datamodule=data, min_lr=1e-5, max_lr=1e-3)
        model.hparams.lr = lr_finder.suggestion()

    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")

    wandb.finish()

if __name__ == "__main__":
    main()