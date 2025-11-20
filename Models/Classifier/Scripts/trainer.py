import pytorch_lightning as pl
import argparse
import wandb

from dataloader import SkeletonDataModule
from model import TCN

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

PATH_DATA           = "../../../Dataset/Data/Processed/data.csv"
PATH_LOGS           = "../Model/Logs"
PATH_CHECKPOINTS    = "../Model/Checkpoints"

PROJECT_NAME        = "SaberPredict"

BATCH_SIZE          = 32
MAX_EPOCHS          = 25
TUNED_LR            = 0.0009549925860214359

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model", type=str, default="gnn-baseline")
    args = parser.parse_args()

    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=args.model,
        save_dir=PATH_LOGS,
        log_model=True,
    )

    data = SkeletonDataModule(
        csv_path=PATH_DATA,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    data.setup()

    model = TCN(
        num_classes=data.num_classes,
        label_dict=data.label_dict,
        lr=TUNED_LR
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=PATH_CHECKPOINTS,
        filename="gnn-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback]
    )

    ckpt_path = PATH_CHECKPOINTS + "/large.ckpt" if args.resume else None
    trainer.fit(model, data, ckpt_path=ckpt_path)
    trainer.test(model, data)

    wandb.finish()

if __name__ == "__main__":
    main()