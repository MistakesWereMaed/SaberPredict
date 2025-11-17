import pytorch_lightning as pl
import wandb

from dataloader import SkeletonDataModule
from model import GNN

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

PATH_DATA           = "../../../Dataset/Data/Processed/data.csv"
PATH_LOGS           = "../Model/Logs"
PATH_CHECKPOINTS    = "../Model/Checkpoints"

PROJECT_NAME        = "SaberPredict"
MODEL_NAME          = "gnn-baseline"

BATCH_SIZE          = 32
MAX_EPOCHS          = 50

def main():
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=MODEL_NAME,
        save_dir=PATH_LOGS,
        log_model=True,
    )

    data = SkeletonDataModule(
        csv_path=PATH_DATA,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    data.setup()

    model = GNN(
        num_classes=data.num_classes,
        label_dict=data.label_dict,
        lr=1e-3
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=PATH_CHECKPOINTS,
        filename="{MODEL_NAME}-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, data)
    trainer.test(model, data)

    wandb.finish()

if __name__ == "__main__":
    main()