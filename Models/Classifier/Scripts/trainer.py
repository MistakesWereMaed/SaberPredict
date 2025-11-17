import pytorch_lightning as pl
import wandb

from skeleton_dataset import SkeletonDataModule
from skeleton_gnn_lightning import GNN

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

PATH_DATA           = "../../../Dataset/Data/Processed/data.csv"
PATH_LOGS           = "../Model/Logs"
PATH_CHECKPOINTS    = "../Model/Checkpoints"

PROJECT_NAME        = "skeleton-action-recognition"
MODEL_NAME          = "gnn-baseline"

NUM_ACTIONS         = 14

def main():
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=MODEL_NAME,
        log_model=True,
        save_dir=PATH_LOGS,
    )

    data = SkeletonDataModule(
        csv_path=PATH_DATA,
        batch_size=32,
        num_workers=4,
    )

    model = GNN(
        num_classes=NUM_ACTIONS,
        lr=1e-3
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=PATH_CHECKPOINTS,
        filename="gnn-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, lr_callback],
    )

    trainer.fit(model, data)
    trainer.test(model, data)

    wandb.finish()

if __name__ == "__main__":
    main()