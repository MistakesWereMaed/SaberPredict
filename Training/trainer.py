import pytorch_lightning as pl
import argparse
import wandb

from Training.data_module import SkeletonDataModule
from Pipeline.classifier import TCN

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner import Tuner


PATH_TRAIN      = "Dataset/Data/Processed/cls_train.csv"
PATH_TEST       = "Dataset/Data/Processed/cls_test.csv"

PATH_SEQ_TRAIN  = "Dataset/Data/Processed/cls_seq_train.csv"
PATH_SEQ_TEST   = "Dataset/Data/Processed/cls_seq_test.csv"

PATH_LOGS        = "Training/Logs"
PATH_CHECKPOINTS = "Training/Checkpoints"

PROJECT_NAME = "SaberPredict"

BATCH_SIZE  = 32
MAX_EPOCHS  = 75
TUNED_LR    = 7.585775750291839e-05


def main():
    pl.seed_everything(42, workers=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", default=False)
    args = parser.parse_args()

    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name="TCN",
        save_dir=PATH_LOGS,
    )

    data = SkeletonDataModule(
        train_frame_csv=PATH_TRAIN,
        val_frame_csv=PATH_TEST,
        train_sequence_csv=PATH_SEQ_TRAIN,
        val_sequence_csv=PATH_SEQ_TEST,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    data.setup()

    model = TCN(
        num_classes=data.num_classes,
        label_dict=data.label_dict,
        max_epochs=MAX_EPOCHS,
        lr=TUNED_LR,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=PATH_CHECKPOINTS,
        filename="TCN-{epoch}-{val_acc:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=model.hparams.max_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
    )

    if args.tune:
        model.hparams.use_onecycle = False

        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model,
            datamodule=data,
            min_lr=1e-5,
            max_lr=1e-3,
        )
        
        print(lr_finder.suggestion())
        return

    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")

    wandb.finish()

if __name__ == "__main__":
    main()
