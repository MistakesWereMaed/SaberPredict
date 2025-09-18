import pytorch_lightning as pl
import wandb
import argparse

from pytorch_lightning.loggers import WandbLogger

from dataloader import load_data
from model import FencerScorer

PATH_CLIPS = "../../Dataset/clips_last30"

# ---------------------------
# Training script
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()

    # Init wandb
    wandb_logger = WandbLogger(name="Basic", project="SaberPredict", log_model="all")

    # Data
    train, val = load_data(PATH_CLIPS)

    # Model
    model = FencerScorer(lr=args.lr)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_checkpointing=False
    )

    # Train
    trainer.fit(model, train, val)

    # Finish wandb
    wandb.finish()

if __name__ == "__main__":
    main()   