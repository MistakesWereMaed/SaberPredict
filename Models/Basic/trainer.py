import torch
import pytorch_lightning as pl
import wandb
import argparse

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from dataloader import FencingDataset
from model import FencerScorer

# ---------------------------
# Training script
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Init wandb
    wandb_logger = WandbLogger(project="fencing-scorer", log_model="all")

    # Dataset + loaders
    train_dataset = FencingDataset(csv_file="train.csv", clip_dir="clips_30f", transform=None)
    val_dataset = FencingDataset(csv_file="val.csv", clip_dir="clips_30f", transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model
    model = FencerScorer(lr=args.lr)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Finish wandb
    wandb.finish()


if __name__ == "__main__":
    main()   