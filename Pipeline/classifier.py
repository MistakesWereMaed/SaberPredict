import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from typing import List, Optional
from torchmetrics import Accuracy, ConfusionMatrix

# ----------------------------
# Normalization
# ----------------------------

def normalize_input(x: torch.Tensor) -> torch.Tensor:
    """
    Per-sample, per-frame normalization.
    Center joints and scale by max pairwise joint distance.
    """
    B, T, V, C = x.shape

    center = x.mean(dim=2, keepdim=True)  # (B,T,1,C)
    x = x - center

    x_flat = x.reshape(B * T, V, C)
    dists = torch.cdist(x_flat, x_flat, p=2)
    max_d = dists.view(B, T, -1).max(dim=2)[0].view(B, T, 1, 1)

    return x / (max_d + 1e-6)


# ----------------------------
# TCN Building Block
# ----------------------------

class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))

        res = x if self.residual is None else self.residual(x)
        return self.relu(out + res)


# ----------------------------
# TCN Classifier
# ----------------------------

class TCN(pl.LightningModule):
    """
    Sequence-level action classifier with optional frame-level extension.
    """

    def __init__(
        self,
        label_dict: dict,
        num_classes: int,
        num_joints: int = 17,
        coord_dim: int = 2,
        tcn_channels: List[int] = [128, 256, 512],
        kernel_size: int = 3,
        dropout: float = 0.1,
        embed_dim: int = 768,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        use_onecycle: bool = True,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes

        # Input channels = joints * coords
        in_channels = num_joints * coord_dim

        layers = []
        prev = in_channels
        for i, ch in enumerate(tcn_channels):
            layers.append(
                TemporalBlock(
                    prev, ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
            prev = ch

        self.tcn = nn.Sequential(*layers)

        # Frame-wise projection (kept for future per-frame supervision)
        self.frame_proj = nn.Linear(prev, embed_dim)

        # Sequence-level classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    # ----------------------------
    # Forward
    # ----------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (B, T, V, C)
            mask: optional (B, T) boolean tensor for valid frames
        Returns:
            logits: (B, T, num_classes)
            frame_embeddings: (B, T, embed_dim)
        """
        B, T, V, C = x.shape

        x = normalize_input(x)

        # (B, T, V, C) → (B, V*C, T)
        x = x.permute(0, 2, 3, 1).reshape(B, V * C, T)

        feats = self.tcn(x)                  # (B, C_out, T)
        feats = feats.permute(0, 2, 1)       # (B, T, C_out)

        frame_emb = self.frame_proj(feats)   # (B, T, D)

        logits = self.classifier(frame_emb)  # (B, T, num_classes)

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)

        return logits, frame_emb


    # ----------------------------
    # Steps
    # ----------------------------

    def _shared_step(self, batch, stage: str):
        x = batch["keypoints"]           # (B,T,V,C)
        y = batch["labels"]              # (B,)
        mask = batch.get("mask", None)   # optional

        logits, _ = self(x, mask)
        B, T, C = logits.shape

        logits = logits.reshape(B * T, C)
        y = y.reshape(B * T)

        loss = F.cross_entropy(
            logits, y,
            label_smoothing=self.hparams.label_smoothing
        )

        preds = logits.argmax(dim=1)

        acc_map = {
            "train": self.train_acc,
            "val": self.val_acc,
            "test": self.test_acc,
        }

        acc = acc_map[stage](preds, y)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

        if stage == "test":
            self.confmat(preds, y)

        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    def test_step(self, batch, _):
        return self._shared_step(batch, "test")

    # ----------------------------
    # Optimizer
    # ----------------------------

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if not self.hparams.use_onecycle:
            return opt

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # ----------------------------
    # Test Visualization
    # ----------------------------

    def on_test_epoch_end(self):
        self._draw_plots()

    def _draw_plots(self):
        # Compute confusion matrix
        confmat = self.confmat.compute().detach().cpu()

        # 1. Per-class accuracy plot
        per_class_acc = confmat.diag() / confmat.sum(axis=1).clip(min=1)

        num_classes = confmat.shape[0]
        class_names = [self.hparams.label_dict[i] for i in range(num_classes)]

        fig_acc, ax_acc = plt.subplots(figsize=(12, 8))
        bars = ax_acc.bar(range(num_classes), per_class_acc)

        ax_acc.set_xticks(range(num_classes))
        ax_acc.set_xticklabels(class_names, rotation=45, ha="right")
        ax_acc.set_ylim(0.0, 1.0)
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Classes")
        ax_acc.set_title("Per-Class Accuracy")
        ax_acc.grid(axis="y", linestyle="--", alpha=0.3)

        for bar, acc in zip(bars, per_class_acc):
            ax_acc.text(
                bar.get_x() + bar.get_width()/2, acc + 0.02,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=8)

        self.logger.experiment.log({"per_class_accuracy_plot": wandb.Image(fig_acc)})
        plt.close(fig_acc)

        # 2. Confusion Matrix Heatmap
        fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
        im = ax_cm.imshow(confmat, cmap="Blues")

        # colorbar
        plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

        # ticks & labels
        ax_cm.set_xticks(range(num_classes))
        ax_cm.set_yticks(range(num_classes))
        ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
        ax_cm.set_yticklabels(class_names)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True Label")
        ax_cm.set_title("Confusion Matrix")

        # annotate cells
        thresh = confmat.max() * 0.6
        for i in range(num_classes):
            for j in range(num_classes):
                value = confmat[i, j].item()
                ax_cm.text(j, i, str(value),
                        ha="center", va="center",
                        color="white" if value > thresh else "black")

        plt.tight_layout()
        self.logger.experiment.log({"confusion_matrix": wandb.Image(fig_cm)})
        plt.close(fig_cm)
