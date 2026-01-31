import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from torchmetrics import Accuracy, ConfusionMatrix

def normalize_input(x):
    # Simple per-sample center + scale normalization (same as earlier helpers)
    # center by mean across joints per frame, then divide by max pairwise dist across joints per frame.
    B, T, V, C = x.shape
    center = x.mean(dim=2, keepdim=True)  # (B, T, 1, C)
    k = x - center
    # estimate body size
    k_resh = k.reshape(B * T, V, C)
    dists = torch.cdist(k_resh, k_resh, p=2)  # (B*T, V, V)
    max_d = dists.view(B, T, -1).max(dim=2)[0].view(B, T, 1, 1)  # (B, T, 1, 1)
    eps = 1e-6
    return k / (max_d + eps)

class TemporalBlock(nn.Module):
    """Single residual temporal block: Conv1d -> ReLU -> Dropout -> Conv1d -> ReLU -> Dropout + residual"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation  # symmetric padding to preserve length (non-causal)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # adjust residual if channel dims differ
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(pl.LightningModule):
    """
    TCN baseline classifier (FenceNet-style).
    Input: x shape (B, T, V, C)  (e.g. (B, 8, 17, 2))
    Output: logits (B, num_classes), embedding (B, embed_dim)
    """
    def __init__(
        self,
        label_dict: dict,
        num_classes: int,
        num_joints: int = 17,
        coord_dim: int = 2,
        tcn_channels: List[int] = [128, 256, 512],  # channels for each temporal level
        kernel_size: int = 3,
        dropout: float = 0.1,
        fc_hidden: int = 768,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        use_onecycle: bool = True,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        # TCN input channels = V * coord_dim (one channel per coordinate per joint)
        in_ch = num_joints * coord_dim

        layers = []
        num_levels = len(tcn_channels)
        prev_channels = in_ch
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(prev_channels, tcn_channels[i], kernel_size, dilation, dropout))
            prev_channels = tcn_channels[i]
        self.network = nn.Sequential(*layers)

        tcn_out_ch = tcn_channels[-1]
        # embedding projection after temporal pooling (global time avg)
        self.embedding_proj = nn.Sequential(
            nn.Linear(tcn_out_ch, fc_hidden),
            nn.ReLU(),
            nn.LayerNorm(fc_hidden),
        )

        self.classifier = nn.Linear(fc_hidden, num_classes)

        # metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        # optimizer config
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_onecycle = use_onecycle
        self.max_epochs = max_epochs

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, V, C)
        returns: logits (B, num_classes), emb (B, fc_hidden)
        """
        B, T, V, C = x.shape

        # Normalize per-sample per-frame (center by mean, scale by max dist)
        x = normalize_input(x)  # (B,T,V,C)

        # reshape to (B, channels, T)
        # channels = V * C (each joint coordinate is its own channel)
        x_ch = x.permute(0, 2, 3, 1).reshape(B, V * C, T)  # (B, V*C, T)

        tcn_out = self.network(x_ch)  # (B, out_ch, T)

        # global temporal pooling (average)
        pooled = tcn_out.mean(dim=2)  # (B, out_ch)

        emb = self.embedding_proj(pooled)  # (B, fc_hidden)
        logits = self.classifier(emb)  # (B, num_classes)
        return logits, emb

    # training / validation steps

    def training_step(self, batch, batch_idx):
        return self._step(self, batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(self, batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(self, batch, batch_idx, "test")
    
    def on_test_epoch_end(self):
        self._draw_plots()

    def _step(self, batch, batch_idx, mode: str):
        """
        Shared logic for train/val/test steps.
        mode: one of {"train", "val", "test"}.
        """
        x = batch["keypoints"]        # (B, T, V, 2)
        y = batch["label"] 

        logits, emb = self(x)

        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        preds = torch.argmax(logits, dim=1)

        # Metric dictionary for cleaner code
        metric_map = {
            "train": self.train_acc,
            "val":   self.val_acc,
            "test":  self.test_acc,
        }

        # Logging
        prefix = f"{mode}_"
        self.log(prefix + "loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if mode in metric_map:
            acc = metric_map[mode](preds, y)
            self.log(prefix + "acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        # Confusion matrix only during test
        if mode == "test" and hasattr(self, "confmat"):
            self.confmat(preds, y)

        return {
            "loss": loss,
            "preds": preds,
            "target": y,
            "emb": emb,
            "logits": logits,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.use_onecycle:
            # OneCycleLR requires total_steps - use estimated from trainer if available
            scheduler = None
            try:
                total_steps = self.trainer.estimated_stepping_batches
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.lr,
                    total_steps=total_steps,
                    pct_start=0.1,
                    anneal_strategy="cos",
                    div_factor=25.0,
                    final_div_factor=1e4,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1
                    }
                }
            except Exception:
                # Trainer not attached yet -> return optimizer only and let the caller handle scheduling
                return optimizer
        else:
            # simple lr scheduler by epoch
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        
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