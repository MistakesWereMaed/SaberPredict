import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import List
from torchmetrics import Accuracy, ConfusionMatrix

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
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')
            nn.init.zeros_(self.downsample.bias)

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


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with increasing dilation"""
    def __init__(self, in_channels: int, channels: List[int], kernel_size: int = 3, dropout: float = 0.1):
        """
        channels: list of out_channels per level (len = n_levels)
        dilation doubles each layer: 1, 2, 4, ...
        """
        super().__init__()
        layers = []
        num_levels = len(channels)
        prev_channels = in_channels
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(prev_channels, channels[i], kernel_size, dilation, dropout))
            prev_channels = channels[i]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T)
        return self.network(x)  # (B, C_last, T)

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
        num_frames: int = 8,
        coord_dim: int = 2,
        tcn_channels: List[int] = [128, 128, 256],  # channels for each temporal level
        kernel_size: int = 3,
        dropout: float = 0.1,
        fc_hidden: int = 768,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        use_onecycle: bool = True,
        max_epochs: int = 25,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.V = num_joints
        self.T = num_frames
        self.coord_dim = coord_dim

        # TCN input channels = V * coord_dim (one channel per coordinate per joint)
        in_ch = self.V * self.coord_dim
        self.tcn = TemporalConvNet(in_channels=in_ch, channels=tcn_channels, kernel_size=kernel_size, dropout=dropout)

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

        self._init_weights()

    def _init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_fn)

    def _normalize_input(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, V, C)
        returns: logits (B, num_classes), emb (B, fc_hidden)
        """
        B, T, V, C = x.shape
        assert V == self.V and T == self.T and C == self.coord_dim, f"Expect (B,{self.T},{self.V},{self.coord_dim})"

        # Normalize per-sample per-frame (center by mean, scale by max dist)
        x = self._normalize_input(x)  # (B,T,V,C)

        # reshape to (B, channels, T)
        # channels = V * C (each joint coordinate is its own channel)
        x_ch = x.permute(0, 2, 3, 1).reshape(B, V * C, T)  # (B, V*C, T)

        tcn_out = self.tcn(x_ch)  # (B, out_ch, T)

        # global temporal pooling (average)
        pooled = tcn_out.mean(dim=2)  # (B, out_ch)

        emb = self.embedding_proj(pooled)  # (B, fc_hidden)
        logits = self.classifier(emb)  # (B, num_classes)
        return logits, emb

    # -------------------------
    # training / validation steps
    # -------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch

        logits, emb = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        preds = torch.argmax(logits, dim=1)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc(preds, y), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits, emb = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        preds = torch.argmax(logits, dim=1)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc(preds, y), prog_bar=True, on_step=False, on_epoch=True)

        return {"val_loss": loss, "preds": preds, "target": y}

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits, emb = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        preds = torch.argmax(logits, dim=1)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc(preds, y), prog_bar=True)

        self.confmat(preds, y)

        return {"test_loss": loss, "preds": preds, "target": y}
    
    def on_test_epoch_end(self):
        # Compute confusion matrix and add per-class accuracies
        confmat = self.confmat.compute().detach().cpu()
        per_class_acc = confmat.diag() / confmat.sum(dim=1).clamp(min=1)

        num_classes = confmat.shape[0]
        class_names = [self.hparams.label_dict[i] for i in range(num_classes)]

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(range(num_classes), per_class_acc, align="center")

        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Classes")

        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()

        # annotate bar values
        for bar, acc in zip(bars, per_class_acc):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{acc:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        # Log to wandb
        self.logger.experiment.log({"per_class_accuracy_plot": wandb.Image(fig)})
        plt.close(fig)

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