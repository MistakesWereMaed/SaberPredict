import torch
import torch.nn as nn
import pytorch_lightning as pl

import models._model as m

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

class model(pl.LightningModule):
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

        self.apply(m.init_weights)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, V, C)
        returns: logits (B, num_classes), emb (B, fc_hidden)
        """
        B, T, V, C = x.shape
        assert V == self.V and T == self.T and C == self.coord_dim, f"Expect (B,{self.T},{self.V},{self.coord_dim})"

        # Normalize per-sample per-frame (center by mean, scale by max dist)
        x = m.normalize_input(x)  # (B,T,V,C)

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
        return m.step(self, batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return m.step(self, batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return m.step(self, batch, batch_idx, "test")
    
    def on_test_epoch_end(self):
        m.draw_plots(self)

    def configure_optimizers(self):
        return m.configure_optimizers(self)