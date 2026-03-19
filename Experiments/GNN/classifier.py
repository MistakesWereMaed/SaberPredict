import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from torchmetrics import Accuracy, ConfusionMatrix


# =========================
# Normalization
# =========================
def normalize_input(x):
    B, T, V, C = x.shape
    center = x.mean(dim=2, keepdim=True)
    k = x - center

    k_resh = k.reshape(B * T, V, C)
    dists = torch.cdist(k_resh, k_resh, p=2)
    max_d = dists.view(B, T, -1).max(dim=2)[0].view(B, T, 1, 1)

    return k / (max_d + 1e-6)


# =========================
# Graph (COCO 17)
# =========================
def get_adjacency(num_nodes=17):
    edges = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (5, 6), (11, 12),
        (5, 11), (6, 12)
    ]

    A = torch.zeros((num_nodes, num_nodes))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    A += torch.eye(num_nodes)
    D = torch.diag(torch.sum(A, dim=1) ** -0.5)
    return D @ A @ D


# =========================
# ST-GCN Block
# =========================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.1):
        super().__init__()
        self.A = A

        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(9,1),
                      padding=(4,0), stride=(stride,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)

        # Graph conv
        x = torch.einsum('bctv,vw->bctw', x, self.A)
        x = self.gcn(x)

        # Temporal conv
        x = self.tcn(x)

        return F.relu(x + res)


# =========================
# Lightning ST-GCN
# =========================
class STGCN(pl.LightningModule):
    """
    ST-GCN classifier (pose-based action recognition)
    Input: (B, T, V, C)
    Output: logits, embedding
    """

    def __init__(
        self,
        label_dict,
        class_weights=None,
        num_classes: int = 5,
        num_joints: int = 17,
        coord_dim: int = 4,
        channels: List[int] = [64, 64, 64, 128, 128, 256, 256],
        fc_hidden: int = 256,
        dropout: float = 0.1,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        use_onecycle: bool = True,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        A = get_adjacency(num_joints)
        self.register_buffer("A", A.to("cuda"))

        self.data_bn = nn.BatchNorm1d(num_joints * coord_dim)

        # Build ST-GCN layers
        layers = []
        in_ch = coord_dim

        for i, out_ch in enumerate(channels):
            stride = 2 if i in [3, 5] else 1
            layers.append(STGCNBlock(in_ch, out_ch, self.A, stride=stride, dropout=dropout))
            in_ch = out_ch

        self.layers = nn.ModuleList(layers)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.embedding_proj = nn.Sequential(
            nn.Linear(channels[-1], fc_hidden),
            nn.ReLU(),
            nn.LayerNorm(fc_hidden),
        )

        self.classifier = nn.Linear(fc_hidden, num_classes)

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        # Optim config
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_onecycle = use_onecycle
        self.max_epochs = max_epochs

    # =========================
    # Forward
    # =========================
    def forward(self, x):
        """
        x: (B, T, V, C)
        """
        B, T, V, C = x.shape

        x = normalize_input(x)

        # (B, T, V, C) -> (B, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Batch norm
        x = x.permute(0, 3, 1, 2).contiguous()  # (B,V,C,T)
        x = x.view(B, V*C, T)
        x = self.data_bn(x)
        x = x.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()

        # ST-GCN layers
        for layer in self.layers:
            x = layer(x)

        # Pool
        x = self.pool(x).view(B, -1)

        emb = self.embedding_proj(x)
        logits = self.classifier(emb)

        return logits, emb

    # =========================
    # Steps
    # =========================
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self._draw_plots()

    def _step(self, batch, batch_idx, mode):
        x = batch["keypoints"]
        y = batch["label"]

        logits, emb = self(x)

        class_weights = self.hparams.class_weights
        if class_weights is not None:
            weights = torch.tensor(
                [class_weights[i] for i in range(len(class_weights))],
                dtype=torch.float32,
                device=logits.device
            )
            loss = F.cross_entropy(logits, y, weight=weights)
        else:
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        preds = torch.argmax(logits, dim=1)

        metric_map = {
            "train": self.train_acc,
            "val": self.val_acc,
            "test": self.test_acc,
        }

        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if mode in metric_map:
            acc = metric_map[mode](preds, y)
            self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        if mode == "test":
            self.confmat(preds, y)

        return {"loss": loss, "preds": preds, "target": y, "emb": emb}

    # =========================
    # Optimizer
    # =========================
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.use_onecycle:
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
                return {"optimizer": optimizer,
                        "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
            except:
                return optimizer
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    # =========================
    # Plots
    # =========================
    def _draw_plots(self):
        confmat = self.confmat.compute().detach().cpu()

        per_class_acc = confmat.diag() / confmat.sum(axis=1).clip(min=1)
        num_classes = confmat.shape[0]
        class_names = [self.hparams.label_dict[i] for i in range(num_classes)]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(range(num_classes), per_class_acc)
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_ylim(0, 1)
        ax.set_title("Per-Class Accuracy")

        self.logger.experiment.log({"per_class_accuracy": wandb.Image(fig)})
        plt.close(fig)

        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        im = ax_cm.imshow(confmat, cmap="Blues")
        plt.colorbar(im, ax=ax_cm)

        ax_cm.set_xticks(range(num_classes))
        ax_cm.set_yticks(range(num_classes))
        ax_cm.set_xticklabels(class_names, rotation=45)
        ax_cm.set_yticklabels(class_names)

        self.logger.experiment.log({"confusion_matrix": wandb.Image(fig_cm)})
        plt.close(fig_cm)