import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from torchmetrics import Accuracy, ConfusionMatrix
from torch_geometric.nn import GCNConv

skeleton = [
    (0, 1), (1, 3),
    (3, 5), (1, 2),
    (0, 2), (2, 4),
    (4, 6),
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 6),             # shoulders
    (11, 12),           # hips
    (5, 11), (6, 12),   # torso
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]

class GNN(pl.LightningModule):
    def __init__(self, num_classes, label_dict, lr):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = 2
        self.hidden = 64

        # Spatial GNN layers
        self.gnn1 = GCNConv(self.in_channels, self.hidden)
        self.gnn2 = GCNConv(self.hidden, self.hidden)

        # Temporal convolution over 4 frames
        self.temporal_conv = nn.Conv1d(
            in_channels=self.hidden,
            out_channels=self.hidden,
            kernel_size=3,
            padding=1
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(self.hidden, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Precompute static skeleton graph
        edges = torch.tensor(skeleton, dtype=torch.long).t()  # shape (2, E)
        edges_undirected = torch.cat([edges, edges.flip(0)], dim=1)
        self.register_buffer("edge_index_base", edges_undirected)

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        B, T, V, C = x.shape
        x = x.reshape(B*T, V, C)
        x = x.reshape(B*T*V, C)   # flatten all nodes

        # Build edge index for all frames in the batch
        E = self.edge_index_base.shape[1]
        repeats = torch.arange(B*T, device=self.device).repeat_interleave(E) * V
        edge_index = self.edge_index_base.repeat(1, B*T) + repeats

        # Spatial GNN
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))

        # Reshape back
        x = x.reshape(B, T, V, self.hidden)

        # Temporal
        x = x.mean(dim=2)       # average over joints → (B, T, H)
        x = x.permute(0, 2, 1)  # → (B, H, T)

        x = self.temporal_conv(x)  # (B, H, T)
        x = x.mean(dim=2)          # temporal pooling

        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = self.train_acc(preds, y)

        self.confmat.update(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = self.val_acc(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
    
    def on_fit_end(self):
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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)