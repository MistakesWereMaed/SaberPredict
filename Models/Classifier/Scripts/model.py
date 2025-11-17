import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import Accuracy
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
    def __init__(self, num_classes: int, lr: float = 1e-3):
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

    def _run_model(self, x):
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

    def forward(self, x):
        return self._run_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = self.train_acc(preds, y)

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
        acc = self.test_acc(preds, y)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)