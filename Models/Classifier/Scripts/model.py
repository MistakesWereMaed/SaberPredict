import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from torchmetrics import Accuracy, ConfusionMatrix
from torch_geometric.nn import GINConv

EDGES = [
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

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c

        # Learnable centers
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        # features: (B, feat_dim)
        # labels: (B,)
        centers_batch = self.centers[labels]        # (B, feat_dim)
        return self.lambda_c * ((features - centers_batch)**2).mean()

class GNN(pl.LightningModule):
    def __init__(self, num_classes, label_dict, lr):
        super().__init__()
        self.save_hyperparameters()

        self.center_loss_fn = CenterLoss(
            num_classes=num_classes,
            feat_dim=64,   # size of embedding before FC
            lambda_c=0.1
        )
        
        # 1. Embed 2D coords
        self.node_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # 2. GNN layers with residual connections
        self.gnn1 = GINConv(nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))
        self.gnn2 = GINConv(nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))

        # 3. LayerNorm helps stability
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)

        # 4. Learnable attention over time
        self.temporal_attn = nn.MultiheadAttention(64, 4)

        # 5. Classifier
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

        # Precompute skeleton edges
        edges = torch.tensor(EDGES, dtype=torch.long).t()  # shape (2, E)
        edges_undirected = torch.cat([edges, edges.flip(0)], dim=1)
        self.register_buffer("edge_index_base", edges_undirected)
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = Accuracy(task="multiclass", num_classes=num_classes)
        self.confmat   = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        B, T, V, C = x.shape

        # Build edge index for all nodes in batch
        E = self.edge_index_base.shape[1]
        repeats = torch.arange(B*T, device=x.device).repeat_interleave(E) * V
        edge_index = self.edge_index_base.repeat(1, B*T) + repeats

        # Node embedding
        x = self.node_embed(x)

        # Flatten nodes
        x = x.reshape(B*T*V, -1)

        # GNN with residuals
        h = F.relu(self.gnn1(x, edge_index))
        x = self.ln1(h + x)

        h = F.relu(self.gnn2(x, edge_index))
        x = self.ln2(h + x)

        # Reshape back
        x = x.reshape(B, T, V, -1)

        # Temporal pooling
        x = x.mean(dim=2)  # average joints → (B, T, H)

        # Temporal attention
        x = x.permute(1,0,2)  # T,B,H
        x, _ = self.temporal_attn(x, x, x)
        x = x.mean(dim=0)  # B,H

        emb = x  # after temporal pooling (B, H)
        logits = self.fc(emb)
        return logits, emb
    
    def loss(self, preds, y):
        return F.cross_entropy(preds, y, label_smoothing=0.1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, emb = self(x)

        ce_loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        center_loss = self.center_loss_fn(emb, y)

        loss = ce_loss + center_loss

        self.log("train_loss", loss)
        self.log("train_ce", ce_loss)
        self.log("train_center_loss", center_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds, _ = self(x)
        loss = self.loss(preds, y)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds, _ = self(x)
        acc = self.val_acc(preds, y)

        self.log("test_acc", acc, prog_bar=True)
    
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        return optimizer