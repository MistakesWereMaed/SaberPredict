import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

import models._model as m

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

class model(pl.LightningModule):
    def __init__(
        self,
        label_dict: dict,
        num_classes: int,
        num_joints: int = 17,
        num_frames: int = 8,
        coord_dim: int = 2,
        dropout: float = 0.1,
        fc_hidden: int = 768,
        embed_dim=768,
        gnn_hidden=768,
        attn_heads=64,
        center_lambda=0.1,
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

        # Center loss
        self.center_loss_fn = CenterLoss(
            num_classes=num_classes,
            feat_dim=gnn_hidden,   # embedding size before FC
            lambda_c=center_lambda
        )

        self.dropout = nn.Dropout(dropout)

        # 1. Node embedding from 2D coords
        self.node_embed = nn.Sequential(
            nn.Linear(coord_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, gnn_hidden)
        )

        # 2. GNN layers with residual connections
        self.gnn1 = GINConv(nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.ReLU(),
            nn.Linear(gnn_hidden, gnn_hidden)
        ))

        self.gnn2 = GINConv(nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.ReLU(),
            nn.Linear(gnn_hidden, gnn_hidden)
        ))

        # 3. LayerNorm for stability
        self.ln1 = nn.LayerNorm(gnn_hidden)
        self.ln2 = nn.LayerNorm(gnn_hidden)

        # 4. Temporal attention
        self.temporal_attn = nn.MultiheadAttention(gnn_hidden, attn_heads)

        # 5. Classifier
        self.fc = nn.Sequential(
            nn.Linear(gnn_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_classes)
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

        # optimizer config
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_onecycle = use_onecycle
        self.max_epochs = max_epochs

    def forward(self, x):
        """
        x: (B, T, V, C)
        returns: logits (B, num_classes), emb (B, fc_hidden)
        """
        B, T, V, C = x.shape
        assert V == self.V and T == self.T and C == self.coord_dim

        # Build edge index for batched graphs
        E = self.edge_index_base.shape[1]
        repeats = torch.arange(B*T, device=x.device).repeat_interleave(E) * V
        edge_index = self.edge_index_base.repeat(1, B*T) + repeats

        x = m.normalize_input(x)

        # Node embedding
        x = self.node_embed(x)
        x = self.dropout(x)

        # Flatten nodes
        x = x.reshape(B*T*V, -1)

        # GNN Layer 1 + residual
        h = F.relu(self.gnn1(x, edge_index))
        h = self.dropout(h)
        x = self.ln1(h + x)

        # GNN Layer 2 + residual
        h = F.relu(self.gnn2(x, edge_index))
        h = self.dropout(h)
        x = self.ln2(h + x)

        # Reshape
        x = x.reshape(B, T, V, -1)

        # Temporal pooling across joints
        x = x.mean(dim=2) # (B, T, H)

        # Multihead Attention
        x = x.permute(1, 0, 2) # (T, B, H)
        x, _ = self.temporal_attn(x, x, x)
        x = self.dropout(x)
        x = x.mean(dim=0) # (B, H)

        # Final embedding + classifier
        emb = x
        logits = self.fc(emb)

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