import torch
import torch.nn as nn
import pytorch_lightning as pl

import models._model as m

from torchmetrics import Accuracy, ConfusionMatrix

class model(pl.LightningModule):
    def __init__(
        self,
        label_dict: dict,
        num_classes: int,
        num_joints: int = 17,
        num_frames: int = 8,
        coord_dim: int = 2,
        dropout: float = 0.1,
        hidden_dim: int = 768,
        mlp_layers: int = 3,
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

        layers = []
        layers.append(nn.Linear(coord_dim * num_joints * num_frames, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(mlp_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.hidden_dim = hidden_dim

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

        # Flatten entire sequence per batch
        x = x.reshape(B, T * V * C)

        # Feed through MLP
        emb = self.mlp(x)
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