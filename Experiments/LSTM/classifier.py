import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
# LSTM Model
# =========================
class LSTMClassifier(pl.LightningModule):
    """
    LSTM baseline classifier
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
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        fc_hidden: int = 256,
        bidirectional: bool = False,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        use_onecycle: bool = True,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        input_size = num_joints * coord_dim
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.embedding_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_hidden),
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

        # Normalize
        x = normalize_input(x)

        # Flatten joints into feature vector
        x = x.view(B, T, V * C)  # (B, T, input_size)

        # LSTM
        out, _ = self.lstm(x)  # (B, T, hidden)

        # Take last timestep (standard baseline)
        last = out[:, -1]  # (B, hidden)

        emb = self.embedding_proj(last)
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
    # Optimizer (same as TCN)
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