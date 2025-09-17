import torch
import torch.nn as nn
import torchvision.models.video as models
import pytorch_lightning as pl

from torchmetrics.classification import Accuracy

# ---------------------------
# Model definition
# ---------------------------
class FencerScorer(pl.LightningModule):
    def __init__(self, lr=1e-4, num_classes=2):
        super().__init__()
        # 3D ResNet18 backbone
        self.model = models.r3d_18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Loss + metric
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters()  # saves lr, num_classes, etc.

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        videos, labels = batch  # [B, C, T, H, W], [B]
        logits = self(videos)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)