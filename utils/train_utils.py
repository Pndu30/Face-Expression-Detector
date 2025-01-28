import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.functional.segmentation import generalized_dice_score as dice
from pathlib import Path
from monai.losses import DiceCELoss, DiceLoss
import pytorch_lightning as pl
from torch.optim import AdamW
import monai.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer, SliceInferer


class HardUnetTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        loss=DiceLoss,
        optim=AdamW,
        sched=CosineAnnealingLR,
        lr=0.0001,
        decay=0.01,
        momentum=0.9,
        device="cpu",
        roi_size_w=128,
        roi_size_h=128,
    ):
        super().__init__()
        self.net = model
        self.loss = loss()
        self.dice_metric1 = DiceMetric(reduction="mean_batch", get_not_nans=True)
        self.dice_metric2 = DiceMetric(reduction="mean_batch", get_not_nans=True)
        self.max_epochs = 500
        self.post1 = transforms.Compose([transforms.Activations(sigmoid=True)])
        self.post2 = transforms.Compose([transforms.AsDiscrete(threshold=0.5)])
        
        if isinstance(optim, torch.optim.SGD):
            self.optim = optim(self.net.parameters(), lr=lr, weight_decay=decay, momentum=momentum)
        else:
            self.optim = optim(self.net.parameters(), lr=lr, weight_decay=decay)
        
        if sched is not None:
            if sched == torch.optim.lr_scheduler.CosineAnnealingLR:
                self.sched = sched(self.optim, T_max=self.max_epochs)
            else:
                self.sched = sched(self.optim)
        self.save_hyperparameters(ignore=["unet", "loss"])

    def num_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = self.optim
        scheduler = self.sched
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"].float()
        y_hat = self.net(x)
        y_hat = self.post1(y_hat)
        y_hat = self.post2(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"].float()
        y_hat = self.net(x)
        y_hat = self.post1(y_hat)
        loss = self.loss(y_hat, y)
        y_hat = self.post2(y_hat)
        train_dice = self.dice_metric2(y_hat, y)
        mean_train_dice, _ = self.dice_metric2.aggregate()
        self.log("mean_train_dice", mean_train_dice, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.dice_metric2.reset()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"].float()
        y_hat = self.predict_step(batch, batch_idx)
        val_dice = self.dice_metric1(y_hat, y)
        return {"val_dice": val_dice}

    def on_validation_epoch_end(self):
        mean_val_dice, _ = self.dice_metric1.aggregate()
        self.log("val_dice", mean_val_dice, prog_bar=True)
        self.dice_metric1.reset()

