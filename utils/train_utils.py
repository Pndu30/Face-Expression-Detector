import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class HardUnetTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        loss=nn.BCELoss,
        optim=AdamW,
        sched=CosineAnnealingLR,
        lr=0.0001,
        decay=0.01,
        momentum=0.9,
        device="cpu",
    ):
        super().__init__()
        self.model = model
        self.loss = loss()
        self.max_epochs = 500
        
        if isinstance(optim, torch.optim.SGD):
            self.optim = optim(self.model.parameters(), lr=lr, weight_decay=decay, momentum=momentum)
        else:
            self.optim = optim(self.model.parameters(), lr=lr, weight_decay=decay)
        
        if sched is not None:
            if sched == torch.optim.lr_scheduler.CosineAnnealingLR:
                self.sched = sched(self.optim, T_max=self.max_epochs)
            else:
                self.sched = sched(self.optim)

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = self.optim
        scheduler = self.sched
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"].float()
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"].float()
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"].float()
        y_hat = self.predict_step(batch, batch_idx)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        return {"val_loss": val_loss}

