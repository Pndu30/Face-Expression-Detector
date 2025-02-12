import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class ResEmoteNetTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        loss=nn.CrossEntropyLoss,
        optim=Adam,
        sched=None,
        lr=0.0001,
        decay=0.01,
        momentum=0.9,
        n_class=7
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
        else:
            self.sched = sched

        self.acc = MulticlassAccuracy(num_classes=n_class)
        self.f1 = MulticlassF1Score(num_classes=n_class)

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = self.optim
        if self.sched is not None:
            scheduler = self.sched
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.predict_step(batch, batch_idx)
        val_loss = self.loss(y_hat, y)
        val_acc = self.acc(y_hat, y)
        val_f1 = self.f1(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc.item(), prog_bar=True)
        self.log("val_f1", val_f1.item(), prog_bar=True)
        return {"val_loss": val_loss, "val_acc": val_acc.item(), "val_f1": val_f1.item()}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.predict_step(batch, batch_idx)
        test_loss = self.loss(y_hat, y)
        test_acc = self.acc(y_hat, y)
        test_f1 = self.f1(y_hat, y)
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", test_acc.item(), prog_bar=True)
        self.log("test_f1", test_f1.item(), prog_bar=True)
        return {"test_loss": test_loss, "test_acc": test_acc.item(), "test_f1": test_f1.item()}
        

