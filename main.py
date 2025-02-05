import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers  import WandbLogger
import wandb
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers  import WandbLogger
from utils.cam_utils import cam_init, cam_loop
from models.ResEmoteNet import ResEmoteNet
from data.FER2013_dataset import FER2013Dataset
from utils.train_utils import ResEmoteNetTrainer


def train_model(model, lr=0.0015, optim=torch.optim.AdamW, 
                loss_fn=torch.nn.CrossEntropyLoss, scheduler=None, 
                epochs=800, val_epoch=20, batch=64, wandb=True, 
                wandb_name='awa', save=True, PATH='./models/weights/weights.pth'):
    datamodule = FER2013Dataset(batch_size=batch)
    datamodule.setup()
    if wandb:
        wandb.init(project=wandb_name)
        logger = WandbLogger()
    net = ResEmoteNet(inch=3, outch=7, softmax=False)
    model = ResEmoteNetTrainer(model=net, lr=lr, optim=optim, sched=scheduler, loss=loss_fn)
    test_loader = datamodule.test_dataloader()

    # initialize Lightning's trainer.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=val_epoch,
    )

    # train
    trainer.fit(model, datamodule)
    trainer.test(model, test_loader)
    if wandb:
        wandb.finish()

    if save:
        torch.save(model.state_dict(), PATH)

def main():
    
    while True:
        pass