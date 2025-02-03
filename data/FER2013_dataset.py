import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FER2013
from pathlib import Path
import os
import pytorch_lightning as pl

class ISLESDataModule_2D(pl.LightningDataModule):
    def __init__(
        self,
        data_properties,
        modalities=["dwi"],
        fold=0,
        batch_size=2,
        num_workers=0,
        device="cpu",
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.data_properties = data_properties
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = kwargs

        self.train_transform = self.get_train_transform()
        self.val_transform = self.get_val_transform()
        self.train_set = self.val_set = self.test_set = None

    def setup(self, train_size=None, stage=None):
        train_data = []
        val_data = []

        self.train_set = Dataset(
            train_data, **self.dataset_kwargs
        )

        self.val_set = Dataset(
            val_data, **self.dataset_kwargs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.num_workers,
        )
    
    def get_train_transform(self):
        train_transform = [
            transforms.ToTensord(["image", "label"], device=self.device),
        ]
        return transforms.Compose(train_transform)

    def get_val_transform(self):
        val_transform = [
            transforms.ToTensord(["image", "label"], device=self.device),
        ]
        return transforms.Compose(val_transform)

if __name__ == '__main__':
    import json
    import torch
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    with open(fr'.\src\data\ISLES_dataset.json', 'r') as file:
        data = json.load(file)
    datamodule = ISLESDataModule_2D(batch_size=64, data_properties=data, modalities=['dwi'])

    # # Total image to read in. In this case, it's 10 (for both train and val). With split = 0.7, 7 wll go to train and 3 will go to val
    datamodule.setup()

    # #Loadin the data according to the upper parameters
    train_loader = datamodule.train_dataloader()
    print(train_loader)
    print(len(train_loader))
    for batch in train_loader:
        print(batch["image"].shape)
        print(torch.unique(batch["label"]))
        break