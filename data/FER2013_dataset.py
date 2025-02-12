import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import FER2013
from pathlib import Path
import os
import pytorch_lightning as pl
import csv
import numpy as np


class FER2013Dataset(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=2,
        num_workers=0,
        device="cpu",
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = kwargs

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def setup(self, train_size=None, stage=None):
        train_path = os.path.join(os.getcwd(), 'data', 'FER2013', 'train')
        test_path = os.path.join(os.getcwd(), 'data', 'FER2013', 'test')

        # Load data using ImageFolder which assumes a folder structure where each subfolder is a class
        self.train_dataset = datasets.ImageFolder(root=train_path, transform=self.transform)
        self.test_dataset = datasets.ImageFolder(root=test_path, transform=self.transform)
        # print(self.train_dataset.class_to_idx)

        val_size = int(0.2 * len(self.train_dataset))
        train_size = len(self.train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == '__main__':
    datamodule = FER2013Dataset(batch_size=12)

    # # Total image to read in. In this case, it's 10 (for both train and val). With split = 0.7, 7 wll go to train and 3 will go to val
    datamodule.setup()

    #Loadin the data according to the upper parameters
    train_loader = datamodule.train_dataloader()
    print(train_loader)
    print(len(train_loader))    
    for batch in train_loader:
        x, y = batch
        print(x.shape, y.shape)
        print(torch.unique(x))
        print(y[0])
        break