import lightning as L
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


class CIFAR10Data(L.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 128, train_workers: int = 6, val_workers: int = 2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.transform = Compose([ToTensor()])

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.val_dataset = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.val_workers,
            pin_memory=True,
            persistent_workers=True,
        )
