import random
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageFilter
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

from emp_ssl.config import PretrainConfig, EvaluateConfig


class RandomGaussianBlur:

    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, image: Image) -> Image:
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            image = image.filter(ImageFilter.GaussianBlur(sigma))

        return image


class MultiPatchAugmentation:

    def __init__(self, num_patches: int) -> None:
        self.num_patches = num_patches
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.25, 0.25), ratio=(1, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(),
            RandomGaussianBlur(p=0.1),
            transforms.RandomSolarize(0.5, p=0.2),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __call__(self, image: Image) -> Tensor:
        patches = [self.transform(image) for _ in range(self.num_patches)]
        patches = torch.stack(patches)

        return patches


class PretrainDataModule(LightningDataModule):

    def __init__(self, config: PretrainConfig) -> None:
        super().__init__()

        self.train_batch_size = config.batch_size
        self.valid_batch_size = config.batch_size
        self.num_workers = config.num_workers

        root = config.dataset

        train_transform = MultiPatchAugmentation(config.train_patches)
        valid_transform = MultiPatchAugmentation(config.valid_patches)

        self.train_dataset = CIFAR10(root, transform=train_transform, download=True, train=True)
        self.valid_source_dataset = CIFAR10(root, transform=valid_transform, download=True, train=True)
        self.valid_target_dataset = CIFAR10(root, transform=valid_transform, download=True, train=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, self.train_batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(self.valid_source_dataset, batch_size=self.valid_batch_size, num_workers=self.num_workers),
            DataLoader(self.valid_target_dataset, batch_size=self.valid_batch_size, num_workers=self.num_workers)
        ]


def load_embeddings_dataset(root: Path) -> TensorDataset:
    embeddings = torch.load(root / 'embeddings.pt')
    labels = torch.load(root / 'labels.pt')

    return TensorDataset(embeddings, labels)


class EvaluateDataModule(LightningDataModule):

    def __init__(self, config: EvaluateConfig) -> None:
        super().__init__()

        self.train_dataset = load_embeddings_dataset(config.dataset / 'train')
        self.valid_dataset = load_embeddings_dataset(config.dataset / 'valid')

        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, self.batch_size, num_workers=0)




