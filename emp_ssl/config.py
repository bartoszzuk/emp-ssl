import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self


@dataclass
class EvaluateConfig:
    dataset: Path | str = 'embeddings'
    batch_size: int = 50
    max_epochs: int = 100
    num_workers: int = os.cpu_count() - 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 4096
    seed: int = 42
    dev: bool = False

    @classmethod
    def from_command_line(cls, arguments: argparse.Namespace) -> Self:
        return cls(**vars(arguments))


@dataclass
class PretrainConfig:
    dataset: Path | str = 'cifar10'
    batch_size: int = 50
    train_patches: int = 20
    valid_patches: int = 128
    max_epochs: int = 1
    num_workers: int = os.cpu_count() - 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    projection_dim: int = 1024
    hidden_dim: int = 4096
    num_neighbours: int = 200
    temperature: float = 0.5
    invariance_coefficient: int = 200
    seed: int = 42
    dev: bool = False

    @classmethod
    def from_command_line(cls, arguments: argparse.Namespace) -> Self:
        return cls(**vars(arguments))
