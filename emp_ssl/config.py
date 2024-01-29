import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self


@dataclass
class EvaluateConfig:
    dataset: Path | str = 'embeddings'
    batch_size: int = 100
    max_epochs: int = 100
    num_workers: int = os.cpu_count() - 1
    learning_rate: float = 0.03
    weight_decay: float = 5e-5
    embedding_dim: int = 4096
    seed: int = 42

    @classmethod
    def from_command_line(cls, arguments: argparse.Namespace) -> Self:
        return cls(**vars(arguments))


@dataclass
class PretrainConfig:
    dataset: Path | str = 'cifar10'
    invariance_coefficient: int = 200
    batch_size: int = 100
    train_patches: int = 20
    valid_patches: int = 128
    max_epochs: int = 1
    num_workers: int = os.cpu_count() - 1
    learning_rate: float = 0.03
    weight_decay: float = 1e-4
    projection_dim: int = 1024
    hidden_dim: int = 4096
    num_neighbours: int = 20
    temperature: float = 0.07
    seed: int = 42

    @classmethod
    def from_command_line(cls, arguments: argparse.Namespace) -> Self:
        return cls(**vars(arguments))
