import argparse
import os
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from emp_ssl.config import PretrainConfig
from emp_ssl.data import PretrainDataModule
from emp_ssl.models import ResNet, KNearestNeighbours
from emp_ssl.modules import PretrainModule

torch.set_float32_matmul_precision('medium')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='cifar10')
    parser.add_argument('--invariance-coefficient', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train-patches', type=int, default=20)
    parser.add_argument('--valid-patches', type=int, default=64)
    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count() - 1)
    parser.add_argument('--learning-rate', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--projection-dim', type=int, default=1024)
    parser.add_argument('--hidden-dim', type=int, default=2048)
    parser.add_argument('--num-neighbours', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev', action='store_true')

    return parser.parse_args()


def pretrain():
    config = PretrainConfig.from_command_line(parse_arguments())

    seed_everything(config.seed)

    data = PretrainDataModule(config)
    knn = KNearestNeighbours(config.num_neighbours, config.temperature)

    model = ResNet(config)
    model = PretrainModule(model, knn, config)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            monitor='Valid|Top1 Accuracy',
            save_top_k=1,
            mode='max',
            verbose=True,
            filename='{epoch}-{Valid|Top1 Accuracy:.2f}',
        ),
    ]

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        max_epochs=config.max_epochs,
        logger=TensorBoardLogger(save_dir='logs', name=''),
        callbacks=callbacks,
        deterministic=True,
        check_val_every_n_epoch=config.max_epochs,
        limit_train_batches=100 if config.dev else 1.0,
        limit_val_batches=100 if config.dev else 1.0,
    )

    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    pretrain()
