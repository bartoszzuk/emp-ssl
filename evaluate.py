import argparse
import os
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from emp_ssl.config import EvaluateConfig
from emp_ssl.data import EvaluateDataModule
from emp_ssl.modules import EvaluateModule

torch.set_float32_matmul_precision('medium')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='embeddings')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count() - 1)
    parser.add_argument('--learning-rate', type=float, default=0.03)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--hidden-dim', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def evaluate():
    config = EvaluateConfig.from_command_line(parse_arguments())

    seed_everything(config.seed)

    data = EvaluateDataModule(config)

    model = nn.Linear(config.hidden_dim, 10)
    model = EvaluateModule(model, config)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            monitor='Valid|Top1 Accuracy',
            save_top_k=1,
            save_last=True,
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
        logger=TensorBoardLogger(save_dir='evaluate-logs', name=''),
        callbacks=callbacks,
        deterministic=True
    )

    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    evaluate()
