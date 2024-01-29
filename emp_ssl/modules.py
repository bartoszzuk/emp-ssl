from pathlib import Path
from typing import Literal

import torch
import torchmetrics
from lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

from emp_ssl import losses
from emp_ssl.config import PretrainConfig, EvaluateConfig
from emp_ssl.models import KNearestNeighbours
from emp_ssl.optimizers import LARS


class PretrainModule(LightningModule):

    def __init__(self, model: nn.Module, knn: KNearestNeighbours, config: PretrainConfig) -> None:
        super().__init__()
        self.model = model
        self.knn = knn

        self.max_epochs = config.max_epochs
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.train_patches = config.train_patches
        self.valid_patches = config.valid_patches
        self.invariance_coefficient = config.invariance_coefficient
        self.train_steps = (50000 // config.batch_size) * config.max_epochs

        self.top1_accuracy = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=1)
        self.top5_accuracy = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=5)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        optimizer = LARS(self.model.parameters(),
                         self.learning_rate,
                         momentum=0.9,
                         nesterov=True,
                         weight_decay=self.weight_decay,
                         clip=True)

        scheduler = CosineAnnealingLR(optimizer, self.train_steps)
        scheduler = {'scheduler': scheduler, 'interval': 'step'}

        return [optimizer], [scheduler]

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        patches, _ = batch
        patches = torch.flatten(patches, end_dim=1)

        projections, _ = self.model(patches)
        projections = torch.nn.functional.normalize(projections)
        projections = projections.view(-1, self.train_patches, projections.size(-1))

        total_coding_rate = losses.total_coding_rate(projections)
        cosine_similarity = losses.cosine_similarity_loss(projections)

        loss = total_coding_rate + self.invariance_coefficient * cosine_similarity

        self.log('Train|Total Coding Rate', total_coding_rate, on_step=True)
        self.log('Train|Cosine Similarity', cosine_similarity, on_step=True)
        self.log('Train|Loss', loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], index: int, dataloader_idx: int = 0) -> None:
        patches, labels = batch
        patches = torch.flatten(patches, end_dim=1)

        projections, embeddings = self.model(patches)
        embeddings = embeddings.view(-1, self.valid_patches, embeddings.size(-1))
        embeddings = torch.mean(embeddings, dim=1)

        if dataloader_idx == 0:
            self.knn.add_train_samples(embeddings, labels)

        if dataloader_idx == 1:
            scores = self.knn.score(embeddings)

            self.top1_accuracy(scores, labels)
            self.top5_accuracy(scores, labels)
            self.log('Valid|Top1 Accuracy', self.top1_accuracy, on_epoch=True, add_dataloader_idx=False)
            self.log('Valid|Top5 Accuracy', self.top5_accuracy, on_epoch=True, add_dataloader_idx=False)

            # Add valid samples for embeddings dataset
            if self.current_epoch == self.max_epochs - 1:
                self.knn.add_valid_samples(embeddings, labels)

    def save_embeddings_dataset(self, embeddings: Tensor, labels: Tensor, split: Literal['train', 'valid']) -> None:
        root = Path(self.logger.log_dir) / 'embeddings' / split
        root.mkdir(parents=True, exist_ok=True)

        torch.save(embeddings, root / 'embeddings.pt')
        torch.save(labels, root / 'labels.pt')

        print(f'Epoch {self.current_epoch}, global step {self.global_step}: Saved {split} embeddings to {root}')

    def on_validation_end(self) -> None:
        # Do not save embeddings during sanity check
        if self.global_step != 0 and self.current_epoch == self.max_epochs - 1:
            train_embeddings = torch.concat(self.knn.train_embeddings)
            valid_embeddings = torch.concat(self.knn.valid_embeddings)

            train_labels = torch.concat(self.knn.train_labels)
            valid_labels = torch.concat(self.knn.valid_labels)

            self.save_embeddings_dataset(train_embeddings, train_labels, split='train')
            self.save_embeddings_dataset(valid_embeddings, valid_labels, split='valid')

        self.knn.reset()


class EvaluateModule(LightningModule):

    def __init__(self, model: nn.Module, config: EvaluateConfig) -> None:
        super().__init__()
        self.model = model

        self.weight_decay = config.weight_decay
        self.learning_rate = config.learning_rate
        self.max_epochs = config.max_epochs

        self.top1_accuracy = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=1)
        self.top5_accuracy = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=5)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    self.learning_rate,
                                    momentum=0.9,
                                    weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, self.max_epochs)

        return [optimizer], [scheduler]

    def training_step(self, batch: Tensor) -> Tensor:
        embeddings, labels = batch

        logits = self.model(embeddings)

        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log('Train|Loss', loss, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor) -> None:
        embeddings, labels = batch

        logits = self.model(embeddings)
        scores = torch.softmax(logits, dim=1)

        loss = torch.nn.functional.cross_entropy(logits, labels)

        self.top1_accuracy(scores, labels)
        self.top5_accuracy(scores, labels)

        self.log('Valid|Top5 Accuracy', self.top5_accuracy, on_epoch=True)
        self.log('Valid|Top1 Accuracy', self.top1_accuracy, on_epoch=True)
        self.log('Valid|Loss', loss, on_epoch=True, prog_bar=True)
