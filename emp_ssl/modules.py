import torch
import torchmetrics
from lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

from emp_ssl import losses
from emp_ssl.config import PretrainConfig
from emp_ssl.models import KNearestNeighbours
from emp_ssl.optimizers import LARS


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class PretrainModule(LightningModule):

    def __init__(self, model: nn.Module, knn: KNearestNeighbours, config: PretrainConfig) -> None:
        super().__init__()
        self.model = model
        self.knn = knn

        self.max_epochs = config.max_epochs
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.num_patches = config.num_patches
        self.invariance_coefficient = config.invariance_coefficient

        self.top1_accuracy_valid = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=1)
        self.top5_accuracy_valid = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=5)

    def configure_optimizers(self) -> (list[Optimizer], list[LRScheduler]):
        optimizer = LARS(self.model.parameters(),
                         self.learning_rate,
                         momentum=0.9,
                         nesterov=True,
                         weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, self.max_epochs)

        return [optimizer], [scheduler]

    def training_step(self, batch: Tensor) -> Tensor:
        patches, _ = batch
        patches = torch.flatten(patches, end_dim=1)

        projections, _ = self.model(patches)

        projections = torch.nn.functional.normalize(projections)
        projections = torch.stack(projections.split(self.num_patches))
        projections = torch.transpose(projections, 0, 1)

        total_coding_rate = losses.total_coding_rate(projections)
        cosine_similarity = losses.cosine_similarity_loss(projections)

        loss = total_coding_rate + self.invariance_coefficient * cosine_similarity

        self.log('Train|Total Coding Rate', total_coding_rate, on_step=True)
        self.log('Train|Cosine Similarity', cosine_similarity, on_step=True)
        self.log('Train|Loss', loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], index: int, dataloader_idx: int = 0) -> None:
        images, labels = batch

        _, embeddings = self.model(images)

        if dataloader_idx == 0:
            self.knn.add(embeddings, labels)

        if dataloader_idx == 1:
            scores = self.knn.score(embeddings)

            self.top1_accuracy_valid(scores, labels)
            self.top5_accuracy_valid(scores, labels)
            self.log('Valid|Top1 Accuracy', self.top1_accuracy_valid, on_epoch=True, add_dataloader_idx=False)
            self.log('Valid|Top5 Accuracy', self.top5_accuracy_valid, on_epoch=True, add_dataloader_idx=False)

    def on_validation_end(self) -> None:
        self.knn.reset()
