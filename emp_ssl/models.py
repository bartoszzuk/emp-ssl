import torch
import torchmetrics
import torchvision
from torch import nn, Tensor

from emp_ssl.config import PretrainConfig


class ResNet18(nn.Module):

    def __init__(self, config: PretrainConfig):
        super().__init__()

        backbone = torchvision.models.resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        backbone = [layer for name, layer in backbone.named_children() if name not in {'maxpool', 'fc'}]

        head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, config.hidden_dim, bias=False),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU()
        )

        self.backbone = nn.Sequential(*backbone, head)
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=False),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.projection_dim)
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        embeddings = self.backbone(inputs)
        return self.projector(embeddings), embeddings


class KNearestNeighbours:

    def __init__(self, num_neighbours: int = 200, temperature: float = 0.5, num_labels: int = 10) -> None:
        self.num_neighbours = num_neighbours
        self.temperature = temperature
        self.num_labels = num_labels

        self.train_embeddings = []
        self.valid_embeddings = []
        self.train_labels = []
        self.valid_labels = []

    def add_train_samples(self, embeddings: Tensor, labels: Tensor) -> None:
        self.train_embeddings.append(embeddings)
        self.train_labels.append(labels)

    def add_valid_samples(self, embeddings: Tensor, labels: Tensor) -> None:
        self.valid_embeddings.append(embeddings)
        self.valid_labels.append(labels)

    def reset(self) -> None:
        self.train_embeddings = []
        self.valid_embeddings = []
        self.train_labels = []
        self.valid_labels = []

    def score(self, embeddings: Tensor) -> Tensor:
        train_embeddings = torch.concat(self.train_embeddings)
        similarity = torchmetrics.functional.pairwise_cosine_similarity(embeddings, train_embeddings)

        weights, indices = similarity.topk(k=self.num_neighbours, dim=-1)
        weights = torch.exp(weights / self.temperature)

        labels = torch.concat(self.train_labels).expand(embeddings.size(0), -1)
        labels = torch.gather(labels, dim=-1, index=indices)
        labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels)

        scores = torch.sum(labels * weights.unsqueeze(-1), dim=1)

        return scores

