import torch
from torch import Tensor


def total_coding_rate(inputs: Tensor, epsilon: float = 0.2) -> Tensor:
    batch_size = inputs.shape[1]
    embedding_dim = inputs.shape[2]

    identity = torch.eye(embedding_dim, device=inputs.device)
    scalar = embedding_dim / (batch_size * epsilon)

    covariance = torch.matmul(inputs.transpose(1, 2), inputs)
    covariance = identity + scalar * covariance

    loss = torch.logdet(covariance) / 2

    return -1 * loss.mean()


def cosine_similarity_loss(inputs: Tensor) -> Tensor:
    centroid = inputs.mean(dim=0)

    similarity = torch.cosine_similarity(inputs, centroid, dim=-1)
    similarity = torch.mean(similarity, dim=1)

    return -1 * similarity.mean()
