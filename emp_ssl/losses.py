import torch
from torch import Tensor, nn


class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  # [d, B]
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def forward(self, X):
        return - self.compute_discrimn_loss(X.T)


class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)

        z_sim = 0
        for i in range(num_patch):
            z_sim += torch.nn.functional.cosine_similarity(z_list[i], z_avg, dim=1).mean()

        z_sim = z_sim / num_patch
        z_sim_out = z_sim.clone().detach()

        return -z_sim, z_sim_out


# Expected: -43.9996
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
    centroid = centroid.unsqueeze(dim=0)
    centroid = centroid.expand_as(inputs)

    similarity = torch.cosine_similarity(inputs, centroid)

    return -1 * similarity.mean()

# class Similarity_Loss(nn.Module):
#     def __init__(self, ):
#         super().__init__()
#         pass
#
#     def forward(self, z_list, z_avg):
#         z_sim = 0
#         num_patch = len(z_list)
#         z_list = torch.stack(list(z_list), dim=0)
#         z_avg = z_list.mean(dim=0)
#
#         z_sim = 0
#         for i in range(num_patch):
#             z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
#
#         z_sim = z_sim / num_patch
#         z_sim_out = z_sim.clone().detach()
#
#         return -z_sim, z_sim_out


# class TotalCodingRate(nn.Module):
#     def __init__(self, eps=0.01):
#         super(TotalCodingRate, self).__init__()
#         self.eps = eps
#
#     def compute_discrimn_loss(self, W):
#         """Discriminative Loss."""
#         p, m = W.shape  # [d, B]
#         I = torch.eye(p, device=W.device)
#         scalar = p / (m * self.eps)
#         logdet = torch.logdet(I + scalar * W.matmul(W.T))
#         return logdet / 2.
#
#     def forward(self, X):
#         return - self.compute_discrimn_loss(X.T)
