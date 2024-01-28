from typing import Callable, Iterator

import torch
from torch.nn import Parameter
from torch.optim import SGD


class LARS(SGD):

    def __init__(self,
                 params: Iterator[Parameter],
                 lr: float,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 clip: bool = False,
                 trust_coefficient: float = 0.005,
                 epsilon: float = 1e-8) -> None:
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.clip = clip
        self.epsilon = epsilon
        self.trust_coefficient = trust_coefficient

    @torch.no_grad()
    def step(self, closure: Callable = None) -> None:
        weight_decays = []

        for group in self.param_groups:
            weight_decay = group.get('weight_decay', 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group['weight_decay'] = 0

            # update the parameters
            for parameter in group['params']:
                if parameter.grad is not None and parameter.ndim != 1:
                    self.update_parameter(parameter, group, weight_decay)

        # update the optimizer
        super().step(closure=closure)

        # return weight decay control to optimizer
        for index, group in enumerate(self.param_groups):
            group['weight_decay'] = weight_decays[index]

    def update_parameter(self, parameter: Parameter, group: dict, weight_decay: float) -> None:
        # calculate new norms
        parameter_norm = torch.norm(parameter.data)
        gradients_norm = torch.norm(parameter.grad.data)

        if parameter_norm != 0 and gradients_norm != 0:
            learning_rate = self.trust_coefficient * parameter_norm
            learning_rate /= (gradients_norm + parameter_norm * weight_decay + self.epsilon)

            # clip learning rate
            if self.clip:
                learning_rate = min(learning_rate / group['lr'], 1)

            # update params with clipped learning rate
            parameter.grad.data += weight_decay * parameter.data
            parameter.grad.data *= learning_rate
