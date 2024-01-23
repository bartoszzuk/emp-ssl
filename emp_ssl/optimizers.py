from typing import Callable, Iterator

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer


# Lightning's flash implementation of LARS optimizer
class LARS(Optimizer):

    def __init__(self,
                 params: Iterator[Parameter],
                 lr: float,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 trust_coefficient: float = 0.001,
                 epsilon: float = 1e-8) -> None:

        if lr is None or lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'dampening': dampening,
            'weight_decay': weight_decay,
            'nesterov': nesterov,
        }

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError('Nesterov momentum requires a momentum and zero dampening')

        self.epsilon = epsilon
        self.trust_coefficient = trust_coefficient

        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure: Callable = None) -> Tensor:
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for parameter in group['params']:
                if parameter.grad is None:
                    continue

                parameter_norm = torch.norm(parameter.data)

                gradient = parameter.grad
                gradient_norm = torch.norm(parameter.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0 and parameter_norm != 0 and gradient_norm != 0:
                    learning_rate = parameter_norm / (gradient_norm + parameter_norm * weight_decay + self.epsilon)
                    learning_rate *= self.trust_coefficient

                    gradient = gradient.add(parameter, alpha=weight_decay)
                    gradient *= learning_rate

                # sgd part
                if momentum != 0:
                    state = self.state[parameter]
                    if 'momentum_buffer' not in state:
                        buffer = state['momentum_buffer'] = torch.clone(gradient).detach()
                    else:
                        buffer = state['momentum_buffer']
                        buffer.mul_(momentum).add_(gradient, alpha=1 - dampening)

                    gradient = gradient.add(buffer, alpha=momentum) if nesterov else buffer

                parameter.add_(gradient, alpha=-group["lr"])

        return loss
