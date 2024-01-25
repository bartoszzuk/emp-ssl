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

# Lightning's flash implementation of LARS optimizer
# class LARS(Optimizer):
#
#     def __init__(self,
#                  params: Iterator[Parameter],
#                  lr: float,
#                  momentum: float = 0,
#                  dampening: float = 0,
#                  weight_decay: float = 0,
#                  nesterov: bool = False,
#                  trust_coefficient: float = 0.001,
#                  epsilon: float = 1e-8) -> None:
#
#         if lr is None or lr < 0.0:
#             raise ValueError(f'Invalid learning rate: {lr}')
#         if momentum < 0.0:
#             raise ValueError(f'Invalid momentum value: {momentum}')
#         if weight_decay < 0.0:
#             raise ValueError(f'Invalid weight_decay value: {weight_decay}')
#
#         defaults = {
#             'lr': lr,
#             'momentum': momentum,
#             'dampening': dampening,
#             'weight_decay': weight_decay,
#             'nesterov': nesterov,
#         }
#
#         if nesterov and (momentum <= 0 or dampening != 0):
#             raise ValueError('Nesterov momentum requires a momentum and zero dampening')
#
#         self.epsilon = epsilon
#         self.trust_coefficient = trust_coefficient
#
#         super().__init__(params, defaults)
#
#     def __setstate__(self, state: dict) -> None:
#         super().__setstate__(state)
#
#         for group in self.param_groups:
#             group.setdefault('nesterov', False)
#
#     @torch.no_grad()
#     def step(self, closure: Callable = None) -> Tensor:
#         loss = None
#
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
#
#         # exclude scaling for params with 0 weight decay
#         for group in self.param_groups:
#             weight_decay = group['weight_decay']
#             momentum = group['momentum']
#             dampening = group['dampening']
#             nesterov = group['nesterov']
#
#             for parameter in group['params']:
#                 if parameter.grad is None:
#                     continue
#
#                 parameter_norm = torch.norm(parameter.data)
#
#                 gradient = parameter.grad
#                 gradient_norm = torch.norm(parameter.grad.data)
#
#                 # lars scaling + weight decay part
#                 if weight_decay != 0 and parameter_norm != 0 and gradient_norm != 0:
#                     learning_rate = parameter_norm / (gradient_norm + parameter_norm * weight_decay + self.epsilon)
#                     learning_rate *= self.trust_coefficient
#
#                     gradient = gradient.add(parameter, alpha=weight_decay)
#                     gradient *= learning_rate
#
#                 # sgd part
#                 if momentum != 0:
#                     state = self.state[parameter]
#                     if 'momentum_buffer' not in state:
#                         buffer = state['momentum_buffer'] = torch.clone(gradient).detach()
#                     else:
#                         buffer = state['momentum_buffer']
#                         buffer.mul_(momentum).add_(gradient, alpha=1 - dampening)
#
#                     gradient = gradient.add(buffer, alpha=momentum) if nesterov else buffer
#
#                 parameter.add_(gradient, alpha=-group["lr"])
#
#         return loss
