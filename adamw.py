from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {'lr': lr, 'weight_decay': weight_decay, 'betas': betas, 'eps': eps}
        super().__init__(params, defaults)

    def _get_state(self, p: torch.nn.parameter.Parameter) -> tuple[int, float, float]:
        state = self.state[p]
        return state.get('t', 1), state.get('m', 0), state.get('v', 0)
    
    def _set_state(self, p: torch.nn.parameter.Parameter, t: int, m: float, v: float):
        state = self.state[p]
        state['t'] = t + 1
        state['m'] = m
        state['v'] = v

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:

            lr, weight_decay = group['lr'], group['weight_decay']
            betas, eps = group['betas'], group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # t: iteration
                # m: estimated first moment
                # v: estimated second moment
                t, m, v = self._get_state(p)
                
                # Gradient of the loss at the current tiem step
                grad = p.grad.data

                # Update the first moment estimate
                m = betas[0] * m + (1 - betas[0]) * grad

                # Update the second moment estimate
                v = betas[1] * v + (1 - betas[1]) * grad * grad

                # Compute adjusted learning rate for iteration t
                lr_t = lr * (math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t))

                # Update the parameters
                p.data -= lr_t * (m / (torch.sqrt(v) + eps))

                # Apply weight decay
                p.data -= lr * weight_decay * p.data

                # Save 
                self._set_state(p, t, m, v)
        return loss


def lr_cosine_schedule(it: int, 
                       max_learning_rate: float, 
                       min_learning_rate: float,
                       warmup_iters: int,
                       cosine_cycle_iters: int):
    # warm-up
    if it < warmup_iters:
        return (it * max_learning_rate) / warmup_iters
    
    # post-annealing
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # cosine annealing
    return min_learning_rate + 0.5 * (1 + math.cos((math.pi * (it - warmup_iters)) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]

    total_norm = 0.0
    for g in grads:
        total_norm += torch.sum(g * g)
    total_norm = math.sqrt(total_norm)

    if total_norm < max_l2_norm:
        return 
    
    scale_factor = max_l2_norm / (total_norm + 1e-6)

    for p in parameters:
        if p.grad is None:
            continue
        p.grad.mul_(scale_factor)
    