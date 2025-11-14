import torch
from torch.optim import Optimizer
from typing import Optional


class NATOOptimizer(Optimizer):
    """
    NATO Optimizer (Noise-Augmented Tethered Optimization)

    Parameters
    ----------
    params : iterable
        Parameters to optimize.
    lr : float
        Base learning rate.
    beta1, beta2 : float
        Exponential decay rates for moment estimates.
    epsilon : float
        Numerical stability term.
    gamma : float
        Tether coefficient (encourages small changes relative to previous param).
    lr_decay : float
        Multiplicative LR decay factor.
    decay_step : int
        Number of epochs between decay steps.
    """

    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, gamma: float = 0.01, lr_decay: float = 0.95,
                 decay_step: int = 2):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
                        epsilon=epsilon, gamma=gamma,
                        lr_decay=lr_decay, decay_step=decay_step,
                        initial_lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, epoch: Optional[int] = None, closure: Optional[callable] = None):
        """
        Perform a single optimization step.

        Parameters
        ----------
        epoch : Optional[int]
            If provided, used to compute epoch-wise LR decay.
        closure : Optional[callable]
            A closure that re-evaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group.get("initial_lr", group.get("lr", 1e-3))
            decay_step = group.get("decay_step", 0)
            if epoch is not None and decay_step > 0:
                lr = lr * (group.get("lr_decay", 1.0) ** (epoch // decay_step))

            beta1 = group.get("beta1", 0.9)
            beta2 = group.get("beta2", 0.999)
            eps = group.get("epsilon", 1e-8)
            gamma = group.get("gamma", 0.01)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state.setdefault(p, {})

                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["prev"] = p.data.clone()

                m = state["m"]
                v = state["v"]
                prev = state["prev"]

                # update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                # bias-corrected estimates (simplified)
                m_hat = m / (1 - beta1)
                v_hat = v / (1 - beta2)

                update = lr * m_hat / (torch.sqrt(v_hat) + eps)

                # tether term encourages small steps relative to previous params
                update = update + gamma * (p.data - prev)

                p.data = p.data - update

                state["prev"] = p.data.clone()

        return loss
