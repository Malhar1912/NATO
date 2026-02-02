"""
nato_opt.kakeya
---------------

Kakeya Directional Penalty for gradient optimization.

This module implements a directional consistency penalty based on the
Kakeya conjecture principles. It penalizes rapid changes in gradient
direction by computing the cosine similarity between consecutive gradients.

The penalty encourages smoother optimization trajectories by discouraging
erratic gradient direction changes, which can lead to more stable training.

Mathematical Formulation:
    penalty = λ_k * Σ cos²(grad_t, grad_{t-1})

Where:
    - grad_t is the current gradient
    - grad_{t-1} is the previous gradient
    - λ_k is the penalty coefficient

References:
    - Kakeya conjecture and directional optimization theory
    - NATO Optimizer technical documentation
"""

from typing import Dict, Any
import torch
import torch.nn as nn


def kakeya_directional_penalty(
    model: nn.Module,
    state: Dict[nn.Parameter, Dict[str, Any]],
    lambda_k: float = 1e-4
) -> torch.Tensor:
    """
    Compute Kakeya directional penalty based on gradient direction consistency.

    This penalty function tracks gradient directions across training steps and
    penalizes parameters whose gradients maintain high cosine similarity with
    their previous gradients. This encourages exploration of diverse gradient
    directions during optimization.

    Args:
        model: PyTorch model whose parameters' gradients will be analyzed.
        state: A dictionary to store previous gradients for each parameter.
               This should be a persistent dict maintained across training steps.
               Structure: {parameter: {'prev_grad': Tensor}}
        lambda_k: Penalty coefficient (default: 1e-4). Higher values increase
                  the penalty strength.

    Returns:
        torch.Tensor: Scalar penalty value to be added to the loss function.
                      Returns 0.0 if no previous gradients are stored.

    Example:
        >>> import torch
        >>> from nato_opt import kakeya_directional_penalty
        >>> 
        >>> model = torch.nn.Linear(10, 5)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> kakeya_state = {}  # persistent state dict
        >>> 
        >>> # Training loop
        >>> for inputs, targets in dataloader:
        ...     optimizer.zero_grad()
        ...     outputs = model(inputs)
        ...     loss = criterion(outputs, targets)
        ...     
        ...     # Add Kakeya penalty after backward (gradients must exist)
        ...     loss.backward()
        ...     k_penalty = kakeya_directional_penalty(model, kakeya_state)
        ...     total_loss = loss + k_penalty
        ...     
        ...     optimizer.step()

    Note:
        - Gradients must exist (call .backward() before this function)
        - The state dict is modified in-place to store previous gradients
        - First call for each parameter returns 0 penalty (no previous gradient)
    """
    penalty = torch.tensor(0.0, dtype=torch.float32)
    
    # Infer device from first parameter with gradient
    device = None
    for p in model.parameters():
        if p.grad is not None:
            device = p.device
            penalty = penalty.to(device)
            break
    
    if device is None:
        return penalty  # No gradients available

    for p in model.parameters():
        if p.grad is None:
            continue

        grad = p.grad.view(-1)

        # Initialize state for this parameter if not exists
        if p not in state:
            state[p] = {}

        if 'prev_grad' not in state[p]:
            state[p]['prev_grad'] = grad.detach().clone()
            continue

        prev_grad = state[p]['prev_grad']

        # Compute cosine similarity between current and previous gradient
        grad_norm = torch.norm(grad)
        prev_norm = torch.norm(prev_grad)
        
        # Avoid division by zero
        if grad_norm > 1e-8 and prev_norm > 1e-8:
            cos_sim = torch.dot(grad, prev_grad) / (grad_norm * prev_norm + 1e-8)
            penalty = penalty + cos_sim ** 2

        # Update stored gradient for next iteration
        state[p]['prev_grad'] = grad.detach().clone()

    return lambda_k * penalty
