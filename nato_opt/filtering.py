"""
nato_opt.filtering
------------------

Generalized N-D FFT gradient filtering utilities.

- Works on any tensor dimensionality >= 2 (skips 1-D biases by default)
- Ensures FFT runs on the gradient tensor's device (so CUDA path used when available)
- Type hints and docstrings added
"""

from typing import Optional
import torch
import numpy as np


def _center_slices(shape: torch.Size, cutoff_ratio: float):
    """
    Build a tuple of slice objects selecting a centered hyperrectangle in `shape`.
    Each slice length is max(1, int(dim * cutoff_ratio)).
    """
    slices = []
    for dim in shape:
        c = max(1, int(dim * cutoff_ratio))
        start = max(0, dim // 2 - c)
        end = min(dim, dim // 2 + c)
        slices.append(slice(start, end))
    return tuple(slices)


def low_pass_filter_gradients(
    model: torch.nn.Module,
    cutoff_ratio: float = 0.5,
    skip_bias: bool = True,
    in_place: bool = True,
) -> None:
    """
    Apply an N-dimensional centered low-pass filter to the gradients of `model`'s parameters.

    This function:
      - computes an N-D FFT of each parameter gradient tensor (on the tensor's device),
      - zeroes out high-frequency components outside a centered hyper-rectangle defined by
        `cutoff_ratio`,
      - inverse FFTs back and writes the real part into param.grad in-place.

    Args:
        model: torch.nn.Module
        cutoff_ratio: ratio of frequencies to keep in each dimension (0 < cutoff_ratio <= 1.0)
        skip_bias: whether to skip 1-D parameters (biases)
        in_place: whether to replace param.grad in-place (True recommended)
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.data
        if skip_bias and grad.dim() < 2:
            continue

        # Ensure we operate on the gradient device — allows GPU acceleration
        device = grad.device

        # Compute FFTn on the same device (torch.fft supports CUDA)
        grad_fft = torch.fft.fftn(grad)

        slices = _center_slices(grad_fft.shape, cutoff_ratio)

        # Create a mask selecting the low-frequency components
        mask = torch.zeros(grad_fft.shape, dtype=torch.bool, device=device)
        mask[slices] = True

        # Keep only low-frequency components
        filtered_fft = torch.zeros_like(grad_fft)
        filtered_fft[mask] = grad_fft[mask]

        # Inverse FFT (returns complex), take real part
        grad_filtered = torch.fft.ifftn(filtered_fft).real

        if in_place:
            param.grad.data.copy_(grad_filtered)
        else:
            param.grad.data = grad_filtered
