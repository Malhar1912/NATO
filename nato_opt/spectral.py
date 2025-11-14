"""
nato_opt.spectral
------------------

Fourier Spectral Penalty utilities.

- configurable to apply penalty only to conv kernels and/or linear weights
- GPU-aware (uses torch.fft on the same device as the parameters)
- optional whitelist/blacklist of module names
"""

from typing import Optional, Iterable
import torch
import torch.nn as nn


def _get_module_map(model: nn.Module) -> dict:
    """
    Build a mapping from module name -> module object for lookup.
    """
    return dict(model.named_modules())


def _is_conv_module(module: nn.Module) -> bool:
    """
    Return True if module is a ConvNd (1D/2D/3D)
    """
    # All Conv modules in torch inherit from _ConvNd
    from torch.nn.modules.conv import _ConvNd  # local import for safety
    return isinstance(module, _ConvNd)


def _is_linear_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def fourier_spectral_penalty(
    model: nn.Module,
    lambda_fsp: float = 1e-6,
    include_conv: bool = True,
    include_linear: bool = True,
    module_whitelist: Optional[Iterable[str]] = None,
    module_blacklist: Optional[Iterable[str]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute a Fourier Spectral Penalty (FSP) applied only to selected parameter tensors.

    Behaviour:
      - By default only applies to parameters associated with Conv modules (Conv1d/2d/3d)
        and Linear modules, controlled by include_conv / include_linear.
      - You can further restrict by `module_whitelist` (only modules with names in the list
        will be considered) or exclude by `module_blacklist`.
      - The function infers the device from the model parameters if not provided and ensures
        the FFT is performed on the same device (so torch.fft runs on GPU when available).

    Args:
        model: torch.nn.Module
        lambda_fsp: scaling factor applied to the penalty (float)
        include_conv: if True, include convolutional kernels (ConvNd)
        include_linear: if True, include linear weights
        module_whitelist: iterable of module names to restrict to (optional)
        module_blacklist: iterable of module names to exclude (optional)
        device: torch.device to accumulate penalty on (optional). If None, inferred.

    Returns:
        torch.Tensor: scalar penalty ready to be added to the loss (on `device`).
    """
    if device is None:
        # infer device from first trainable parameter
        for _, p in model.named_parameters():
            if p.requires_grad:
                device = p.device
                break
        if device is None:
            device = torch.device("cpu")

    module_map = _get_module_map(model)

    penalty = torch.tensor(0.0, dtype=torch.float32, device=device)

    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # identify parent module for this parameter
        # e.g. pname "features.conv1.weight" -> module_name "features.conv1"
        if "." in pname:
            module_name = pname.rsplit(".", 1)[0]
        else:
            module_name = ""  # top-level parameter with no named module

        module = module_map.get(module_name, None)

        # Module name filters
        if module_whitelist is not None and module_name not in module_whitelist:
            continue
        if module_blacklist is not None and module_name in module_blacklist:
            continue

        # Select by module type and param name ("weight")
        is_weight = pname.endswith("weight")
        if not is_weight:
            # skip biases by default; you could add an option to include 1D params
            continue

        applies = False
        if module is not None:
            if include_conv and _is_conv_module(module):
                applies = True
            if include_linear and _is_linear_module(module):
                applies = applies or True  # include linear
        else:
            # No module found (top-level). Heuristic: apply if tensor is 2D and include_linear True
            if include_linear and p.dim() == 2:
                applies = True

        if not applies:
            continue

        # Move weight to device for FFT operation to leverage GPU if available
        w = p.detach().to(device)

        # Do N-D FFT on weight and compute squared magnitude sum
        w_fft = torch.fft.fftn(w)
        penalty = penalty + torch.sum(torch.abs(w_fft) ** 2)

    return (lambda_fsp * penalty).to(device)
