"""
nato_opt.fft_helpers

Helpers for GPU-optimized FFT usage, and optional TorchScript/tracing support.

Note:
- torch.fft.fftn runs on GPU if the tensor is on CUDA.
- Full TorchScript support for torch.fft.* can be limited; we provide an optional
  tracing helper that produces a callable by tracing the operation on a sample tensor.
"""

from typing import Callable, Optional
import torch


def get_fftn_callable(sample_tensor: Optional[torch.Tensor] = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return a callable `fftn(tensor)` that performs an N-D FFT on the input tensor.
    If `sample_tensor` is provided, the function will attempt to produce a traced
    callable using `torch.jit.trace` which can be saved/used in TorchScript contexts.

    Args:
        sample_tensor: example tensor used for tracing (optional).

    Returns:
        Callable that accepts a Tensor and returns its fftn (complex) result.
    """
    def _fftn_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftn(x)

    if sample_tensor is None:
        return _fftn_fn

    # Try to trace
    try:
        traced = torch.jit.trace(_fftn_fn, sample_tensor)
        return traced
    except Exception:
        # tracing failed, return python callable
        return _fftn_fn
