# NATO Optimizer

[![PyPI version](https://img.shields.io/pypi/v/nato-opt.svg)](https://pypi.org/project/nato-opt/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NATO** — NATO Optimizer with Fourier Spectral Penalty (FSP), Kakeya Directional Penalty, and generalized N-D FFT gradient filtering for PyTorch.

## Features

- 🚀 **NATOOptimizer**: Custom optimizer with built-in learning rate scheduling
- 📊 **Fourier Spectral Penalty (FSP)**: Regularization based on frequency-domain analysis
- 🎯 **Kakeya Directional Penalty**: Gradient direction consistency regularization
- 🔧 **Low-Pass Gradient Filtering**: N-D FFT-based gradient smoothing
- ⚡ **GPU Accelerated**: Full CUDA support for all operations

## Installation

### From PyPI
```bash
pip install nato-opt
```

### From Source (Editable)
```bash
git clone https://github.com/Malhar1912/NATO.git
cd NATO
pip install -e .
```

## Quick Start

```python
import torch
from nato_opt import (
    NATOOptimizer,
    fourier_spectral_penalty,
    low_pass_filter_gradients,
    kakeya_directional_penalty
)

model = ...  # any torch.nn.Module
optimizer = NATOOptimizer(model.parameters(), lr=1e-3)
kakeya_state = {}  # persistent state for Kakeya penalty

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Compute penalties
        fsp = fourier_spectral_penalty(model, lambda_fsp=1e-6)
        total_loss = loss + fsp
        total_loss.backward()
        
        # Apply Kakeya penalty (optional)
        k_penalty = kakeya_directional_penalty(model, kakeya_state, lambda_k=1e-4)
        
        # Apply gradient filtering before optimizer step
        low_pass_filter_gradients(model, cutoff_ratio=0.5)
        optimizer.step(epoch)  # pass epoch for LR decay
```

## API Reference

### NATOOptimizer

```python
NATOOptimizer(params, lr=1e-3, ...)
```

Custom optimizer extending SGD with built-in learning rate scheduling.

**Parameters:**
- `params`: Iterable of parameters to optimize
- `lr`: Learning rate (default: 1e-3)

---

### fourier_spectral_penalty

```python
fourier_spectral_penalty(
    model,
    lambda_fsp=1e-6,
    include_conv=True,
    include_linear=True,
    module_whitelist=None,
    module_blacklist=None,
    device=None
) -> torch.Tensor
```

Compute Fourier Spectral Penalty (FSP) on model weights.

**Parameters:**
- `model`: PyTorch model
- `lambda_fsp`: Penalty coefficient (default: 1e-6)
- `include_conv`: Include Conv layers (default: True)
- `include_linear`: Include Linear layers (default: True)
- `module_whitelist`: Only include these module names
- `module_blacklist`: Exclude these module names

**Returns:** Scalar penalty tensor

---

### kakeya_directional_penalty

```python
kakeya_directional_penalty(
    model,
    state,
    lambda_k=1e-4
) -> torch.Tensor
```

Compute Kakeya directional penalty based on gradient direction consistency.

Penalizes gradients that maintain high cosine similarity with previous gradients, encouraging exploration of diverse gradient directions during optimization.

**Parameters:**
- `model`: PyTorch model
- `state`: Persistent dict to store previous gradients `{param: {'prev_grad': tensor}}`
- `lambda_k`: Penalty coefficient (default: 1e-4)

**Returns:** Scalar penalty tensor

---

### low_pass_filter_gradients

```python
low_pass_filter_gradients(
    model,
    cutoff_ratio=0.5
)
```

Apply low-pass FFT filtering to gradients, smoothing high-frequency noise.

**Parameters:**
- `model`: PyTorch model with computed gradients
- `cutoff_ratio`: Frequency cutoff (0-1), lower = more filtering

---

### adjust_learning_rate

```python
adjust_learning_rate(optimizer, epoch, ...)
```

Utility function for learning rate scheduling.

## Requirements

- Python >= 3.8
- PyTorch
- NumPy

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- **Malhar Pangarkar** - [malharpangarkar19@gmail.com](mailto:malharpangarkar19@gmail.com)
- **Atharva Khambete** - [atharvakhambete1@gmail.com](mailto:atharvakhambete1@gmail.com)

## Citation

If you use NATO Optimizer in your research, please cite:

```bibtex
@software{nato_optimizer,
  title = {NATO Optimizer: Neural Adaptive Training Optimizer},
  author = {Pangarkar, Malhar and Khambete, Atharva},
  year = {2026},
  url = {https://github.com/Malhar1912/NATO}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
