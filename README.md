# NATO

**NATO** — NATO Optimizer with Fourier Spectral Penalty (FSP) and generalized N-D FFT gradient filtering.

## Install (editable local)
```bash
pip install -e .
```

## Quick usage
```python
import torch
from nato_opt import NATOOptimizer, fourier_spectral_penalty, low_pass_filter_gradients

model = ...  # any torch.nn.Module
optimizer = NATOOptimizer(model.parameters(), lr=1e-3)

# training step
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
fsp = fourier_spectral_penalty(model, lambda_fsp=1e-6)
total = loss + fsp
total.backward()
low_pass_filter_gradients(model, cutoff_ratio=0.5)
optimizer.step(epoch)  # pass epoch if you want LR decay
```
