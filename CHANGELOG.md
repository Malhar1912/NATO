# Changelog

All notable changes to the NATO Optimizer package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-02

### Added
- **Kakeya Directional Penalty** (`kakeya_directional_penalty`): New penalty function that tracks gradient directions across training steps and penalizes high cosine similarity with previous gradients, encouraging exploration of diverse gradient directions
- Comprehensive documentation for all public APIs
- CHANGELOG.md for version tracking

### Improved
- Enhanced README.md with badges, detailed API reference, and usage examples
- Better organized package exports in `__init__.py`

## [0.1.0] - 2026-01-XX

### Added
- Initial release of NATO Optimizer
- `NATOOptimizer`: Core optimizer with learning rate scheduling
- `fourier_spectral_penalty`: Fourier Spectral Penalty (FSP) for regularization
- `low_pass_filter_gradients`: N-D FFT gradient filtering
- `adjust_learning_rate`: Learning rate adjustment utilities
