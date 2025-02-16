# Antibody Generation Package

## Overview

This package implements the Discrete Walk-Jump Sampling (dWJS) approach for antibody sequence generation, as described in the paper "Protein Discovery with Discrete Walk-Jump Sampling" (Frey et al., 2023). The method combines:

1. **Energy-Based Modeling**: Learns a smoothed energy function of antibody sequences
2. **Langevin MCMC Sampling**: Explores the smoothed data manifold
3. **Denoising Network**: Projects samples back to the true data manifold

Key advancements include:
- Single noise level training for both EBM and denoiser
- Contrastive divergence training for energy-based models
- One-step denoising for improved sample quality
- Robust evaluation metrics for protein generation

## Theoretical Background

The dWJS approach addresses challenges in discrete generative modeling by:

1. Learning a smoothed energy function E(x) through contrastive divergence
2. Sampling from the smoothed manifold using Langevin dynamics:
   x_{t+1} = x_t - η∇E(x_t) + √(2η)ε_t
3. Projecting samples back to the true manifold using a denoising network:
   x_clean = f_θ(x_noisy)

This combines the benefits of energy-based and score-based models while simplifying training and sampling.

--- 

This package provides a comprehensive framework for generating and evaluating antibody sequences using
Energy-Based Models (EBMs) and Denoising Networks. The system implements the Discrete Walk-Jump
Sampling (dWJS) approach for sequence generation.

## Key Features:
- **Energy-Based Modeling**: Learns the underlying distribution of antibody sequences
- **Denoising Network**: Cleans noisy sequences to generate high-quality samples
- **MCMC Sampling**: Implements Langevin dynamics for efficient sampling
- **Evaluation Metrics**: Provides comprehensive metrics for sequence quality assessment

## Modules:
1. `config.py`: Configuration management with default and custom settings
2. `data_processing.py`: Dataset handling and sequence preprocessing
3. `models.py`: Neural network architectures (EBM and Denoising Network)
4. `sampling.py`: MCMC sampling methods including dWJS
5. `training.py`: Training loops for both EBM and Denoising Network
6. `evaluation.py`: Metrics for sequence quality assessment
7. `main.py`: Main execution script
## Example Usage:

To use the package with custom configuration, add this to your main.py:
For more customization options, you can modify the following parameters in `config.py`:


``` python
from antibody_generation import Config, main

config = Config(
DATA={'num_samples': 10000, 'batch_size': 128},
TRAINING={'ebm_epochs': 50, 'denoiser_epochs': 50}
) ```

## Installation:
1. Clone the repository:
``` bash
git clone https://github.com/ayushb03/discrete-walk-jump-sampling.git
```
2. Install dependencies:
``` bash
uv pip install -r requirements.txt
```
3. Run the main script:
``` bash
uv run main.py
```