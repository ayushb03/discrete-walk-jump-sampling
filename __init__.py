"""
Antibody Generation Package

This package provides tools for generating and evaluating antibody sequences using
Energy-Based Models and Denoising Networks.

Modules:
    - config: Configuration management
    - data_processing: Dataset handling and preprocessing
    - models: Neural network architectures
    - sampling: MCMC sampling methods
    - training: Training loops
    - evaluation: Evaluation metrics
    - main: Main execution script
"""

from .config import Config
from .data_processing import AntibodyDataset
from .models import EBM, DenoisingNetwork
from .sampling import langevin_mcmc_sampling, discrete_walk_jump_sampling
from .training import train_ebm, train_denoiser
from .evaluation import (
    estimate_critical_noise_level,
    distributional_conformity_score,
    internal_diversity
)

__all__ = [
    'Config',
    'AntibodyDataset',
    'EBM',
    'DenoisingNetwork',
    'langevin_mcmc_sampling',
    'discrete_walk_jump_sampling',
    'train_ebm',
    'train_denoiser',
    'estimate_critical_noise_level',
    'distributional_conformity_score',
    'internal_diversity'
] 