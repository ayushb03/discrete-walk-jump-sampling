import torch
import numpy as np
from typing import Tuple
from torch import Tensor
from models import EBM, DenoisingNetwork

# ================================
# 4. Discrete Walk-Jump Sampling (dWJS)
# ================================
def langevin_mcmc_sampling(ebm: EBM, initial_samples: Tensor, steps: int, step_size: float, noise_std: float) -> Tensor:
    """
    Perform Langevin MCMC sampling.
    
    Args:
        ebm: Energy-Based Model
        initial_samples: Initial samples
        steps: Number of MCMC steps
        step_size: Step size for Langevin dynamics
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        Samples from the model distribution
    """
    samples = initial_samples.clone()
    samples.requires_grad_(True)
    for _ in range(steps):
        noise = torch.randn_like(samples) * noise_std
        energy_grad = torch.autograd.grad(ebm(samples).sum(), samples)[0]
        samples = samples - step_size * energy_grad + np.sqrt(2 * step_size) * noise
        samples = samples.detach()
        samples.requires_grad_(True)
    return samples.detach()


def discrete_walk_jump_sampling(ebm: EBM, denoiser: DenoisingNetwork, initial_noise: Tensor, 
                               mcmc_steps: int, step_size: float, noise_std: float) -> Tensor:
    """
    Perform Discrete Walk-Jump Sampling.
    
    Args:
        ebm: Trained Energy-Based Model
        denoiser: Trained Denoising Network
        initial_noise: Initial noisy samples
        mcmc_steps: Number of Langevin MCMC steps
        step_size: Step size for Langevin dynamics
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        Clean samples after denoising
    """
    # Step 1: Walk (Langevin MCMC sampling)
    noisy_samples = langevin_mcmc_sampling(ebm, initial_noise, mcmc_steps, step_size, noise_std)
    # Step 2: Jump (Denoising)
    clean_samples = denoiser(noisy_samples)
    # Convert to one-hot encoding
    clean_samples = torch.argmax(clean_samples, dim=-1)
    return clean_samples