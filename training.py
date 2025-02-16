import torch
import torch.nn as nn
from sampling import langevin_mcmc_sampling

def train_ebm(ebm, dataloader, optimizer, epochs, noise_std):
    """
    Train the Energy-Based Model.
    
    Args:
        ebm: Energy-Based Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        epochs: Number of training epochs
        noise_std: Standard deviation of Gaussian noise
    """
    ebm.train()
    for epoch in range(epochs):
        for batch in dataloader:
            noisy_batch = batch + torch.randn_like(batch) * noise_std
            noisy_batch.requires_grad_(True)
            # Positive phase: Energy of noisy training data
            pos_energy = ebm(noisy_batch).mean()
            # Negative phase: Energy of samples from the model
            neg_samples = langevin_mcmc_sampling(ebm, noisy_batch, steps=10, step_size=0.01, noise_std=noise_std)
            neg_energy = ebm(neg_samples).mean()
            # Loss: Contrastive Divergence
            loss = pos_energy - neg_energy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


def train_denoiser(denoiser, dataloader, optimizer, epochs, noise_std):
    """
    Train the Denoising Network.
    
    Args:
        denoiser: Denoising model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        epochs: Number of training epochs
        noise_std: Standard deviation of Gaussian noise to add to training data
    """
    denoiser.train()
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for batch in dataloader:
            noisy_batch = batch + torch.randn_like(batch) * noise_std
            denoised_batch = denoiser(noisy_batch)
            loss = criterion(denoised_batch, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")