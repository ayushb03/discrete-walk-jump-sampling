from config import Config
from data_processing import AntibodyDataset
from models import EBM, DenoisingNetwork
from sampling import discrete_walk_jump_sampling
from training import train_ebm, train_denoiser
from evaluation import (estimate_critical_noise_level,
                       distributional_conformity_score,
                       internal_diversity)
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim

def main():
    # Initialize config
    config = Config()
    
    # Generate synthetic antibody sequences
    np.random.seed(config.DATA['random_seed'])
    vocab = "ACDEFGHIKLMNPQRSTVWY-"
    sequences = [''.join(np.random.choice(list(vocab), size=config.DATA['max_len'])) 
                 for _ in range(config.DATA['num_samples'])]
    
    # Preprocess data
    dataset = AntibodyDataset(sequences, max_len=config.DATA['max_len'], vocab_size=config.DATA['vocab_size'])
    dataloader = DataLoader(dataset, batch_size=config.DATA['batch_size'], shuffle=True)

    # Estimate critical noise level
    ϖc = estimate_critical_noise_level(dataset)
    noise_std = ϖc if config.TRAINING['noise_std'] is None else config.TRAINING['noise_std']
    config.TRAINING['noise_std'] = noise_std
    config.SAMPLING['sampling_noise_std'] = noise_std

    # Initialize models
    input_dim = config.DATA['max_len'] * config.DATA['vocab_size']
    ebm = EBM(input_dim=input_dim, 
              hidden_dim=config.MODEL['hidden_dim'],
              vocab_size=config.DATA['vocab_size'],
              seq_len=config.DATA['max_len'])
    
    denoiser = DenoisingNetwork(input_dim=input_dim,
                                hidden_dim=config.MODEL['hidden_dim'],
                                num_layers=config.MODEL['num_layers'],
                                vocab_size=config.DATA['vocab_size'],
                                seq_len=config.DATA['max_len'])

    # Optimizers
    ebm_optimizer = optim.Adam(ebm.parameters(), lr=config.TRAINING['ebm_lr'])
    denoiser_optimizer = optim.Adam(denoiser.parameters(), lr=config.TRAINING['denoiser_lr'])

    # Train models
    train_ebm(ebm, dataloader, ebm_optimizer, 
              epochs=config.TRAINING['ebm_epochs'], 
              noise_std=config.TRAINING['noise_std'])

    train_denoiser(denoiser, dataloader, denoiser_optimizer,
                   epochs=config.TRAINING['denoiser_epochs'],
                   noise_std=config.TRAINING['noise_std'])

    # Generate samples
    initial_noise = torch.randn(config.SAMPLING['num_samples'], 
                               config.DATA['max_len'], 
                               config.DATA['vocab_size'])
    
    generated_samples = discrete_walk_jump_sampling(
        ebm, denoiser, initial_noise,
        mcmc_steps=config.SAMPLING['mcmc_steps'],
        step_size=config.SAMPLING['mcmc_step_size'],
        noise_std=config.SAMPLING['sampling_noise_std']
    )

    # Evaluate generated samples
    reference_samples = next(iter(dataloader))  # Use a batch of real data as reference
    dcs = distributional_conformity_score(generated_samples, reference_samples)  # No property model needed
    print(f"Distributional Conformity Score: {dcs}")
    unique_samples = set(tuple(sample.numpy()) for sample in generated_samples)
    print(f"Unique Samples: {len(unique_samples)}")
    int_div = internal_diversity(generated_samples)
    print(f"Internal Diversity: {int_div}")

if __name__ == "__main__":
    main()