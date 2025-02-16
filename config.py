class Config:
    # Data parameters
    DATA = {
        'max_len': 297,               # Maximum sequence length
        'vocab_size': 21,             # Vocabulary size (20 amino acids + gap)
        'batch_size': 32,             # Batch size for training
        'num_samples': 10,          # Number of samples in dataset
        'random_seed': 42,            # Random seed for reproducibility
    }

    # Model architecture parameters
    MODEL = {
        'hidden_dim': 128,            # Hidden dimension size
        'num_layers': 3,              # Number of CNN layers
        'cnn_kernel_size': 3,         # CNN kernel size
        'cnn_padding': 1,             # CNN padding
    }

    # Training parameters
    TRAINING = {
        'ebm_lr': 1e-3,               # Learning rate for EBM
        'denoiser_lr': 1e-3,          # Learning rate for Denoiser
        'ebm_epochs': 2,             # Number of EBM training epochs
        'denoiser_epochs': 5,        # Number of Denoiser training epochs
        'noise_std': None,            # Noise standard deviation (None = auto)
        'mcmc_steps': 10,             # MCMC steps during training
        'mcmc_step_size': 0.01,       # MCMC step size during training
    }

    # Sampling parameters
    SAMPLING = {
        'num_samples': 10,            # Number of samples to generate
        'mcmc_steps': 100,            # MCMC steps during sampling
        'mcmc_step_size': 0.01,       # MCMC step size during sampling
        'sampling_noise_std': None,   # Noise std for sampling (None = auto)
    }

    # Evaluation parameters
    EVALUATION = {
        'use_property_model': False,  # Whether to use a property model for evaluation
        'num_reference_samples': 100, # Number of reference samples for evaluation
    }

    def __init__(self, **kwargs):
        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"No such config section: {key}")

# Example of custom configuration
custom_config = Config(
    DATA={'num_samples': 5000, 'batch_size': 64},
    TRAINING={'ebm_epochs': 20, 'denoiser_epochs': 20},
    SAMPLING={'num_samples': 100}
) 