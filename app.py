import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import wasserstein_distance
from config import Config

# ================================
# 1. Data Preprocessing
# ================================
class AntibodyDataset(Dataset):
    def __init__(self, sequences, max_len=297, vocab_size=21):
        """
        Args:
            sequences (list of str): List of antibody sequences.
            max_len (int): Maximum length of sequences after alignment.
            vocab_size (int): Vocabulary size (20 amino acids + gap token).
        """
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.sequences = sequences
        self.one_hot_sequences = self._preprocess_sequences()

    def _preprocess_sequences(self):
        """Convert sequences to one-hot encoding."""
        one_hot_sequences = []
        # Create a mapping from characters to indices
        vocab = "ACDEFGHIKLMNPQRSTVWY-"
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}

        for seq in self.sequences:
            # Pad or truncate sequence to max_len
            seq = seq[:self.max_len].ljust(self.max_len, '-')
            # One-hot encode
            one_hot = np.zeros((self.max_len, self.vocab_size))
            for i, char in enumerate(seq):
                idx = char_to_idx.get(char, 20)  # Use 20 (gap token) for unknown characters
                one_hot[i, idx] = 1
            one_hot_sequences.append(one_hot)
        return np.array(one_hot_sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.one_hot_sequences[idx], dtype=torch.float32)


# ================================
# 2. Energy-Based Model (EBM)
# ================================
class EBM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, vocab_size=21, seq_len=297):
        super(EBM, self).__init__()
        self.embedding = nn.Linear(vocab_size, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        batch_size, seq_len, vocab_size = x.shape
        x = self.embedding(x) + self.positional_encoding
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1).reshape(batch_size, -1)  # Flatten
        return self.mlp(x)


# ================================
# 3. Denoising Network (ByteNet-like Architecture)
# ================================
class DenoisingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, vocab_size=21, seq_len=297):
        super(DenoisingNetwork, self).__init__()
        self.embedding = nn.Linear(vocab_size, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, y):
        batch_size, seq_len, vocab_size = y.shape
        embedded = self.embedding(y) + self.positional_encoding
        h = embedded.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)
        for cnn in self.cnn_layers:
            h = torch.relu(cnn(h))
        h = h.permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)
        return self.output_layer(h)


# ================================
# 4. Discrete Walk-Jump Sampling (dWJS)
# ================================
def langevin_mcmc_sampling(ebm, initial_samples, steps, step_size, noise_std):
    """Perform Langevin MCMC sampling."""
    samples = initial_samples.clone()
    samples.requires_grad_(True)
    for _ in range(steps):
        noise = torch.randn_like(samples) * noise_std
        energy_grad = torch.autograd.grad(ebm(samples).sum(), samples)[0]
        samples = samples - step_size * energy_grad + np.sqrt(2 * step_size) * noise
        samples = samples.detach()
        samples.requires_grad_(True)
    return samples.detach()


def discrete_walk_jump_sampling(ebm, denoiser, initial_noise, mcmc_steps, step_size, noise_std):
    """
    Perform Discrete Walk-Jump Sampling.
    Args:
        ebm: Trained Energy-Based Model.
        denoiser: Trained Denoising Network.
        initial_noise: Initial noisy samples.
        mcmc_steps: Number of Langevin MCMC steps.
        step_size: Step size for Langevin dynamics.
        noise_std: Standard deviation of Gaussian noise.
    Returns:
        Clean samples after denoising.
    """
    # Step 1: Walk (Langevin MCMC sampling)
    noisy_samples = langevin_mcmc_sampling(ebm, initial_noise, mcmc_steps, step_size, noise_std)
    # Step 2: Jump (Denoising)
    clean_samples = denoiser(noisy_samples)
    # Convert to one-hot encoding
    clean_samples = torch.argmax(clean_samples, dim=-1)
    return clean_samples


# ================================
# 5. Training Loop
# ================================
def train_ebm(ebm, dataloader, optimizer, epochs, noise_std):
    """Train the Energy-Based Model."""
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
    """Train the Denoising Network."""
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


# ================================
# 6. Evaluation Metrics
# ================================
def estimate_critical_noise_level(dataset):
    distances = pairwise_distances(dataset.one_hot_sequences.reshape(len(dataset), -1))
    φ = np.median(distances)
    ϖc = np.sqrt(φ / dataset.max_len)
    return ϖc


def distributional_conformity_score(generated_samples, reference_samples, property_model=None):
    """Calculate distributional conformity score.
    
    Args:
        generated_samples: Generated antibody sequences
        reference_samples: Reference antibody sequences
        property_model: Optional model to compute sequence properties
    """
    if property_model is None:
        # If no property model is provided, use a simple length-based property
        generated_props = [len(s) for s in generated_samples]
        reference_props = [len(s) for s in reference_samples]
    else:
        generated_props = property_model(generated_samples)
        reference_props = property_model(reference_samples)
    
    # Compute Wasserstein distance using scipy's implementation
    w_dist = wasserstein_distance(generated_props, reference_props)
    return 1 / (1 + w_dist)


def edit_distance(sample1, sample2):
    return sum(c1 != c2 for c1, c2 in zip(sample1, sample2))


def internal_diversity(samples):
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    distances = pairwise_distances(samples, metric=edit_distance)
    return distances.mean()


# ================================
# 7. Main Function
# ================================
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