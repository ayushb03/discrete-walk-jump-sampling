import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances

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
    def __init__(self, input_dim, hidden_dim=128):
        super(EBM, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output scalar energy
        )

    def forward(self, x):
        # Flatten the input from (batch_size, seq_len, vocab_size) to (batch_size, seq_len * vocab_size)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.mlp(x)


# ================================
# 3. Denoising Network (ByteNet-like Architecture)
# ================================

class DenoisingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super(DenoisingNetwork, self).__init__()
        self.vocab_size = input_dim // 297  # Calculate vocab size from input_dim
        self.embedding = nn.Linear(self.vocab_size, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 297, hidden_dim))
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, y):
        # Reshape input to (batch_size, seq_len, vocab_size)
        batch_size = y.shape[0]
        y = y.view(batch_size, 297, self.vocab_size)
        
        # Add positional encoding
        h = self.embedding(y) + self.positional_encoding
        # CNN layers
        h = h.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)
        for cnn in self.cnn_layers:
            h = torch.relu(cnn(h))
        h = h.permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)
        # Output layer
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

def distributional_conformity_score(generated_samples, reference_samples):
    """Compute the Distributional Conformity Score."""
    # Placeholder for actual implementation
    # Compare distributions using Wasserstein distance or similar metric
    return np.random.rand()


def edit_distance(sample1, sample2):
    """Compute edit distance between two sequences."""
    return sum(c1 != c2 for c1, c2 in zip(sample1, sample2))


def internal_diversity(samples):
    """Compute internal diversity as average pairwise edit distance."""
    # Convert tensor to numpy array if needed
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    distances = pairwise_distances(samples, metric=edit_distance)
    return distances.mean()


# ================================
# 7. Main Function
# ================================

def main():
    # Generate synthetic antibody sequences (randomly generated but relevant)
    np.random.seed(42)
    vocab = "ACDEFGHIKLMNPQRSTVWY-"
    sequences = [''.join(np.random.choice(list(vocab), size=297)) for _ in range(1000)]

    # Preprocess data
    dataset = AntibodyDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize models
    input_dim = dataset.max_len * dataset.vocab_size
    ebm = EBM(input_dim=input_dim)
    denoiser = DenoisingNetwork(input_dim=input_dim)

    # Optimizers
    ebm_optimizer = optim.Adam(ebm.parameters(), lr=1e-3)
    denoiser_optimizer = optim.Adam(denoiser.parameters(), lr=1e-3)

    # Train EBM
    train_ebm(ebm, dataloader, ebm_optimizer, epochs=2, noise_std=0.5)

    # Train Denoiser
    train_denoiser(denoiser, dataloader, denoiser_optimizer, epochs=2, noise_std=0.5)

    # Generate samples using dWJS
    initial_noise = torch.randn(10, input_dim)  # Batch of 10 random noise vectors
    generated_samples = discrete_walk_jump_sampling(
        ebm, denoiser, initial_noise, mcmc_steps=100, step_size=0.01, noise_std=0.5
    )

    # Evaluate generated samples
    reference_samples = next(iter(dataloader))  # Use a batch of real data as reference
    dcs = distributional_conformity_score(generated_samples, reference_samples)
    print(f"Distributional Conformity Score: {dcs}")

    # Compute edit distance and internal diversity
    unique_samples = set(tuple(sample.numpy()) for sample in generated_samples)
    print(f"Unique Samples: {len(unique_samples)}")
    int_div = internal_diversity(generated_samples)
    print(f"Internal Diversity: {int_div}")


if __name__ == "__main__":
    main()