import torch
import torch.nn as nn
from typing import Tuple

class EBM(nn.Module):
    """
    Energy-Based Model for antibody sequence generation.
    
    Args:
        input_dim: Input dimension size
        hidden_dim: Hidden dimension size (default: 128)
        vocab_size: Vocabulary size (default: 21)
        seq_len: Sequence length (default: 297)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, vocab_size: int = 21, seq_len: int = 297) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EBM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, vocab_size)
            
        Returns:
            Energy values of shape (batch_size, 1)
        """
        batch_size, seq_len, vocab_size = x.shape
        x = self.embedding(x) + self.positional_encoding
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1).reshape(batch_size, -1)  # Flatten
        return self.mlp(x)

class DenoisingNetwork(nn.Module):
    """
    Denoising Network for antibody sequence generation.
    
    Args:
        input_dim: Input dimension size
        hidden_dim: Hidden dimension size (default: 128)
        num_layers: Number of CNN layers (default: 3)
        vocab_size: Vocabulary size (default: 21)
        seq_len: Sequence length (default: 297)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, vocab_size: int = 21, seq_len: int = 297) -> None:
        super(DenoisingNetwork, self).__init__()
        self.embedding = nn.Linear(vocab_size, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Denoising Network.
        
        Args:
            y: Input tensor of shape (batch_size, seq_len, vocab_size)
            
        Returns:
            Denoised output of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len, vocab_size = y.shape
        embedded = self.embedding(y) + self.positional_encoding
        h = embedded.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)
        for cnn in self.cnn_layers:
            h = torch.relu(cnn(h))
        h = h.permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)
        return self.output_layer(h)