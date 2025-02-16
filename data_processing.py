import torch
from torch.utils.data import Dataset
import numpy as np

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