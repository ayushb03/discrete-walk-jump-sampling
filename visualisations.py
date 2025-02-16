import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

def plot_energy_landscape(ebm, seq_len=297, vocab_size=21):
    # Create a grid of samples
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Evaluate energy on grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            sample = torch.zeros(1, seq_len, vocab_size)
            sample[:, :, 0] = X[i,j]
            sample[:, :, 1] = Y[i,j]
            Z[i,j] = ebm(sample).item()
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Sequence Feature 1')
    ax.set_ylabel('Sequence Feature 2')
    ax.set_zlabel('Energy')
    ax.set_title('Energy Landscape of Antibody Sequences')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('figures/energy_landscape.png')
    plt.close()

def plot_training_progress(ebm_losses, denoiser_losses):
    # Create DataFrame
    data = pd.DataFrame({
        'Epoch': list(range(len(ebm_losses))) + list(range(len(denoiser_losses))),
        'Loss': ebm_losses + denoiser_losses,
        'Model': ['EBM']*len(ebm_losses) + ['Denoiser']*len(denoiser_losses)
    })
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Epoch', y='Loss', hue='Model', style='Model', 
                 markers=True, dashes=False, palette='Set2')
    plt.title('Training Progress of EBM and Denoiser')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/training_progress.png')
    plt.close()

def plot_sequence_diversity(real_samples, generated_samples):
    # Convert real samples (strings) to one-hot encoding
    vocab = "ACDEFGHIKLMNPQRSTVWY-"
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    
    def sequence_to_one_hot(seq):
        one_hot = np.zeros((len(seq), len(vocab)))
        for i, char in enumerate(seq):
            idx = char_to_idx.get(char, len(vocab)-1)
            one_hot[i, idx] = 1
        return one_hot
    
    # Convert samples to one-hot
    real_one_hot = np.array([sequence_to_one_hot(s) for s in real_samples])
    generated_one_hot = np.zeros((len(generated_samples), generated_samples.shape[1], len(vocab)))
    for i, sample in enumerate(generated_samples):
        for j, idx in enumerate(sample):
            generated_one_hot[i, j, idx] = 1
    
    # Pad/truncate to match lengths
    max_len = max(real_one_hot.shape[1], generated_one_hot.shape[1])
    if real_one_hot.shape[1] < max_len:
        padding = np.zeros((real_one_hot.shape[0], max_len - real_one_hot.shape[1], real_one_hot.shape[2]))
        real_one_hot = np.concatenate([real_one_hot, padding], axis=1)
    elif generated_one_hot.shape[1] < max_len:
        padding = np.zeros((generated_one_hot.shape[0], max_len - generated_one_hot.shape[1], generated_one_hot.shape[2]))
        generated_one_hot = np.concatenate([generated_one_hot, padding], axis=1)
    
    # Combine and reduce dimensions
    combined = np.vstack([real_one_hot.reshape(len(real_one_hot), -1), 
                         generated_one_hot.reshape(len(generated_one_hot), -1)])
    
    # Dimensionality reduction
    max_components = min(50, combined.shape[0], combined.shape[1])
    pca = PCA(n_components=max_components)
    pca_result = pca.fit_transform(combined)
    
    perplexity = min(30, pca_result.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(pca_result)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_result[:len(real_samples), 0], tsne_result[:len(real_samples), 1], 
                label='Real Sequences', alpha=0.6)
    plt.scatter(tsne_result[len(real_samples):, 0], tsne_result[len(real_samples):, 1], 
                label='Generated Sequences', alpha=0.6)
    plt.title('Sequence Diversity Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/sequence_diversity.png')
    plt.close()

def plot_property_distribution(real_samples, generated_samples):
    # Convert generated samples from indices to sequences
    vocab = "ACDEFGHIKLMNPQRSTVWY-"
    generated_sequences = []
    for sample in generated_samples:
        # Remove padding characters ('-') before calculating length
        sequence = ''.join([vocab[idx] for idx in sample if idx != 20])  # 20 is the index for '-'
        generated_sequences.append(sequence)
    
    # Calculate sequence properties
    real_lengths = [len(s.rstrip('-')) for s in real_samples]  # Remove padding from real samples
    gen_lengths = [len(s) for s in generated_sequences]
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.kdeplot(real_lengths, label='Real Sequences', color='blue', fill=True, alpha=0.3, warn_singular=False)
    sns.kdeplot(gen_lengths, label='Generated Sequences', color='red', fill=True, alpha=0.3, warn_singular=False)
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/sequence_length_distribution.png')
    plt.close()

def plot_quality_metrics(dcs_scores, diversity_scores):
    # Create DataFrame
    data = pd.DataFrame({
        'Epoch': range(len(dcs_scores)),
        'Distributional Conformity': dcs_scores,
        'Internal Diversity': diversity_scores
    })
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data['Epoch'], data['Distributional Conformity'], 
             label='Distributional Conformity', marker='o', color='green')
    plt.plot(data['Epoch'], data['Internal Diversity'], 
             label='Internal Diversity', marker='s', color='purple')
    plt.title('Sampling Quality Metrics Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/quality_metrics.png')
    plt.close()