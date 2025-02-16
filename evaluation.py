import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import wasserstein_distance

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
