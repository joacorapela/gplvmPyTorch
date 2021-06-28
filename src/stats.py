import torch
import numpy as np
import scipy.stats

def pca_reduce(X, latent_dim, center=True):
    if latent_dim > X.shape[1]:  # pragma: no cover
        raise ValueError("Cannot have more latent dimensions than observed")
    U, S, V = torch.pca_lowrank(A=X, q=latent_dim, center=center)
    X_centered = X-torch.mean(input=X, dim=0)
    X_reduced = torch.matmul(X_centered, V[:,:latent_dim])
    return X_reduced
