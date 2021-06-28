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

def get_density_mixture_of_2DIndependentGaussians(x, y, means, variances):
    # means, variances \in N x 2
    mog_density = np.zeros((len(y), len(x)), dtype=np.double)
    scales = np.sqrt(variances)
    for i in range(means.shape[0]):
        x_gaussian = scipy.stats.norm.pdf(x, loc=means[i,0], scale=scales[i,0])
        y_gaussian = scipy.stats.norm.pdf(y, loc=means[i,1], scale=scales[i,1])
        mog_density += np.outer(y_gaussian, x_gaussian)
    return mog_density
