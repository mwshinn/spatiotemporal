# Part of the spatiotemporal package for python
# Copyright 2022 Max Shinn <m.shinn@ucl.ac.uk>
# Available under the MIT license
import numpy as np
import scipy.linalg
import scipy.spatial

def get_eigenvalues(cm):
    """Find the eigenvalues of the correlation matrix `cm`.

    They will always be real and non-negative since correlation matrices are
    positive semidefinite
    """
    return scipy.linalg.eigvalsh(cm)

def make_perfectly_symmetric(cm):
    """Eliminate numerical errors in a correlation matrix."""
    return np.maximum(cm, cm.T)

def distance_matrix_euclidean(distances):
    """Returns a Euclidean distance matrix.

    `distances` should be a Nx3 numpy matrix, providing the xyz coordinates for
    N brain regions.

    """
    return scipy.spatial.distance.cdist(distances, distances)

def spatial_exponential_floor(distances, sa_lmbda, sa_inf):
    """Find a hypothetical spatial correlation matrix"""
    return np.exp(-distances/sa_lmbda)*(1-sa_inf)+sa_inf
