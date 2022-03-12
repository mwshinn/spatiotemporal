# Part of the spatiotemporal package for python
# Copyright 2022 Max Shinn <m.shinn@ucl.ac.uk>
# Available under the MIT license
import numpy as np
from .tools import get_eigenvalues, make_perfectly_symmetric
import scipy.stats

def eigensurrogate_matrix(cm, seed=None):
    """Eigensurrogate model, from [Shinn et al (2022)](https://www.biorxiv.org/content/10.1101/2021.06.01.446561v1)

    Determine the eigenvalues of the correlation matrix, and then sample a new
    correlation matrix with the same eigenvalues.

    Args:
      cm (NxN numpy array): a correlation matrix
      seed (int, optional): the random seed.  If not specified, it will use
          the current state of the numpy random number generator.

    Returns:
      NxN numpy array: a correlation matrix with the same eigenvalues as `cm`
    """
    desired_evs = get_eigenvalues(cm)
    rng = np.random.RandomState(seed)
    if min(desired_evs) < 0:
        desired_evs = np.maximum(0, desired_evs)
        newsum = np.sum(desired_evs)
        desired_evs[0] -= (newsum - len(desired_evs))
        print(f"Warning: eigenvalues were less than zero in source matrix by {newsum-len(desired_evs)}")
    m = scipy.stats.random_correlation.rvs(eigs=desired_evs, tol=1e-12, random_state=rng)
    np.fill_diagonal(m, 1)
    return make_perfectly_symmetric(m)

def eigensurrogate_timeseries(cm, N_timepoints, seed=None):
    """Timeseries from the eigensurrogate model.

    Sample timeseries which have the correlation matrix given by the
    eigensurrogate model.  Note that there are many ways to sample timeseries
    from the eigensurrogate model, but this is the simplest (a multivariate
    normal distribution).

    Args:
      cm (NxN numpy array): a correlation matrix
      N_timepoints (int): the length of the timeseries to sample
      seed (int, optional): the random seed.  If not specified, it will use
          the current state of the numpy random number generator.

    Returns:
      NxN_timepoints numpy array: timeseries generated from the eigensurrogate model

    """
    surrogate = eigensurrogate_matrix(cm)
    N_regions = cm.shape[0]
    rng = np.random.RandomState(seed)
    msqrt = scipy.linalg.sqrtm(surrogate)
    return msqrt @ rng.randn(N_regions, N_timepoints)

def phase_randomize(tss, seed=None):
    """Phase-randomized surrogate timeseries.

    Scramble a set of timeseries independently by preserving the amplitudes in
    Fouries space but randomly sampling new phases from the uniform distribution [0, 2π].

    Args:
      tss (NxT numpy array): should be a NxT matrix, where N is the number of timeseries and T is
          the number of samples in the timeseries.
      seed (int, optional): the random seed.  If not specified, it will use
          the current state of the numpy random number generator.

    Returns:
        NxT numpy array: surrogate timeseries of the same shape as `tss`
    """
    surrogates = np.fft.rfft(tss, axis=1)
    (N, n_time) = tss.shape
    len_phase = surrogates.shape[1]
    # Generate random phases uniformly distributed in the
    # interval [0, 2*Pi]
    phases = np.random.RandomState(seed).uniform(low=0, high=2 * np.pi, size=(N, len_phase))
    # Add random phases uniformly distributed in the interval [0, 2*Pi]
    surrogates *= np.exp(1j * phases)
    # Calculate IFFT and take the real part, the remaining imaginary
    # part is due to numerical errors.
    return np.real(np.fft.irfft(surrogates, n=n_time, axis=1))

def zalesky_surrogate(cm, seed=None):
    """Zalesky matching surrogate, from [Zalesky et al (2012)](https://doi.org/10.1016/j.neuroimage.2012.02.001)

    Generate matrices with identical mean-FC and var-FC.  Adapted from code
    taken from [Zalesky et al (2012)](https://doi.org/10.1016/j.neuroimage.2012.02.001)

    Args:
      cm (NxN numpy array): a correlation matrix
      seed (int, optional): the random seed.  If not specified, it will use
          the current state of the numpy random number generator.

    Returns:
      NxN numpy array: a correlation matrix with the same mean and variance as `cm`

    """
    N_regions = cm.shape[0]
    tri = np.triu_indices(N_regions, 1)
    rng = np.random.RandomState(seed)
    desired_mean = np.mean(cm[tri])
    desired_var = np.var(cm[tri])
    def fitmean(mu, n):
        """
        n = number of timepoints
        """
        x = rng.randn(N_regions, n) # Each ROW is a different region.  This is inconsistent with the paper.
        y = rng.randn(n, 1)
        amax = 10
        amin = 0
        while np.abs(amax-amin) > .001:
            a = amin + (amax-amin)/2
            rho = np.corrcoef(x+a*(y@np.ones((1, N_regions))).T)
            assert rho.shape[0] == N_regions, "Bad shape"
            muhat = np.mean(rho[tri])
            if muhat > desired_mean:
                amax = a
            else:
                amin = a
        return rho
    nmax = 1000
    nmin = 2
    while nmax - nmin > 1:
        n = int(np.floor(nmin + (nmax-nmin)/2))
        rho = fitmean(desired_mean, n)
        muhat = np.mean(rho[tri])
        sigma2hat = np.var(rho[tri])
        if sigma2hat > desired_var:
            nmin = n
        else:
            nmax = n
    return make_perfectly_symmetric(rho)
