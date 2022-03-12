# Part of the spatiotemporal package for python
# Copyright 2022 Max Shinn <m.shinn@ucl.ac.uk>
# Available under the MIT license
import numpy as np
import pandas
import scipy.spatial
import scipy.optimize

def spatial_autocorrelation(cm, dist, discretization=1):
    """Calculate the SA-λ and SA-∞ measures of spatial autocorrelation, defined in [Shinn et al (2022)](https://www.biorxiv.org/content/10.1101/2021.06.01.446561v1)

    Args:
      cm (NxN numpy array): NxN correlation matrix of timeseries, where N is the number of
          timeseries
      dist (NxN numpy array): the NxN distance matrix, representing the spatial distance
          between location of each of the timeseries.  This should usually be the
          output of the `distance_matrix_euclidean` function.
      discretization (int): The size of the bins to use when computing the SA parameters.
          The size of the discretization should ensure that there are a sufficient number of
          observations in each bin, but also enough total bins to make a meaningful estimation.
          Try increasing it or decreasing it according to the scale of your data.  Data that has values
          up to around 100 should be fine with the default.  Decrease or increase as necessary
          to get an appropriate estimation.

    Returns:
      tuple of (SA-λ, SA-∞)
    """
    cm_flat = cm.flatten()
    dist_flat = dist.flatten()
    df = pandas.DataFrame(np.asarray([dist_flat, cm_flat]).T, columns=["dist", "corr"])
    df['dist_bin'] = np.round(df['dist']/discretization)*discretization
    df_binned = df.groupby('dist_bin').mean().reset_index().sort_values('dist_bin')
    binned_dist_flat = df_binned['dist_bin']
    binned_cm_flat = df_binned['corr']
    binned_dist_flat[0] = 1
    spatialfunc = lambda v : np.exp(-binned_dist_flat/v[0])*(1-v[1])+v[1]
    with np.errstate(all='warn'):
        res = scipy.optimize.minimize(lambda v : np.sum((binned_cm_flat-spatialfunc(v))**2), [10, .3], bounds=[(.1, 100), (-1, 1)])
    return (res.x[0], res.x[1])

def temporal_autocorrelation(x):
    """Compute TA-Δ₁ (lag-1 temporal autocorrelation) from the timeseries.

    Args:
      x: the timeseries of which to compute TA-Δ₁

        If `x` is a single list or one-dimensional numpy array, return the
        TA-Δ₁ estimate.  If `x` contains nested lists or is a NxT numpy, return
        a numpy array of length N, giving the TA-Δ₁ of each row of x.

    Returns:
      list of floats: The temporal autocorrelation of each timeseries in
          `x` is multidimensional, return nested lists in the same shape as the
           leading dimensions of `x`.

    Note:
        This is the biased estimator, but it is the default in numpy.  This
        is what we use throughout the manuscript.  The purpose of this function
        is to standardize computing TA-Δ₁.

    """
    if isinstance(x[0], (list, np.ndarray)):
        return np.asarray([temporal_autocorrelation(xe) for xe in x])
    return np.corrcoef(x[0:-1], x[1:])[0,1]

def long_memory(x, minscale, multivariate=False):
    """Estimate the long memory coefficient, from [Achard and Gannaz (2006)](https://doi.org/10.1111/jtsa.12170)

    See [Achard and Gannaz (2006)](https://doi.org/10.1111/jtsa.12170) for
    details of the coefficient and its estimator.

    Args:
      x (NxT numpy array): the matrix of timeseries (rows are regions, columns are timepoints)
      minscale (int): the minimum wavelet scale used to perform the estimation.
          As a rule of thumb, if data are low pass filtered, minscale should be the
          multiple of nyquist corresponding to the filter frequency, e.g. 2 if
          filtering is performed at half nyquist.
      multivariate: the type of estimation to perform, described in [Achard and Gannaz (2006)](https://doi.org/10.1111/jtsa.12170).
          Note that multivariate=True is extremely slow for any
          reasonably sized correlation matrix.

    Returns:
      float: The long memory coefficient

    Warning:
        This function is rather fragile: it requires rpy2 to be installed, as well
        as R, with the multiwave package.  It works on my computer but your results
        may vary.  If it doesn't work for you, export your data and use the
        "multiwave" package directly in R.

    """
    try:
        import rpy2.robjects.packages
        import rpy2.robjects.numpy2ri
    except ImportError:
        print("Rpy2 not available, long memory coefficient estimation not available.  "
              "Try using the 'multiwave' package in R instead.")
    x = x.transpose()
    rpy2.robjects.numpy2ri.activate()
    try:
        multiwave = rpy2.robjects.packages.importr('multiwave')
    except Exception as e:
        print("Please install the multiwave package in R")
        raise e
    filt = multiwave.scaling_filter("Daubechies", 8).rx2('h')
    if multivariate:
        res = list(multiwave.mww(x, filt, np.asarray([minscale,11])).rx2('d'))
    else:
        res = [multiwave.mww(x[:,i], filt, np.asarray([minscale,11])).rx2('d')[0] for i in range(0, x.shape[1])]
    rpy2.robjects.numpy2ri.deactivate()
    return res

