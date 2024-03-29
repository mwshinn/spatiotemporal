# Part of the spatiotemporal package for python
# Copyright 2022 Max Shinn <m.shinn@ucl.ac.uk>
# Available under the MIT license
import numpy as np
import scipy.signal
import scipy.optimize
from .tools import spatial_exponential_floor

def spatiotemporal_model_timeseries(distance_matrix, sa_lambda, sa_inf, ta_delta1s, num_timepoints, sample_rate, highpass_freq, seed=None):
    """Simulate the spatiotemporal model from [Shinn et al (2023)](https://www.nature.com/articles/s41593-023-01299-3)

    Args:
      distance_matrix (NxN numpy array): the NxN distance matrix, representing the spatial distance
          between location of each of the timeseries.  This should usually be the
          output of the `distance_matrix_euclidean` function.
      sa_lambda (float): the SA-λ parameter
      sa_inf (float): the SA-∞ parameter
      ta_delta1s (length-N list of floats): a list of TA-Δ₁ values, of length N, for generating the timeseries
      num_timepoints (int): length of timeseries to generate
      sample_rate (float): the spacing between timepoints (e.g. TR in fMRI)
      highpass_freq (float): if non-zero, apply a highpass filter above the
          given frequency.  A good default is 0.01 for fMRI timeseries.
      seed (int, optional): the random seed.  If not specified, it will use
          the current state of the numpy random number generator.

    Returns:
      NxT numpy array: For each of the N nodes, a timeseries of length T=`num_timepoints` according to the spatiotemporal model
    """
    assert num_timepoints % 2 == 0, "Must be even timeseries length"
    # Filtered brown noise spectrum
    spectrum = make_spectrum(tslen=num_timepoints, sample_rate=sample_rate, alpha=2, highpass_freq=highpass_freq)
    spectra = np.asarray([spectrum]*len(ta_delta1s))
    # Spatial autocorrelation matrix
    corr = spatial_exponential_floor(distance_matrix, sa_lambda, sa_inf)
    # Create spatially embedded timeseries with the given spectra
    tss = correlated_spectral_sampling(corr, spectra, seed=seed)
    # Compute the standard deviation of nosie we need to add to get the desired TA-delta1
    noises = [how_much_noise(spectrum, max(.001, ta_delta1)) for ta_delta1 in ta_delta1s]
    # Add noise to the timeseries
    rng = np.random.RandomState(seed)
    tss += rng.randn(tss.shape[0], tss.shape[1]) * np.asarray(noises).reshape(-1,1)
    return tss

def intrinsic_timescale_sa_model_timeseries(distance_matrix, sa_lambda, sa_inf, ta_delta1s, num_timepoints, sample_rate, highpass_freq, seed=0):
    """Simulate the intrinsic timescale + spatial autocorrelation model from [Shinn et al (2023)](https://www.nature.com/articles/s41593-023-01299-3)

    Args:
      distance_matrix (NxN numpy array): the NxN distance matrix, representing the spatial distance
          between location of each of the timeseries.  This should usually be the
          output of the `distance_matrix_euclidean` function.
      sa_lambda (float): the SA-λ parameter
      sa_inf (float): the SA-∞ parameter
      ta_delta1s (length-N list of floats): a list of TA-Δ₁ values, of length N, for generating the timeseries
      num_timepoints (int): length of timeseries to generate
      sample_rate (float): the spacing between timepoints (e.g. TR in fMRI)
      highpass_freq (float): if non-zero, apply a highpass filter above the
          given frequency.  A good default is 0.01 for fMRI timeseries.
      seed (int, optional): the random seed.  If not specified, it will use
          the current state of the numpy random number generator.

    Returns:
      NxT numpy array: For each of the N nodes, a timeseries of length T=`num_timepoints` according to the intrinsic timescale + spatial autocorrelation model
    """
    assert num_timepoints % 2 == 0, "Must be even timeseries length"
    # Determine the pink noise exponent alpha from the TA-delta1
    alphas = [ta_to_alpha_fast(sample_rate=sample_rate, tslen=num_timepoints, highpass_freq=highpass_freq, target_ta=max(0,ta_delta1)) for ta_delta1 in ta_delta1s]
    # Use these alpha values to construct desired frequency spectra
    spectra = np.asarray([make_spectrum(tslen=num_timepoints, sample_rate=sample_rate, alpha=alpha, highpass_freq=highpass_freq) for alpha in alphas])
    # Spatial autocorrelation matrix
    corr = spatial_exponential_floor(distance_matrix, sa_lambda, sa_inf)
    # Compute timeseries from desired correlation matrix and frequency spectra
    tss = correlated_spectral_sampling(cm=corr, spectra=spectra, seed=seed)
    return tss


def correlated_spectral_sampling(cm, spectra, seed=None):
    """Generate timeseries with given amplitude spectra and correlation matrices

    This implements Correlated Spectral Sampling, as described in [Shinn et al
    (2023)](https://www.nature.com/articles/s41593-023-01299-3).

    Args:
      cm (NxN numpy array): The correlation matrix
      spectra (Nxk numpy array): A list of Fourier spectra generated by [make_spectrum][spatiotemporal.models.make_spectrum]
          Each of the N spectra are associated with a row/column of `cm`.
      seed (int, optional): the random seed.  If not specified, it will use
          the current state of the numpy random number generator.

    Returns:
      NxT numpy array: N timeseries of length T.  Timeseries i will have a power spectrum given by `spectra[i]`,
          and will be correlated with the other timeseries with correlations cm[i].

    """
    N_regions = cm.shape[0]
    N_freqs = len(spectra[0])
    N_timepoints = (N_freqs-1)*2
    assert spectra.shape == (N_regions, N_freqs)
    sum_squares = np.sum(spectra**2, axis=1, keepdims=True)
    cosine_similarity = (spectra @ spectra.T)/np.sqrt(sum_squares @ sum_squares.T)
    covmat = cm / cosine_similarity
    if np.min(np.linalg.eigvalsh(covmat)) < -1e-8:
        raise PositiveSemidefiniteError("Correlation matrix is not possible with those spectra using this method!")
    randstate = np.random.RandomState(seed)
    rvs = randstate.multivariate_normal(np.zeros(N_regions), cov=covmat, size=N_freqs*2)
    reals = rvs[0:N_freqs].T * spectra
    ims = rvs[N_freqs:].T * spectra
    # Since the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    ims[:,-1] = 0
    # The DC component must be zero and real
    reals[:,0] = 0
    ims[:,0] = 0
    tss = np.fft.irfft(reals + 1J*ims, n=N_timepoints, axis=-1)
    return tss

def make_spectrum(tslen, sample_rate, alpha, highpass_freq):
    """Create a 1/f^alpha spectrum.

    Args:
      tslen (int): the length of the timeseries represented by the spectrum
      sample_rate (float): the spacing between timepoints (e.g. TR in fMRI)
      alpha (float): the pink noise exponent, between 0 and 2.
      highpass_freq (float): if non-zero, apply a highpass filter above the
          given frequency.  A good default is 0.01 for fMRI timeseries.

    Returns:
      length-k numpy array: Return the fourier spectrum (amplitude spectrum)
    """
    freqs = np.fft.rfftfreq(tslen, sample_rate)
    with np.errstate(all="warn"):
        spectrum = freqs**(-alpha/2)
    if highpass_freq > 0:
        butter = scipy.signal.iirfilter(ftype='butter', N=4, Wn=highpass_freq, btype='highpass', fs=1/sample_rate, output='ba')
        butterresp = scipy.signal.freqz(*butter, fs=1/sample_rate, worN=len(freqs), include_nyquist=True)
        assert np.all(np.isclose(freqs, butterresp[0]))
        spectrum = spectrum * np.abs(butterresp[1])
    spectrum[0] = 0
    return spectrum

def how_much_noise(spectrum, target_ta):
    """Determine the standard deviation of noise to add to achieve a target TA-Δ₁.

    This function answers the following question: If I generate timeseries with
    frequency spectrum (amplitude spectrum) `spectrum`, and then add white
    noise to the generated timeseries, what should the standard deviation of
    this white noise be if I want the timeseries to have the TA-Δ₁ coefficient
    `target_ta`?

    Args:
      spectrum (length-k numpy array): the power spectrum to generate from, e.g.,
          that generated from [make_spectrum][spatiotemporal.models.make_spectrum].
      target_ta (float): the desired
          TA-Δ₁.

    Returns:
      float: The standard deviation of white noise to add
    """

    N = len(spectrum)
    weightedsum = np.sum(spectrum[1:]**2*np.cos(np.pi*np.arange(1, N)/N))
    try:
        sigma = np.sqrt((weightedsum - np.sum(spectrum**2)*target_ta)/(target_ta*N**2))
    except FloatingPointError:
        sigma = 0
    return sigma

def ta_to_alpha(tslen, sample_rate, highpass_freq, target_ta):
    """Compute the (pink noise) alpha which would give, noiseless, the given TA-Δ₁.

    Generate timeseries with get_spectrum_ta, i.e. high pass filtered.

    Args:
      tslen (int): the length of the timeseries represented by the spectrum
      sample_rate (float): the spacing between timepoints (e.g. TR in fMRI)
      alpha (float): the pink noise exponent, between 0 and 2.
      highpass_freq (float): if non-zero, apply a highpass filter above the
          given frequency.  A good default is 0.01 for fMRI timeseries.

    Returns:
      float: a value of alpha such that the filtered pink noise with
          this exponent has TA-Δ₁ coefficient `target_ta`.
    """
    objfunc = lambda alpha : (get_spectrum_ta(make_spectrum(tslen, sample_rate, alpha[0], highpass_freq)) - target_ta)**2
    x = scipy.optimize.minimize(objfunc, 1.5, bounds=[(0, 2)])
    return float(x.x[0])

def get_spectrum_ta(spectrum):
    """Given a fourier spectrum, return the expected TA-Δ₁ of a timeseries generated with that spectrum.

    Args:
      spectrum (length-k numpy array): the power spectrum to generate from, e.g.,
          that generated from [make_spectrum][spatiotemporal.models.make_spectrum].

    Returns:
      float: the TA-Δ₁ value that would be expected if a timeseries had the given power spectrum and random phases.

    """
    N = len(spectrum)
    weightedsum = np.sum(spectrum**2*np.cos(np.pi*np.arange(0, N)/N))
    return weightedsum/np.sum(spectrum**2)

ta_to_alpha_cache = {}
def ta_to_alpha_fast(tslen, sample_rate, highpass_freq, target_ta):
    """Identical to `ta_to_alpha`, but discretize and cache to increase speed.

    See [ta_to_alpha][spatiotemporal.models.ta_to_alpha] for documentation.
    """
    global ta_to_alpha_cache
    taround = round(target_ta, 2)
    key = (tslen, sample_rate, highpass_freq, taround)
    if key in ta_to_alpha_cache.keys():
        return ta_to_alpha_cache[key]
    val = ta_to_alpha(*key)
    ta_to_alpha_cache[key] = val
    return val

def make_noisy_spectrum(tslen, sample_rate, alpha, highpass_freq, target_ar1):
    """Similar to make_spectrum, except adds white noise to the spectrum
    (i.e. uniform distribution).  Returns the fourier spectrum (amplitude
    spectrum).

    This also applies the same filter as make_spectrum.

    """
    noiseless_spectrum = make_spectrum(tslen, sample_rate, alpha, highpass_freq)
    N = len(noiseless_spectrum)
    noise = how_much_noise(noiseless_spectrum, target_ar1)
    noisy_spectrum = np.sqrt(noiseless_spectrum**2 + noise**2 * N)
    noisy_spectrum[0] = 0
    return noisy_spectrum

class PositiveSemidefiniteError(Exception):
    pass
