# This is mostly just a simple smoke test.  Would be better to have more
# elaborate testing at some point in the future.  That being said, the
# functions in this package were all pulled from a different package which had
# better testing, and they have all been scientifically validated.  So the main
# risk here is if some numpy or scipy function changes its behavior in a way
# that passes invisibly.

import numpy as np
import spatiotemporal as st
import scipy.stats

tss = [np.cumsum(np.random.RandomState(1000).randn(100,200), axis=1), # Brownian noise
       np.random.RandomState(999).randn(50, 400), # Gaussian noise
       ]

def test_phase_randomize():
    for i,ts in enumerate(tss):
        sur = st.phase_randomize(ts)
        fft = np.mean(np.abs(np.fft.fft(ts)), axis=0)[1:len(ts)//2]
        fft_sur = np.mean(np.abs(np.fft.fft(sur)), axis=0)[1:len(sur)//2]
        assert sur.shape == ts.shape
        assert np.max(np.abs(fft-fft_sur)) < .0001


def test_zalesky():
    for i,ts in enumerate(tss):
        cm = np.corrcoef(ts)
        cm_s = st.zalesky_surrogate(cm, seed=i)
        assert np.abs(np.log(np.mean(cm)/np.mean(cm_s))) < .1
        assert np.abs(np.log(np.mean(cm)/np.mean(cm_s))) < .25

def test_eigensurrogate():
    for i,ts in enumerate(tss):
        cm = np.corrcoef(ts)
        cm_s = st.eigensurrogate_matrix(cm, seed=i)
        ev_cm = st.tools.get_eigenvalues(cm)
        ev_cm_s = st.tools.get_eigenvalues(cm_s)
        assert np.max(np.abs(np.log(ev_cm/ev_cm_s))) < 1e-10

def test_eigensurrogate_timeseries():
    for i,ts in enumerate(tss):
        cm = np.corrcoef(ts)
        ts_s = st.eigensurrogate_timeseries(cm, N_timepoints=1000, seed=i)
        ev_cm = st.tools.get_eigenvalues(cm)
        ev_cm_s = st.tools.get_eigenvalues(np.corrcoef(ts_s))
        assert np.max(np.abs(np.log(ev_cm/ev_cm_s))) < .3

def test_spatial_autocorrelation():
    poss = np.random.rand(200,3)*100
    dists = st.tools.distance_matrix_euclidean(poss)
    for params in [(10, .2), (50, .01), (5, .5), (30, -.4)]:
        cm = st.tools.spatial_exponential_floor(dists, params[0], params[1])
        fitparams = np.asarray(st.spatial_autocorrelation(cm, dists))
        assert np.max(np.abs(np.log(fitparams/params))) < .1

def test_temporal_autocorrelation():
    # Test by going through alpha
    for i,target_ta in enumerate([0, .2, .4, .6, .8]):
        alpha = st.models.ta_to_alpha_fast(1000, 1, .01, target_ta)
        spec = st.models.make_spectrum(1000, 1, alpha, .01)
        ts = st.models.correlated_spectral_sampling(np.asarray([[1]]), np.asarray([spec]), seed=i)
        ta = st.temporal_autocorrelation(ts)
        assert np.abs(ta - target_ta) < .05

# We're not going to test long_memory because it requires such a complicated
# setup.

def test_fingerprint():
    assert st.fingerprint([1, 2, 2, 2], np.asarray([[4, 4, 4, 5], [4, 5, 4, 3], [4, 4, 4, 5], [4, 5, 4, 3]])) == .5
    assert st.fingerprint([1, 2, 2, 1], np.asarray([[4, 4, 4, 5], [4, 5, 4, 3], [4, 4, 4, 5], [4, 5, 4, 3]])) == 0
    assert st.fingerprint([1, 2, 1, 2], np.asarray([[4, 4, 4, 5], [4, 5, 4, 3], [4, 4, 4, 5], [4, 5, 4, 3]])) == 1
    assert st.fingerprint([1, 1, 1, 1], np.asarray([[4, 4, 4, 5], [4, 5, 4, 3], [4, 4, 4, 5], [4, 5, 4, 3]])) == 1

def test_lin():
    assert st.lin([1, 2, 3], [1, 2, 3]) == 1
    assert st.lin(np.asarray([1, 2, 3]), np.asarray([3, 2, 1])) == -1
    assert st.lin([5, 5, 5], [3, 5, 7]) == 0
    assert np.all(st.lin(np.asarray([[5,5,5],[1,2,3]]), np.asarray([[1, 2, 3], [3, 2, 1], [1, 2, 3]])) == np.asarray([[0, 0, 0], [1, -1, 1]]))
    assert 0 < st.lin([1, 2, 3], [12, 13, 14])  < st.lin([1, 2, 3], [2, 3, 4]) < 1

def test_cosine():
    assert st.cosine([1, 2, 3], [2, 3, 4]) < 1
    assert st.cosine([1, 2, 3], [1, 2, 3]) == 1
    assert st.cosine([1, 2, 3], [-1, -2, -3]) == -1

def test_pearson():
    assert st.pearson([1, 2, 3], [2, 3, 4]) == 1
    assert st.pearson(np.asarray([1, 2, 3]), np.asarray([10, 8, 6])) == -1
    assert st.pearson([5, 5, 5], [3, 5, 7]) == 0
    assert np.all(st.pearson(np.asarray([[5,5,5],[1,2,3]]), np.asarray([[1, 2, 3], [3, 2, 1], [11, 12, 13]])) == np.asarray([[0, 0, 0], [1, -1, 1]]))
    m = np.random.randn(100,200)
    assert np.all(np.isclose(np.corrcoef(m), st.pearson(m, m)))

def test_spearman():
    assert st.spearman([1, 2, 3], [2, 3, 4]) == 1
    assert st.spearman(np.asarray([1, 2, 3]), np.asarray([10, 8, 6])) == -1
    assert st.spearman([5, 5, 5], [3, 5, 7]) == 0
    assert np.all(st.spearman(np.asarray([[5,5,5],[1,2,3]]), np.asarray([[1, 2, 3], [3, 2, 1], [11, 12, 13]])) == np.asarray([[0, 0, 0], [1, -1, 1]]))
    assert st.spearman([1, 2, 3], [3, 5, 6]) > st.pearson([1, 2, 3], [3, 5, 6])
    m = np.random.randn(100,200)
    assert np.all(np.isclose(scipy.stats.spearmanr(m.T).correlation, st.spearman(m, m)))

def test_spatiotemporal_model():
    poss = np.random.rand(50,3)*100
    seed = 100
    distance_matrix = st.tools.distance_matrix_euclidean(poss)
    num_timepoints = 50000 # Big to get a better fit
    ta_delta1s = np.random.RandomState(seed).rand(50)*.8
    sample_rate = 1
    highpass_freq = .01
    seed = 1
    for i,params in enumerate([(20, .2), (50, .5), (10, 0)]):
        tss = st.spatiotemporal_model_timeseries(distance_matrix, params[0], params[1], ta_delta1s, num_timepoints, sample_rate, highpass_freq, seed=i+seed+1)
        newtas = st.temporal_autocorrelation(tss)
        assert np.max(np.abs(ta_delta1s-newtas)) < .05
        # Don't know how to test SA-lambda...
        assert (np.mean(np.corrcoef(tss)) - params[1])

def test_intrinsic_timescale_sa_model():
    poss = np.random.rand(50,3)*100
    seed = 200
    distance_matrix = st.tools.distance_matrix_euclidean(poss)
    num_timepoints = 50000 # Big to get a better fit
    ta_delta1s = np.random.RandomState(seed).rand(50)*.3
    sample_rate = 1
    highpass_freq = .01
    seed = 1
    for i,params in enumerate([(20, .05), (50, .1), (10, 0)]):
        tss = st.intrinsic_timescale_sa_model_timeseries(distance_matrix, params[0], params[1], ta_delta1s, num_timepoints, sample_rate, highpass_freq, seed=i+seed+1)
        newtas = st.temporal_autocorrelation(tss)
        assert np.max(np.abs(ta_delta1s-newtas)) < .05
        # Don't know how to test SA-lambda...
        assert (np.mean(np.corrcoef(tss)) - params[1])
