# Part of the spatiotemporal package for python
# Copyright 2022 Max Shinn <m.shinn@ucl.ac.uk>
# Available under the MIT license
import numpy as np
import scipy.stats

def fingerprint(subjects, values):
    """Fingerprinting performance, from [Finn et al (2015)](https://www.nature.com/articles/nn.4135).

    Note:
        This implementation is slightly different than that of Finn et al (2015).
        Here, instead of having separate databases, we use all other scans from all
        other subjects as the "database".  Then, we see if the match is from the
        same subject or different subjects.  This means that if there are more than
        two observations for each subject, there will be more than one "correct"
        best match.  However, it also means that there are many more possible
        incorrect matches for a given subject than there are in Finn et al (2015).

    Args:
      subjects (list or 1xN numpy array): a list of length N giving the subject ID.  N should be
          the total number of observations, e.g., if there are 10 subjects with 3
          scans each, N = 30.
      values (Nxk numpy array): numpy matrix, where N is as above and k is the size of
          the feature on which to perform the fingerprinting.  E.g., if we are
          fingerprinting based on TA-delta1 of each node in a 360 node atlas, k=360.

    Returns:
      float: The fraction of correct fingerprinting identifications
    """

    assert values.shape[0] == len(subjects)
    corrs = np.corrcoef(values)
    np.fill_diagonal(corrs, -1)
    maxcorrs = np.argmax(corrs, axis=0)
    best_match_subject = np.asarray(subjects)[maxcorrs]
    return np.mean(best_match_subject == subjects)

def _cross(x, y, func):
    """Utility function for correlation/covariance functions"""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        x = x[None]
    if y.ndim == 1:
        y = y[None]
    assert x.ndim == 2, "X must be two-dimensional"
    assert y.ndim == 2, "Y must be two-dimensional"
    assert x.shape[1] == y.shape[1], "X and Y must have compatible dimensions"
    res = func(x,y)
    res[np.isnan(res)] = 0
    if res.shape == (1,1):
        return res[0,0]
    return res


def lin(x, y):
    """Lin's concordance correlation coefficient

    Lin's concordnce ranges from -1 to 1 and achieves a tradeoff between
    correlation and variance explained.  This function returns a matrix of the
    concordance for row vectors.

    Args:
      x (list of length N or KxN numpy array): the first vector on which to find Lin's concordance
      y (list of length N or MxN numpy array): the second vector on which to find Lin's concordance

    Returns:
      KxM numpy array: The Lin's concordance matrix between x and y.  If K and M are 1 or x and y are lists, return a float instead.

    """
    _lin = lambda x,y : 2*(x-np.mean(x, axis=1, keepdims=True))@(y-np.mean(y, axis=1, keepdims=True)).T/x.shape[1]/(np.var(x, axis=1)[:,None] + np.var(y, axis=1)[None,:] + (np.mean(x, axis=1)[:,None]-np.mean(y, axis=1)[None,:])**2)
    return _cross(x,y,_lin)

def cosine(x, y):
    """Cosine similarity

    Cosine similarity ranges from -1 to 1 and measures the cosine of the angles
    between the vectors.  This function returns a matrix of the similarity for row vectors.

    Args:
      x (list of length N or KxN numpy array): the first vector on which to find cosine similarity
      y (list of length N or MxN numpy array): the second vector on which to find cosine similarity

    Returns:
      KxM numpy array: The cosine similarity matrix between x and y.  If K and M are 1 or x and y are lists, return a float instead.

    """
    _cosine = lambda x,y : x@y.T/np.sqrt(np.sum(x**2, axis=1)[:,None])/np.sqrt(np.sum(y**2, axis=1)[None,:])
    return _cross(x,y,_cosine)

def pearson(x, y):
    """Matrix of Pearson correlations

    Pearson correlation ranges from -1 to 1.  This function differs from
    np.corrcoef because it allows you to pass x and y as matrices without
    computing the correlation between within-matrix rows.  For large x and y
    this is a substantial speed increase.  This function returns a matrix of
    the correlation for row vectors.  (Sometimes this operation is mistakenly
    called "cross-correlation".)

    Args:
      x (list of length N or KxN numpy array): the first vector on which to find Pearson correlation
      y (list of length N or MxN numpy array): the second vector on which to find Pearson correlation

    Returns:
      KxM numpy array: The Pearson correlation matrix between x and y.  If K and M are 1 or x and y are lists, return a float instead.

    """
    _pearson = lambda x,y : (x-np.mean(x,axis=1,keepdims=True))@(y-np.mean(y,axis=1,keepdims=True)).T/x.shape[1]/np.sqrt(np.var(x, axis=1)[:,None])/np.sqrt(np.var(y, axis=1)[None,:])
    return _cross(x,y,_pearson)

def spearman(x, y):
    """Matrix of Spearman correlations

    Spearman correlation ranges from -1 to 1.  This function differs from
    scipy.stats.spearmanr because it allows you to pass x and y as matrices
    without computing the correlation between within-matrix rows.  For large x
    and y this is a substantial speed increase.  This function returns a matrix
    of the correlation for row vectors (unlike scipy.stats.spearmanr, which uses
    column vectors).

    Args:
      x (list of length N or KxN numpy array): the first vector on which to find Spearman correlation
      y (list of length N or MxN numpy array): the second vector on which to find Spearman correlation

    Returns:
      KxM numpy array: The Spearman correlation matrix between x and y.  If K and M are 1 or x and y are lists, return a float instead.

    """
    def _spearman(x,y):
        x = scipy.stats.rankdata(x, axis=1)
        y = scipy.stats.rankdata(y, axis=1)
        return (x-np.mean(x,axis=1,keepdims=True))@(y-np.mean(y,axis=1,keepdims=True)).T/x.shape[1]/np.sqrt(np.var(x, axis=1)[:,None])/np.sqrt(np.var(y, axis=1)[None,:])
    return _cross(x,y,_spearman)
