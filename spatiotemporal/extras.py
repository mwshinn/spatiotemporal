# Part of the spatiotemporal package for python
# Copyright 2022 Max Shinn <m.shinn@ucl.ac.uk>
# Available under the MIT license
import numpy as np

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
      subjects (list or 1xN numpy array) should be a list of length N giving the subject ID.  N should be
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

def lin(x, y):
    """Lin's concordance correlation coefficient

    Lin's concordnce ranges from -1 to 1 and achieves a tradeoff between
    correlation and variance explained.

    Args:
      x, y (lists or 1xN numpy arrays): the two vectors on which to find Lin's concordance

    Returns:
      float: The Lin's concordance of the two given vectors
    """
    return 2*np.cov(x, y, ddof=0)[0,1]/(np.var(x) + np.var(y) + (np.mean(x)-np.mean(y))**2)

