# Spatiotemporal modeling tools for Python

This package provides tools for modeling and analyzing spatial and temporal
autocorrelation in Python.  It is based on the methods from the paper [Spatial
and temporal autocorrelation weave human brain
networks](https://www.biorxiv.org/content/10.1101/2021.06.01.446561v1).
Included are methods to compute the following statistics:

- Compute TA-Δ<sub>1</sub> (i.e. first-order temporal autocorrelation)
- Compute SA-λ and SA-∞ (i.e. measurements of spatial autocorrelation)
- Lin's concordance
- Fingerprinting performance, from [Finn et al (2015)](https://www.nature.com/articles/nn.4135)

It will also generate surrogate timeseries for the following:

- Spatiotemporal model from [Shinn et al (2022)](https://www.biorxiv.org/content/10.1101/2021.06.01.446561v1)
- Noiseless spatiotemporal model from [Shinn et al (2022)](https://www.biorxiv.org/content/10.1101/2021.06.01.446561v1)
- Zalesky matching model from [Zalesky et al (2012)](https://www.sciencedirect.com/science/article/abs/pii/S1053811912001784)
- Eigensurrogate model from [Shinn et al (2022)](https://www.biorxiv.org/content/10.1101/2021.06.01.446561v1)
- Phase scramble null model

[See complete documentation](https://spatiotemporal.readthedocs.io)

## Installation

To install:

    pip install spatiotemporal

Otherwise, download the package and do:

    python setup.py install --user

System requirements are:

- Numpy
- Scipy
- Pandas

## Citation

If you use this package for a paper, please cite: [Shinn et al (2022)](https://www.biorxiv.org/content/10.1101/2021.06.01.446561v1)

## Contact

Please report bugs to <https://github.com/mwshinn/spatiotemporal/issues>.  This
includes any problems with the documentation.  Pull Requests for bugs are
greatly appreciated.

This package is actively maintained.  However, it is feature complete, so no new
features will not be added.  This is intended to be a supplement for the paper,
not a general purpose package for all aspects of spatiotemporal data analysis.

For all other questions or comments, contact m.shinn@ucl.ac.uk.
