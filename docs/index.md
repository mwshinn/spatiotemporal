# Spatiotemporal modeling tools for Python

This package provides tools for modeling and analyzing spatial and temporal
autocorrelation in Python.  It is based on the methods from the paper [Spatial
and temporal autocorrelation weave human brain
networks](https://www.biorxiv.org/content/10.1101/2021.06.01.446561v1).
Included are methods to compute the following statistics:

- [Compute TA-Δ<sub>1</sub> (i.e. first-order temporal autocorrelation)][spatiotemporal.stats.temporal_autocorrelation]
- [Compute SA-λ and SA-∞ (i.e. measurements of spatial autocorrelation)][spatiotemporal.stats.spatial_autocorrelation]
- [Lin's concordance][spatiotemporal.extras.lin]
- [Fingerprinting performance][spatiotemporal.extras.fingerprint]

It will also generate surrogate timeseries for the following:

- [Spatiotemporal model][spatiotemporal.models.spatiotemporal_model_timeseries]
- [Intrinsic timescale + SA model][spatiotemporal.models.intrinsic_timescale_sa_model_timeseries]
- [Zalesky matching model][spatiotemporal.surrogates.zalesky_surrogate]
- [Eigensurrogate model][spatiotemporal.surrogates.eigensurrogate_matrix]
- [Phase randomization null model][spatiotemporal.surrogates.phase_randomize]


## Other great packages

This package does NOT provide the following methods from the paper, which are
readily available in these other great packages:

- Graph theoretical measures can be computed with [bctpy](https://github.com/aestrivex/bctpy)
- Intraclass Correlation Coefficient (ICC) can be computed using
  [pingouin.intraclass_corr](https://pingouin-stats.org/generated/pingouin.intraclass_corr.html)
- Partial correlation can be computed using
  [pingouin.partial_corr](https://pingouin-stats.org/generated/pingouin.partial_corr.html#pingouin.partial_corr)
- Plotting on the surface of the brain can be accomplished with [wbplot](https://github.com/jbburt/wbplot)
- Layout of the figures was with [CanD](https://github.com/mwshinn/CanD)
