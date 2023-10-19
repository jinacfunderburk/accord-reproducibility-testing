Reproducibility for the ACCORD Simulations
===============

This repository includes code to reproduce the results of the ACCORD simulations.

Installation
---------------

Install the package in development mode using the following command:
```
pip install -e .
```

Directory Structure
---------------

 - __src__: It contains C++ source code for various versions of ACCORD & CONCORD algorithms.
 - __gaccord__: It contains Python classes for the ACCORD & CONCORD algorithms.
 - __simulation-convergence__: It contains code for reproducing results for the linear convergence property of the ACCORD algorithm.
 - __simulation-debiasing__: It contains code for reproducing results for debiasing experiments.
 - __simulation-edge-detection__: It contains code for reproducing results for comparing edge detection rates between methods.
 - __simulation-model-selection__: It contains code for reproducing results for behaviors of various model selection criteria.
 - __output__: It contains all results.