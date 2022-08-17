import gconcord as cc
import numpy as np

# import importlib
# importlib.reload(cc)

# true precision matrix
omega = np.zeros((3,3), order="C")
omega[0, 1] = omega[1, 2] = 2.1
omega += omega.T
np.fill_diagonal(omega, 3)

# true covariance matrix
sigma = np.linalg.inv(omega)

# observations
data = np.random.multivariate_normal([0]*3, sigma, 500)

# sample covariance matrix
S = np.cov(data, rowvar=False)

