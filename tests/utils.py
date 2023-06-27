import numpy as np
import sklearn
import sklearn.metrics
import networkx as nx

def standardize(X, bias=False):
    X_std = X.copy()
    X_std -= X_std.mean(axis=0)
    if bias:
        X_std /= X_std.std(axis=0)
    else:
        X_std /= X_std.std(axis=0, ddof=1)
    
    return X_std

def partial_corr(A):
    std_inv = np.diag(np.sqrt(1/np.diag(A)))
    A_corr = -1 * (std_inv @ A @ std_inv)
    np.fill_diagonal(A_corr, 1)
    
    return A_corr

def compute_tp_fp(Omega, Omega_hat):
    Omega_vec = np.where(Omega!=0, 1, 0).ravel()
    Omega_hat_vec = np.where(Omega_hat!=0, 1, 0).ravel()

    p = len(Omega)
    indx = np.arange(0, p**2, p+1)
    Omega_vec = np.where(Omega_vec != 0, 1, 0)
    Omega_vec = np.delete(Omega_vec, indx)
    Omega_hat_vec = np.where(Omega_hat_vec != 0, 1, 0)
    Omega_hat_vec = np.delete(Omega_hat_vec, indx)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Omega_vec, Omega_hat_vec).ravel()
    return tp, fp

def mcc(Omega, Omega_hat):
    """
    Compute Matthew's Correlation Coefficient (MCC) between two matrices.
    If diag=False, it excludes diagonal elements before computing MCC.
    """
    Omega_vec = Omega.ravel()
    Omega_hat_vec = Omega_hat.ravel()

    p = len(Omega)
    indx = np.arange(0, p**2, p+1)
    Omega_vec = np.where(Omega_vec != 0, 1, 0)
    Omega_vec = np.delete(Omega_vec, indx)
    Omega_hat_vec = np.where(Omega_hat_vec != 0, 1, 0)
    Omega_hat_vec = np.delete(Omega_hat_vec, indx)
    
    return sklearn.metrics.matthews_corrcoef(Omega_vec, Omega_hat_vec)

def precision_recall(Omega, Omega_hat):
    Omega_vec = Omega.ravel()
    Omega_hat_vec = Omega_hat.ravel()

    p = len(Omega)
    indx = np.arange(0, p**2, p+1)
    Omega_vec = np.where(Omega_vec != 0, 1, 0)
    Omega_vec = np.delete(Omega_vec, indx)
    Omega_hat_vec = np.where(Omega_hat_vec != 0, 1, 0)
    Omega_hat_vec = np.delete(Omega_hat_vec, indx)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Omega_vec, Omega_hat_vec).ravel()
    pos = len(np.nonzero(Omega_vec)[0])

    if tp + fp != 0:
        precision = tp/(tp+fp)
    else:
        precision = np.nan
    recall = tp/pos

    return precision, recall

def sensitivity_specificity(Omega, Omega_hat):
    Omega_vec = Omega.ravel()
    Omega_hat_vec = Omega_hat.ravel()

    p = len(Omega)
    indx = np.arange(0, p**2, p+1)
    Omega_vec = np.where(Omega_vec != 0, 1, 0)
    Omega_vec = np.delete(Omega_vec, indx)
    Omega_hat_vec = np.where(Omega_hat_vec != 0, 1, 0)
    Omega_hat_vec = np.delete(Omega_hat_vec, indx)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Omega_vec, Omega_hat_vec).ravel()
    pos = len(np.nonzero(Omega_vec)[0])
    neg = len(Omega_vec) - pos

    sensitivity = tp/pos
    specificity = tn/neg

    return sensitivity, specificity

def pseudo_BIC(X, Theta):
    n, p = X.shape
    Theta_reg = Theta/Theta.diagonal()[None,:]
    
    RSS = (X @ Theta_reg)**2
    RSS_i = RSS.sum(axis=0)
    num_nonzero = len(np.flatnonzero(Theta_reg))
    BIC = (np.log(n) * num_nonzero) + np.inner(np.diag(Theta), RSS_i) - n*np.sum(np.log(np.diag(Theta)))
    
    return BIC

def h1(X, S, lam2):
    return 0.5*np.matmul(X.T, np.matmul(X, S)).trace() + 0.5*lam2*np.linalg.norm(X, 'fro')**2

def h2(X, lam1):
    return -np.log(X.diagonal()).sum() + (lam1 * np.abs(X)).sum()
    
def get_precision(X):
    return .5*(np.diag(np.diag(X)) @ X) + .5*(X.T @ np.diag(np.diag(X)))