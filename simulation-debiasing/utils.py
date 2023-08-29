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

def compute_tp_fp(Theta, Theta_hat):
    Theta_vec = np.where(Theta!=0, 1, 0).ravel()
    Theta_hat_vec = np.where(Theta_hat!=0, 1, 0).ravel()

    p = len(Theta)
    indx = np.arange(0, p**2, p+1)
    Theta_vec = np.where(Theta_vec != 0, 1, 0)
    Theta_vec = np.delete(Theta_vec, indx)
    Theta_hat_vec = np.where(Theta_hat_vec != 0, 1, 0)
    Theta_hat_vec = np.delete(Theta_hat_vec, indx)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Theta_vec, Theta_hat_vec).ravel()
    return tp, fp

def mcc(Theta, Theta_hat):
    """
    Compute Matthew's Correlation Coefficient (MCC) between two matrices.
    If diag=False, it excludes diagonal elements before computing MCC.
    """
    Theta_vec = Theta.ravel()
    Theta_hat_vec = Theta_hat.ravel()

    p = len(Theta)
    indx = np.arange(0, p**2, p+1)
    Theta_vec = np.where(Theta_vec != 0, 1, 0)
    Theta_vec = np.delete(Theta_vec, indx)
    Theta_hat_vec = np.where(Theta_hat_vec != 0, 1, 0)
    Theta_hat_vec = np.delete(Theta_hat_vec, indx)
    
    return sklearn.metrics.matthews_corrcoef(Theta_vec, Theta_hat_vec)

def precision_recall(Theta, Theta_hat):
    Theta_vec = Theta.ravel()
    Theta_hat_vec = Theta_hat.ravel()

    p = len(Theta)
    indx = np.arange(0, p**2, p+1)
    Theta_vec = np.where(Theta_vec != 0, 1, 0)
    Theta_vec = np.delete(Theta_vec, indx)
    Theta_hat_vec = np.where(Theta_hat_vec != 0, 1, 0)
    Theta_hat_vec = np.delete(Theta_hat_vec, indx)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Theta_vec, Theta_hat_vec).ravel()
    pos = len(np.nonzero(Theta_vec)[0])

    if tp + fp != 0:
        precision = tp/(tp+fp)
    else:
        precision = np.nan
    recall = tp/pos

    return precision, recall

def sensitivity_specificity(Theta, Theta_hat):
    Theta_vec = Theta.ravel()
    Theta_hat_vec = Theta_hat.ravel()

    p = len(Theta)
    indx = np.arange(0, p**2, p+1)
    Theta_vec = np.where(Theta_vec != 0, 1, 0)
    Theta_vec = np.delete(Theta_vec, indx)
    Theta_hat_vec = np.where(Theta_hat_vec != 0, 1, 0)
    Theta_hat_vec = np.delete(Theta_hat_vec, indx)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(Theta_vec, Theta_hat_vec).ravel()
    pos = len(np.nonzero(Theta_vec)[0])
    neg = len(Theta_vec) - pos

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

def proj_precision_mat(Omega, nz_indx):
    eig_val, eig_vec = np.linalg.eig(Omega)
    eig_val_new = np.maximum(eig_val, 0.4)
    Omega_temp = eig_vec @ np.diag(eig_val_new) @ eig_vec.T
    Omega_new = np.zeros_like(Omega_temp)
    Omega_new[nz_indx] = Omega_temp[nz_indx]
    np.fill_diagonal(Omega_new, 1)
    Omega_new = 0.5*(Omega_new + Omega_new.T)
    return Omega_new