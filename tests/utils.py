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

def BIC(X, Omega, modified=False, gamma=0.1):
    n, p = X.shape
    Omega_reg = Omega/Omega.diagonal()[None,:]
    
    RSS = (X @ Omega_reg)**2
    RSS_i = RSS.sum(axis=0)
    num_nonzero = len(np.flatnonzero(Omega_reg))
    
    if modified:
        BIC = (n*np.log(RSS_i).sum()) + (np.log(n) * num_nonzero) + (4*num_nonzero*gamma*np.log(p))
    else: 
        BIC = (n*np.log(RSS_i).sum()) + (np.log(n) * num_nonzero)
    
    return BIC

def pseudo_BIC(X, Omega, modified=False, gamma=0.1):
    n, p = X.shape
    Omega_reg = Omega/Omega.diagonal()[None,:]
    
    RSS = (X @ Omega_reg)**2
    RSS_i = RSS.sum(axis=0)
    num_nonzero = len(np.flatnonzero(Omega_reg))

    if modified:
        BIC = (np.log(n) * num_nonzero) + np.inner(np.diag(Omega), RSS_i) - n*np.sum(np.log(np.diag(Omega))) + (4*num_nonzero*gamma*np.log(p))
    else: 
        BIC = (np.log(n) * num_nonzero) + np.inner(np.diag(Omega), RSS_i) - n*np.sum(np.log(np.diag(Omega)))
    
    return BIC