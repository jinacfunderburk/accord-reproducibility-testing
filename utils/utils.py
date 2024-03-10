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

def partial_corr(Theta):
    std_inv = np.diag(np.sqrt(1/np.diag(Theta)))
    Rho = -1 * (std_inv @ Theta @ std_inv)
    np.fill_diagonal(Rho, 1)
    
    return Rho

def partial_corr_to_precision(Rho, diagonal_values):
    p = len(Rho)
    Theta = np.zeros_like(Rho)
    
    for i in range(p):
        for j in range(i + 1, p):
            Theta[i, j] = -Rho[i, j] / np.sqrt(diagonal_values[i] * diagonal_values[j])
            Theta[j, i] = Theta[i, j]
    
    np.fill_diagonal(Theta, diagonal_values)
    
    return Theta

def tp_fp(Theta, Theta_hat):
    Theta_bool = np.where(Theta != 0, 1, 0)
    Theta_hat_bool = np.where(Theta_hat != 0, 1, 0)

    p = len(Theta)
    mask = np.tri(p, p, k=-1, dtype=bool)
    edge_true = Theta_bool[mask]
    edge_hat = Theta_hat_bool[mask]

    tp = np.sum((edge_true == 1) & (edge_hat == 1))
    fp = np.sum((edge_true == 0) & (edge_hat == 1))

    return tp, fp

def precision_recall(Theta, Theta_hat):
    Theta_bool = np.where(Theta != 0, 1, 0)
    Theta_hat_bool = np.where(Theta_hat != 0, 1, 0)

    p = len(Theta)
    mask = np.tri(p, p, k=-1, dtype=bool)
    edge_true = Theta_bool[mask]
    edge_hat = Theta_hat_bool[mask]

    tp = np.sum((edge_true == 1) & (edge_hat == 1))
    fp = np.sum((edge_true == 0) & (edge_hat == 1))
    pos = np.sum(edge_true)

    if tp + fp != 0:
        precision = tp/(tp+fp)
    else:
        precision = np.nan
    recall = tp/pos

    return precision, recall

def mcc(Theta, Theta_hat):
    """
    Compute Matthew's Correlation Coefficient (MCC) between two symmetric matrices.
    """
    Theta_bool = np.where(Theta != 0, 1, 0)
    Theta_hat_bool = np.where(Theta_hat != 0, 1, 0)

    p = len(Theta)
    mask = np.tri(p, p, k=-1, dtype=bool)
    edge_true = Theta_bool[mask]
    edge_hat = Theta_hat_bool[mask]
    
    return sklearn.metrics.matthews_corrcoef(edge_true, edge_hat)

def compute_average_norm(Theta, Theta_hat):
    p = len(Theta)
    Theta_bool = np.where(Theta != 0, 1, 0)
    Theta_hat_bool = np.where(Theta_hat != 0, 1, 0)

    mask = np.tri(p, p, k=-1, dtype=bool)
    edge_true = Theta_bool[mask]
    edge_hat = Theta_hat_bool[mask]
    TP_indx = np.where((edge_true == 1) & (edge_hat == 1) == 1)[0]
    FP_indx = np.where((edge_true == 0) & (edge_hat == 1) == 1)[0]
    FN_indx = np.where((edge_true == 1) & (edge_hat == 0) == 1)[0]
    count_TP = len(TP_indx)
    count_FP = len(FP_indx)
    count_FN = len(FN_indx)

    theta_true = Theta[mask]
    theta_hat = Theta_hat[mask]

    norm_total = np.linalg.norm(np.tril(Theta) - np.tril(Theta_hat))
    norm_TP = np.sqrt(np.mean((theta_true[TP_indx]-theta_hat[TP_indx])**2))
    norm_FP = np.sqrt(np.mean((theta_true[FP_indx]-theta_hat[FP_indx])**2))
    norm_FN = np.sqrt(np.mean((theta_true[FN_indx]-theta_hat[FN_indx])**2))
    norm_diag = np.sqrt(np.mean((np.diag(Theta) - np.diag(Theta_hat))**2))

    return norm_total, norm_TP, norm_FP, norm_FN, norm_diag, count_TP, count_FP, count_FN

def pseudo_BIC(X, Theta, modified=False, gamma=0.1):
    n, p = X.shape
    Theta_reg = Theta/Theta.diagonal()[None,:]
    
    RSS = (X @ Theta_reg)**2
    RSS_i = RSS.sum(axis=0)
    num_param = ((len(np.flatnonzero(Theta)) - p)/2)
    
    if modified:
        BIC = (num_param * np.log(n)) + np.inner(np.diag(Theta), RSS_i) - n*np.sum(np.log(np.diag(Theta))) + 4*num_param*gamma*np.log(p)
    else:
        BIC = (num_param * np.log(n)) + np.inner(np.diag(Theta), RSS_i) - n*np.sum(np.log(np.diag(Theta)))
    
    return BIC

def gauss_BIC(X, Theta):
    n, p = X.shape
    S = np.matmul(X.T, X, dtype=np.float64)/n
    num_param = ((len(np.flatnonzero(Theta)) - p)/2)
    
    BIC = (num_param * np.log(n)) - 2*(np.log(np.linalg.det(Theta)) - np.trace(Theta @ S))
    
    return BIC

def proj_precision_mat(Theta, nz_indx):
    eig_val, eig_vec = np.linalg.eig(Theta)
    eig_val_new = np.maximum(eig_val, 0.4)
    Theta_temp = eig_vec @ np.diag(eig_val_new) @ eig_vec.T
    Theta_new = np.zeros_like(Theta_temp)
    Theta_new[nz_indx] = Theta_temp[nz_indx]
    np.fill_diagonal(Theta_new, 1)
    Theta_new = 0.5*(Theta_new + Theta_new.T)
    return Theta_new

def h1(X, S, lam2):
    return 0.5*np.matmul(X.T, np.matmul(X, S)).trace() + 0.5*lam2*np.linalg.norm(X, 'fro')**2

def h2(X, lam1):
    return -np.log(X.diagonal()).sum() + (lam1 * np.abs(X)).sum()

def newton_one_step(S, Omega):
    first_two_terms = 2*Omega - Omega @ S @ np.diag(np.diag(Omega)) @ Omega
    third_term = 0.5*(Omega - np.diag(np.diag(Omega @ S @ Omega.T)) @ Omega)
    T = first_two_terms - third_term
    return T