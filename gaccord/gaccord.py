import _gaccord as _accord
import numpy as np

def check_symmetry(a, rtol=1e-5, atol=1e-8):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_lambda_matrix(lam1, S):
    if isinstance(lam1, float) | isinstance(lam1, int):
        lam_mat = np.full_like(S, lam1, order='F', dtype='float64')
    elif isinstance(lam1, np.ndarray):
        lam_mat = lam1
    return lam_mat

def ccista(S, Omega_star, lam1=0.1, lam2=0.0, constant_stepsize=0.5, backtracking=True, epstol=1e-5, maxitr=100, penalize_diag=True):
    """
    Modified CONCORD algorithm for convergence analysis

    Parameters
    ----------
    S : ndarray of shape (n_features, n_features)
        Sample covariance matrix
    Omega_star : ndarray of shape (n_features, n_features)
        Proxy of converged Omega
    lam1 : float
        The l1-regularization parameter
    lam2 : float
        The l2-regularization parameter
    constant_stepsize : float
        Constant step size
    backtracking : bool, default=True
        Whether ot nor to perform backtracking
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements

    Returns
    -------
    Omega : ndarray of shape (n_features, n_features)
        Estimated Omega
    hist : ndarray of shape (n_iters, 5)
        The list of values of (inner_iter_count, objective, successive_norm, omega_star_norm, iter_time) at each iteration until convergence
        inner_iter_count is included only when backtracking=True
    """
    assert (type(S) == np.ndarray and S.dtype == 'float64')

    lam_mat = get_lambda_matrix(lam1, S)
    
    assert type(lam_mat) == np.ndarray and lam_mat.dtype == 'float64'
    assert check_symmetry(lam_mat)

    if not penalize_diag:
        np.fill_diagonal(lam_mat, 0)

    hist_inner_itr_count = np.full((maxitr, 1), -1, order='F', dtype='int32')
    hist_hn = np.full((maxitr, 1), -1, order='F', dtype='float64')
    hist_successive_norm = np.full((maxitr, 1), -1, order='F', dtype='float64')
    hist_norm = np.full((maxitr, 1), -1, order='F', dtype='float64')
    hist_iter_time = np.full((maxitr, 1), -1, order='F', dtype='float64')

    if backtracking:
        Omega = _accord.ccista_backtracking(S, Omega_star, lam_mat, lam2, epstol, maxitr, hist_inner_itr_count, hist_hn, hist_successive_norm, hist_norm, hist_iter_time)
        hist = np.hstack([hist_inner_itr_count, hist_hn, hist_successive_norm, hist_norm, hist_iter_time])
    else:
        Omega = _accord.ccista_constant(S, Omega_star, lam_mat, lam2, epstol, maxitr, constant_stepsize, hist_hn, hist_successive_norm, hist_norm, hist_iter_time)
        hist = np.hstack([hist_hn, hist_successive_norm, hist_norm, hist_iter_time])
    
    hist = hist[np.where(hist[:,0]!=-1)]

    return Omega, hist

def accord(S, Omega_star, lam1=0.1, lam2=0.0, stepsize_multiplier=1, backtracking=True, epstol=1e-5, maxitr=100, penalize_diag=True):
    """
    Modified ACCORD algorithm for convergence analysis
    
    Parameters
    ----------
    S : ndarray of shape (n_features, n_features)
        Sample covariance matrix
    Omega_star : ndarray of shape (n_features, n_features)
        Proxy of converged Omega
    lam1 : float
        The l1-regularization parameter
    lam2 : float
        The l2-regularization parameter
    stepsize_multiplier : int or float
        Multiplier for stepsize
    backtracking : bool, default=True
        Whether ot nor to perform backtracking with lower bound
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements
    
    Returns
    -------
    Omega : ndarray of shape (n_features, n_features)
        Estimated Omega
    hist : ndarray of shape (n_iters, 5)
        The list of values of (inner_iter_count, objective, successive_norm, omega_star_norm, iter_time) at each iteration until convergence
        inner_iter_count is included only when backtracking=True
    """
    assert (type(S) == np.ndarray and S.dtype == 'float64')

    lam_mat = get_lambda_matrix(lam1, S)
    
    assert type(lam_mat) == np.ndarray and lam_mat.dtype == 'float64'
    assert check_symmetry(lam_mat)

    if not penalize_diag:
        np.fill_diagonal(lam_mat, 0)

    hist_inner_itr_count = np.full((maxitr, 1), -1, order='F', dtype='int32')
    hist_hn = np.full((maxitr, 1), -1, order='F', dtype='float64')
    hist_successive_norm = np.full((maxitr, 1), -1, order='F', dtype='float64')
    hist_norm = np.full((maxitr, 1), -1, order='F', dtype='float64')
    hist_iter_time = np.full((maxitr, 1), -1, order='F', dtype='float64')

    tau = (stepsize_multiplier*1)/np.linalg.svd(S)[1][0]

    if backtracking:
        Omega = _accord.accord_backtracking(S, Omega_star, lam_mat, lam2, epstol, maxitr, tau, penalize_diag, hist_inner_itr_count, hist_hn, hist_successive_norm, hist_norm, hist_iter_time)
        hist = np.hstack([hist_inner_itr_count, hist_hn, hist_successive_norm, hist_norm, hist_iter_time])
    else:
        Omega = _accord.accord_constant(S, Omega_star, lam_mat, lam2, epstol, maxitr, tau, penalize_diag, hist_hn, hist_successive_norm, hist_norm, hist_iter_time)
        hist = np.hstack([hist_hn, hist_successive_norm, hist_norm, hist_iter_time])
    
    hist = hist[np.where(hist[:,0]!=-1)]

    return Omega, hist

class GraphicalConcord:
    """
    Modified CONCORD algorithm for convergence analysis

    Parameters
    ----------
    Omega_star : ndarray of shape (n_features, n_features)
        Proxy of converged Omega
    lam1 : float
        The l1-regularization parameter
    lam2 : float
        The l2-regularization parameter
    constant_stepsize : float
        Constant step size
    backtracking : bool, default=True
        Whether ot nor to perform backtracking
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements

    Attributes
    ----------
    omega_ : ndarray of shape (n_features, n_features)
        Estimated Omega
    hist_ : ndarray of shape (n_iters, 5)
        The list of values of (inner_iter_count, objective, successive_norm, omega_star_norm, iter_time) at each iteration until convergence
        inner_iter_count is included only when backtracking=True
    """
    def __init__(self, Omega_star=None, lam1=0.1, lam2=0.0, constant_stepsize=0.5, backtracking=True, epstol=1e-5, maxitr=100, penalize_diag=True):
        self.Omega_star = Omega_star
        self.lam1 = lam1
        self.lam2 = lam2
        self.constant_stepsize = constant_stepsize
        self.backtracking = backtracking
        self.epstol = epstol
        self.maxitr = maxitr
        self.penalize_diag = penalize_diag
    
    def fit(self, X, y=None):
        """
        Fit CONCORD

        Parameters
        ----------
        X : ndarray, shape (n_samples, p_features)
            Data from which to compute the inverse covariance matrix
        y : (ignored)
        """
        S = np.cov(X, rowvar=False)

        self.omega_, self.hist_ = ccista(S,
                                         Omega_star=self.Omega_star,
                                         lam1=self.lam1,
                                         lam2=self.lam2,
                                         constant_stepsize=self.constant_stepsize,
                                         backtracking=self.backtracking,
                                         epstol=self.epstol,
                                         maxitr=self.maxitr,
                                         penalize_diag=self.penalize_diag)
        
        return self

class GraphicalAccord:
    """
    Modified ACCORD algorithm for convergence analysis
    
    Parameters
    ----------
    Omega_star : ndarray of shape (n_features, n_features)
        Proxy of converged Omega
    lam1 : float
        The l1-regularization parameter
    lam2 : float
        The l2-regularization parameter
    stepsize_multiplier : int or float
        Multiplier for stepsize
    backtracking : bool, default=True
        Whether ot nor to perform backtracking with lower bound
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements
    
    Attributes
    ----------
    omega_ : ndarray of shape (n_features, n_features)
        Estimated Omega
    hist_ : ndarray of shape (n_iters, 5)
        The list of values of (inner_iter_count, objective, successive_norm, omega_star_norm, iter_time) at each iteration until convergence
        inner_iter_count is included only when backtracking=True
    """
    def __init__(self, Omega_star=None, lam1=0.1, lam2=0.0, stepsize_multiplier=1, backtracking=True, epstol=1e-5, maxitr=100, penalize_diag=True):
        self.Omega_star = Omega_star
        self.lam1 = lam1
        self.lam2 = lam2
        self.stepsize_multiplier = stepsize_multiplier
        self.backtracking = backtracking
        self.epstol = epstol
        self.maxitr = maxitr
        self.penalize_diag = penalize_diag
    
    def fit(self, X, y=None):
        """
        Fit ACCORD

        Parameters
        ----------
        X : ndarray, shape (n_samples, p_features)
            Data from which to compute the inverse covariance matrix
        y : (ignored)
        """
        S = np.cov(X, rowvar=False)
        
        self.omega_, self.hist_ = accord(S,
                                         Omega_star=self.Omega_star,
                                         lam1=self.lam1,
                                         lam2=self.lam2,
                                         stepsize_multiplier=self.stepsize_multiplier,
                                         backtracking=self.backtracking,
                                         epstol=self.epstol,
                                         maxitr=self.maxitr,
                                         penalize_diag=self.penalize_diag)
        
        return self