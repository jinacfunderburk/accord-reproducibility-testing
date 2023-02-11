import _gconcorde as _cce
import numpy as np

def check_symmetry(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_lambda_matrix(lam1, S):
    if isinstance(lam1, float) | isinstance(lam1, int):
        lam_mat = np.full_like(S, lam1, order='F', dtype='float64')
    elif isinstance(lam1, np.ndarray):
        lam_mat = lam1
    return lam_mat

def cce(S, lam1=0.1, lam2=0.0, epstol=1e-5, maxitr=100, penalize_diag=True):
    """
    CONCORDe algorithm for precision matrix estimation
    
    Parameters
    ----------
    S : ndarray of shape (n_features, n_features)
        Sample covariance matrix
    lam : float
        The l1-regularization parameter
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements
    
    Returns
    -------
    Omega : ndarray of shape (n_features, n_features)
        Estimated precision matrix
    hist : ndarray of shape (n_iters, 2)
        The list of values of (successive_norm_diff, objective) at each iteration until convergence
    """
    assert (type(S) == np.ndarray and S.dtype == 'float64')

    lam_mat = get_lambda_matrix(lam1, S)
    
    assert type(lam_mat) == np.ndarray and lam_mat.dtype == 'float64'
    assert check_symmetry(lam_mat)

    if not penalize_diag:
        np.fill_diagonal(lam_mat, 0)

    hist_norm_diff = np.full((maxitr, 1), -1, order='F', dtype='float64')
    hist_hn = np.full((maxitr, 1), -1, order='F', dtype='float64')

    tau = 1/np.linalg.svd(S)[1][0]

    Omega = _cce.cce(S, lam_mat, lam2, epstol, maxitr, tau, penalize_diag, hist_norm_diff, hist_hn)
    hist = np.hstack([hist_norm_diff, hist_hn])
    hist = hist[np.where(hist[:,0]!=-1)]

    return Omega, hist

def BIC(X, Omega, modified=False, gamma=0.1):
    n, p = X.shape
    Omega_reg = Omega/Omega.diagonal()[None,:]
    
    RSS = (X @ Omega_reg.T)**2
    RSS_i = RSS.sum(axis=0)
    num_nonzero = len(np.flatnonzero(Omega_reg))
    
    if modified:
        BIC = (n*np.log(RSS_i).sum()) + (np.log(n) * num_nonzero) + (4*num_nonzero*gamma*np.log(p))
    else: 
        BIC = (n*np.log(RSS_i).sum()) + (np.log(n) * num_nonzero)
    
    return BIC

class GraphicalConcorde:
    """
    CONCORDe algorithm for precision matrix estimation
    
    Parameters
    ----------
    lam : float
        The l1-regularization parameter
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements
    
    Attributes
    ----------
    precision_ : ndarray of shape (n_features, n_features)
        Estimated precision matrix
    hist_ : ndarray of shape (n_iters, 2)
        The list of values of (successive_norm_diff, objective) at each iteration until convergence
    """
    def __init__(self, lam1=0.1, lam2=0.0, epstol=1e-5, maxitr=100, penalize_diag=True):
        self.lam1 = lam1
        self.lam2 = lam2
        self.epstol = epstol
        self.maxitr = maxitr
        self.penalize_diag = penalize_diag
    
    def fit(self, X, y=None):
        """
        Fit CONCORDe

        Parameters
        ----------
        X : ndarray, shape (n_samples, p_features)
            Data from which to compute the inverse covariance matrix
        y : (ignored)
        """
        S = np.cov(X, rowvar=False)

        self.precision_, self.hist_ = cce(S,
                                          lam1=self.lam1,
                                          lam2=self.lam2,
                                          epstol=self.epstol,
                                          maxitr=self.maxitr,
                                          penalize_diag=self.penalize_diag)
        
        return self

class GraphicalConcordeCV(GraphicalConcorde):
    """
    CONCORDe algorithm with cross-validation
    
    Parameters
    ----------
    lams :  int or array-like of shape (n_lams,), dtype=float
        Grid of lambdas for cross-validation
        If an integer is given, it creates a grid of the given number of lambdas
    epstol : float, default=1e-5
        Convergence threshold
    maxitr : int, default=100
        The maximum number of iterations
    penalize_diag : bool, default=True
        Whether or not to penalize the diagonal elements
    val_score : {'BIC', 'EBIC'}, default='EBIC'
        Choice of validation score
    gamma : float, default=None
        The tuning parameter for EBIC, used only if val_score='EBIC'
    
    Attributes
    ----------
    precision_ : ndarray of shape (n_features, n_features)
        Estimated precision matrix
    best_params_ : float
        Best parameter chosen by cross-validation
    cv_results_ : dict of ndarrays
        A dict with keys:
        lam : ndarray of shape (n_lams,)
            Explored lambdas
        score : ndarray of shape (n_lams,)
            BIC/EBIC scores corresponding to lambdas
    hist_ : ndarray of shape (n_iters, 2)
        The list of values of (successive_norm_diff, objective) at each iteration until convergence
    """
    def __init__(self, lam1s=5, lam2s=[0.1], epstol=1e-5, maxitr=100, penalize_diag=True, val_score='EBIC', gamma=0.1):
        self.lam1s = lam1s
        self.lam2s = lam2s
        self.epstol = epstol
        self.maxitr = maxitr
        self.penalize_diag = penalize_diag
        self.val_score = val_score
        self.gamma = gamma

    def fit(self, X, y=None):
        """
        Fit CONCORDe with cross-validation

        Parameters
        ----------
        X : ndarray, shape (n_samples, p_features)
            Data from which to compute the inverse covariance matrix
        y : (ignored)
        """
        if isinstance(self.lam1s, int):
            S = np.cov(X, rowvar=False)
            S.flat[::S.shape[0] + 1] = 0
            lam_max = np.max(np.abs(S))
            lam_min = 0.01 * lam_max
            lam1s = np.logspace(np.log10(lam_min), np.log10(lam_max), self.lam1s)
        elif isinstance(self.lams, list) | isinstance(self.lams, np.ndarray):
            lam1s = self.lam1s

        S = np.cov(X, rowvar=False)
        cv_results = []
        best_lam1, best_lam2, best_score = lam1s[0], self.lam2s[0], np.inf
        for lam1 in lam1s:
            for lam2 in self.lam2s:
                Omega, hist = cce(S,
                                  lam1=lam1,
                                  lam2=lam2,
                                  epstol=self.epstol,
                                  maxitr=self.maxitr,
                                  penalize_diag=self.penalize_diag)

                if self.val_score == 'BIC':
                    curr_score = BIC(X, Omega.toarray(), modified=False, gamma=self.gamma)
                else:
                    curr_score = BIC(X, Omega.toarray(), modified=True, gamma=self.gamma)
                
                cv_results += [{'lam1': lam1,
                                'lam2': lam2,
                                'score': curr_score}]

                if best_score >= curr_score:
                    best_lam1, best_lam2, best_score = lam1, lam2, curr_score
                    self.precision_, self.hist_ = Omega, hist
        
        self.best_param_ = [best_lam1, best_lam2]
        self.cv_results_ = cv_results
            
        return self