import _gconcorde as _cce
import numpy as np

def check_symmetry(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def cceista(S, lambda1, Omega_star, epstol=1e-5, maxitr=100, penalize_diagonal=False):

    assert type(S) == np.ndarray and S.dtype == "float64"

    # assert type(lambda1) == 'float' or (type(data) == np.ndarray and data.shape == lambda1.shape)

    if (type(lambda1) == float) | (type(lambda1) == np.float64):
        lambda1 = np.full_like(S, lambda1, order="F", dtype="float64")
    
    assert type(lambda1) == np.ndarray and lambda1.dtype == "float64"
    assert check_symmetry(lambda1)

    if not penalize_diagonal:
        np.fill_diagonal(lambda1, 0)

    hist_inner_itr_count = np.full((maxitr, 1), -1, order="F", dtype='int32')
    hist_delta_updates = np.full((maxitr, 1), -1, order="F", dtype='float64')
    hist_hn = np.full((maxitr, 1), -1, order="F", dtype='float64')
    hist_norm = np.full((maxitr, 1), -1, order="F", dtype='float64')
    hist_iter_time = np.full((maxitr, 1), -1, order="F", dtype='float64')

    # return _cc.ccista(S, lambda1, lambda2, epstol, maxitr, steptype)
    Xn = _cce.cceista(S, lambda1, Omega_star, epstol, maxitr, hist_inner_itr_count, hist_delta_updates, hist_hn, hist_norm, hist_iter_time)

    hist = np.hstack([hist_inner_itr_count, hist_delta_updates, hist_hn, hist_norm, hist_iter_time])
    hist = hist[np.where(hist[:,0]!=-1)]

    return Xn, hist

def cce_constant(S, lambda1, Cov, Omega_star, epstol=1e-5, maxitr=100, penalize_diagonal=False):

    assert (type(S) == np.ndarray and S.dtype == "float64")

    # assert type(lambda1) == 'float' or (type(data) == np.ndarray and data.shape == lambda1.shape)

    if (type(lambda1) == float) | (type(lambda1) == np.float64):
        lambda1 = np.full_like(S, lambda1, order="F", dtype="float64")
    
    assert type(lambda1) == np.ndarray and lambda1.dtype == "float64"
    assert check_symmetry(lambda1)

    if not penalize_diagonal:
        np.fill_diagonal(lambda1, 0)

    # hist_inner_itr_count = np.full((maxitr, 1), -1, order="F", dtype='int32')
    hist_delta_updates = np.full((maxitr, 1), -1, order="F", dtype='float64')
    hist_hn = np.full((maxitr, 1), -1, order="F", dtype='float64')
    hist_norm = np.full((maxitr, 1), -1, order="F", dtype='float64')
    hist_iter_time = np.full((maxitr, 1), -1, order="F", dtype='float64')

    # tau = 1/power_method(S)
    # tau = 1/np.linalg.svd(S)[1][0]
    tau = 1/np.linalg.svd(Cov)[1][0]

    # return _cc.ccista(S, lambda1, lambda2, epstol, maxitr, steptype)
    Xn = _cce.cce_constant(S, lambda1, Omega_star, epstol, maxitr, tau, penalize_diagonal, hist_delta_updates, hist_hn, hist_norm, hist_iter_time)

    # hist = np.hstack([hist_inner_itr_count, hist_delta_updates, hist_hn])
    hist = np.hstack([hist_delta_updates, hist_hn, hist_norm, hist_iter_time])
    hist = hist[np.where(hist[:,0]!=-1)]

    return Xn, hist

def h1(X, S, constant=False):
    if constant:
        return 0.5*np.matmul(X.T, np.matmul(X, S)).trace()
    else:
        return -np.log(X.diagonal()).sum() + 0.5*np.matmul(X.T, np.matmul(X, S)).trace()

def grad_h1(X, S, constant=False):

    if constant:
        G = np.matmul(X, S)
    else:
        W = np.matmul(X, S)
        G = - np.diag(1/X.diagonal()) + W

    return G

def soft_threshold(mat, lambda1):

    return np.sign(mat)*np.maximum(np.abs(mat) - lambda1, 0)

def quadratic_approx(mat_next, mat, S, tau):

    step = mat_next - mat

    Q = h1(mat, S) + np.matmul(step.T, grad_h1(mat, S)).trace() + (0.5/tau)*np.linalg.norm(step, "fro")**2

    return Q

def lambda1mat(lambda1, p, penalize_diagonal):

    mat = lambda1 * np.ones((p, p), dtype="float64")

    if penalize_diagonal is False:
        np.fill_diagonal(mat, 0)
    
    return mat

def h2(X, lambda1, constant=False):
    if constant:
        return -np.log(X.diagonal()).sum() + (lambda1 * np.abs(X)).sum()
    else:
        return (lambda1 * np.abs(X)).sum()

def subgrad_h2(X, lambda1, g_h1=None):

    subgrad = np.sign(X)

    # if gradient of h1 is given,
    # compute subgradient of h2 to get as close to zero
    if g_h1 is not None:
        # subgradient at discontinuity: i.e. X==0 is in [-1, 1]
        # choose something in this range to make subgrad as small as possible
        sg_d = (X==0) * np.divide(g_h1, lambda1, where=(X==0))
        sg_d = np.sign(sg_d)*np.minimum(np.abs(sg_d), 1)
        subgrad -= sg_d

    return lambda1 * subgrad

def subgrad(S, X, lambda1):

    g_h1 = grad_h1(X, S)
    subg_h2 = subgrad_h2(X, lambda1, g_h1)

    return g_h1 + subg_h2

def pycceista(S, lambda1, Omega_star, epstol=1e-5, maxitr=100, penalize_diagonal=False):

    assert check_symmetry(S)
    p, _ = S.shape

    if type(lambda1) == float:
        lambda1 = lambda1mat(lambda1, p, penalize_diagonal)

    assert check_symmetry(lambda1)

    X = np.identity(p)
    G = grad_h1(X, S)

    run_info = []
    while True:
        tau = 1.0
        c = 0.5

        inner_itr_count = 0
        while True:

            Xn = soft_threshold(X - tau*G, tau*lambda1)

            if np.all(Xn.diagonal() > 0) and (h1(Xn, S) <= quadratic_approx(Xn, X, S, tau)):
                break
            else:
                tau = c*tau
                inner_itr_count += 1
        
        # subg = subgrad(S, Xn, lambda1)
        # delta_subg = np.linalg.norm(subg)/np.linalg.norm(Xn)
        Xnorm = np.linalg.norm(Xn-X)
        hist_norm = np.linalg.norm(Omega_star - Xn)
        hn = h1(Xn, S) + h2(Xn, lambda1)

        itr_info = [[inner_itr_count, Xnorm, hn, hist_norm]]
        run_info += itr_info

        if Xnorm < epstol or len(run_info) > maxitr:
            break
        else:
            X = Xn
            G = grad_h1(X, S)

    return Xn, np.array(run_info)

def power_method(S, n=100):
    u = np.ones(len(S))
    for i in range(n):
        u = S @ u
        eigenvalue = np.max(u)
        u = u/eigenvalue
        
    return eigenvalue

def pycce_constant(S, lambda1, Cov, Omega_star, epstol=1e-5, maxitr=100, penalize_diagonal=False):

    assert check_symmetry(S)
    p, _ = S.shape

    if type(lambda1) == float:
        lambda1 = lambda1mat(lambda1, p, penalize_diagonal)

    assert check_symmetry(lambda1)

    X = np.identity(p)
    # tau = 1/power_method(S)
    tau = 1/power_method(Cov)
    # tau = 1/np.linalg.svd(S)[1][0]

    run_info = []
    while True:
        G = grad_h1(X, S, constant=True)
        step = X - tau*G

        if penalize_diagonal:
            y = np.diag(step) - np.diag(tau*lambda1)
            Xn = soft_threshold(step, tau*lambda1)
            np.fill_diagonal(Xn, 0.5*(y+np.sqrt(y**2 + 4*tau)))
        else:
            y = np.diag(step)
            Xn = soft_threshold(step, tau*lambda1)
            np.fill_diagonal(Xn, 0.5*(y+np.sqrt(y**2 + 4*tau)))

        Xnorm = np.linalg.norm(Xn-X)
        hist_norm = np.linalg.norm(Omega_star - Xn)
        h = h1(Xn, S, constant=True) + h2(Xn, lambda1, constant=True)

        itr_info = [[Xnorm, h, hist_norm]]
        run_info += itr_info

        if Xnorm < epstol or len(run_info) > maxitr:
            break
        else:
            X = Xn

    return Xn, np.array(run_info)