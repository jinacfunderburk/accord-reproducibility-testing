import _gconcord as _cc
import numpy as np

def check_symmetry(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def ccista(S, lambda1, lambda2=0.0, epstol=1e-5, maxitr=100, steptype=1, penalize_diagonal=False):

    assert type(S) == np.ndarray and S.dtype == "float64"

    # assert type(lambda1) == 'float' or (type(data) == np.ndarray and data.shape == lambda1.shape)

    if type(lambda1) == float:
        lambda1 = np.full_like(S, lambda1, dtype="float64")
    
    assert type(lambda1) == np.ndarray and lambda1.dtype == "float64"
    assert check_symmetry(lambda1)

    if not penalize_diagonal:
        np.fill_diagonal(lambda1, 0)
    print(lambda1)

    return _cc.ccista(S, lambda1, lambda2, epstol, maxitr, steptype)
    # return _cc.ccista(S, S)

def h1(X, S):

    return -np.log(X.diagonal()).sum() + 0.5*np.matmul(X, np.matmul(S, X)).trace()

def grad_h1(X, S):

    W = np.matmul(S, X)
    G = - np.diag(1/X.diagonal()) + 0.5*(W + W.T)

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

def h2(X, lambda1):

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

def pyccista(S, lambda1, epstol=1e-5, maxitr=100, penalize_diagonal=False):

    assert check_symmetry(S)
    p, _ = S.shape

    if type(lambda1) == float:
        lambda1 = lambda1mat(lambda1, p, penalize_diagonal)

    assert check_symmetry(lambda1)

    X = np.identity(p)

    run_info = []
    while True:
        tau = 1.0
        c = 0.5

        G = grad_h1(X, S)

        inner_itr_count = 0
        while True:

            step = X - tau*G

            if np.all(step.diagonal() > 0):

                Xn = soft_threshold(step, tau*lambda1)

                if h1(Xn, S) <= quadratic_approx(Xn, X, S, tau):
                    break

            tau = c*tau
            inner_itr_count += 1
        
        subg = subgrad(S, Xn, lambda1)
        delta_subg = np.linalg.norm(subg)/np.linalg.norm(Xn)
        h = h1(Xn, S) + h2(Xn, lambda1)

        itr_info = [[inner_itr_count, delta_subg, h]]
        run_info += itr_info

        # print(itr_info)

        if delta_subg < epstol or len(run_info) > maxitr:
            break
        else:
            X = Xn

    return Xn, np.array(run_info)