import _gconcord as _cc
import numpy as np

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def ccista(S, lambda1, lambda2=0.0, epstol=1e-5, maxitr=100, steptype=1, penalize_diagonal=False):

    assert type(S) == np.ndarray and S.dtype == "float64"

    # assert type(lambda1) == 'float' or (type(data) == np.ndarray and data.shape == lambda1.shape)

    if type(lambda1) == float:
        lambda1 = np.full_like(S, lambda1, dtype="float64")
    
    assert type(lambda1) == np.ndarray and lambda1.dtype == "float64"
    assert check_symmetric(lambda1)

    if not penalize_diagonal:
        np.fill_diagonal(lambda1, 0)
    print(lambda1)

    return _cc.ccista(S, lambda1, lambda2, epstol, maxitr, steptype)
    # return _cc.ccista(S, S)


def h1(X, S):

    # print(X)

    return -np.log(X.diagonal()).sum() + 0.5*np.matmul(X, np.matmul(S, X)).trace()

def grad_h1(X, S):

    W = np.matmul(S, X)
    G = - np.diag(1/X.diagonal()) + 0.5*(W + W.T)

    return G

def h2(X, lambda1):

    return (lambda1 * np.abs(X)).sum()

def subgrad_h2(X, lambda1):

    return lambda1 * np.sign(X)

def soft_threshold(mat, lambda1):

    return np.sign(mat)*np.maximum(np.abs(mat) - lambda1, 0)

def Quad(mat_next, mat, S, tau):

    step = mat_next - mat
    Q = h1(mat, S) + np.matmul(step.T, grad_h1(mat, S)).trace() + (0.5/tau)*np.linalg.norm(step, "fro")**2

    return Q

def pyccista(S, lambda1, lambda2=0.0, epstol=1e-5, maxitr=100, steptype=1, penalize_diagonal=False):

    assert check_symmetric(S)

    if type(lambda1) == float:
        lambda1 = np.full_like(S, lambda1, dtype="float64")

    if penalize_diagonal is False:
        np.fill_diagonal(lambda1, 0)

    assert check_symmetric(lambda1)

    p, _ = S.shape

    X = np.identity(p)

    run_info = []
    while True:
        tau = 1.0
        c = 0.5

        W = np.matmul(S, X)
        G = - np.diag(1/X.diagonal()) + 0.5*(W + W.T)

        inner_itr_count = 0
        while True:

            step = X - tau*G
            if np.any(step.diagonal() < 0):
                tau = c*tau
                continue

            Xn = soft_threshold(step, tau*lambda1)

            if h1(Xn, S) <= Quad(Xn, X, S, tau):
                break
            else:
                tau = c*tau
                inner_itr_count += 1

        subg = np.linalg.norm(grad_h1(Xn, S) + subgrad_h2(Xn, lambda1))/np.linalg.norm(Xn)
        h = h1(Xn, S) + h2(Xn, lambda1)

        itr_info = [[inner_itr_count, subg, h]]
        run_info += itr_info

        print(itr_info)

        if subg <= epstol or len(run_info) > maxitr:
            break
        else:
            X = Xn

    return Xn, np.array(run_info)