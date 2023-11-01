import numpy as np
import networkx as nx
from utils import standardize, partial_corr, proj_precision_mat

def generate_erdos_renyi(p, type='proj', edge_prob=0.01, lower_weight=0.5, upper_weight=1.0, spread_diag=[1, np.sqrt(3)], random_state=2023):
    G = nx.generators.random_graphs.erdos_renyi_graph(n=p, p=edge_prob, seed=random_state)
    Skel = nx.to_numpy_array(G)

     # projection method
    if type == 'proj':
        np.random.seed(random_state)
        edge_weights = np.random.uniform(low=lower_weight, high=upper_weight, size=(p,p))
        edge_signs = np.random.choice([-1,1], size=(p,p))
        Theta = np.multiply(edge_weights, edge_signs)
        Theta = np.multiply(Skel, Theta)
        Theta = np.tril(Theta) + np.tril(Theta).T
        nz_indx = np.nonzero(Theta)
        for i in range(100):
            Theta = proj_precision_mat(Theta, nz_indx)
            if np.linalg.cond(Theta) < 20:
                break
        
        Theta = np.real(Theta)
        Sigma = np.linalg.inv(Theta)

    # Peng's method
    if type == 'peng':
        np.random.seed(random_state)
        edge_weights = np.random.uniform(low=lower_weight, high=upper_weight, size=(p,p))
        edge_signs = np.random.choice([-1,1], size=(p,p))
        Theta = np.multiply(edge_weights, edge_signs)
        Theta = np.multiply(Skel, Theta)
        Theta = np.tril(Theta) + np.tril(Theta).T
        np.fill_diagonal(Theta, 1.5*np.abs(Theta).sum(1))
        diag_inv = np.diag(1/np.sqrt(np.diag(Theta)))
        Theta = diag_inv @ Theta @ diag_inv

        # spread diagonal of precision matrix
        d = np.random.uniform(spread_diag[0], spread_diag[1], p)
        Theta = np.diag(d) @ Theta @ np.diag(d)
        Sigma = np.linalg.inv(Theta)

    return Theta, Sigma

def generate_data(p, n_prop_to_p, Sigma, N=1, random_state=2023):
    Xs = []
    for this in n_prop_to_p:
        n = int(this*p)
        for j in range(N):
            rs = np.random.RandomState(random_state)
            X = rs.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
            X_std = standardize(X, bias=False)
            Xs.append(X_std)
            random_state += 1

    return Xs