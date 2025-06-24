import numpy as np
import ot


def optimal_transport_plan(
    X, Y, cost_matrix, method="sinkhorn_gpu", niter=100, epsilon=0.01
):
    X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])  # uniform mask to every timestamp
    Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])
    transport_plan = ot.sinkhorn(X_pot, Y_pot, cost_matrix, epsilon, numItermax=niter)
    return transport_plan


def cosine_distance(x, y):
    dot = np.dot(x, y.T)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    norms = np.dot(x_norm, y_norm.T)
    cosine_sim = dot / (norms + 1e-8)
    return 1 - cosine_sim


def euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]  # [n, m, d]
    c = np.sqrt(np.sum(diff**2, axis=2))
    return c


def manhattan_distance(x, y):
    "returns L1 distances matrix  $|x_i-y_j|^p$."
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    return np.sum(np.abs(diff), axis=2)
