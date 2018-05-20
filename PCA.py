import numpy as np


def pca(X):
    """Principal Component Analysis

    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with most important dimensions first).
    """
    # get dimensions
    num_data, dim = X.shape

    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    M = np.dot(X,X.T) # covariance matrix, AA', not the A'A like usual
    e,EV = np.linalg.eigh(M) # eigenvalues and eigenvectors
    tmp = np.dot(X.T,EV).T # this is the compact trick
    V = tmp[::-1] # reverse since last eigenvectors are the ones we want
    S = np.sqrt(e[::-1]) # reverse since eigenvalues are in increasing order

    for i in range(V.shape[1]):
        V[:,i] /= S

    # return the projection matrix, the variance and the mean
    return V, S, mean_X
