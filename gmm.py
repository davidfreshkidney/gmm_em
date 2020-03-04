import numpy as np
import numpy.linalg as la
import scipy.stats
from scipy.stats import multivariate_normal as mvn


def e_step(x, pi, mu, sigma):
    """
    @brief      Fix model parameters, update assignments
    @param      x      Samples represented as a numpy array of shape (n, d)
    @param      pi     Mixing coefficient represented as a numpy array of shape
                       (k,)
    @param      mu     Mean of each distribution represented as a numpy array of
                       shape (k, d)
    @param      sigma  Covariance matrix of each distribution represented as a
                       numpy array of shape (k, d, d)
    @return     The "soft" assignment of shape (n, k)
    """
    N, D = x.shape
    K = pi.shape[0]
    A = np.zeros((N, K))
    
    for k in range(K):
        A[:,k] = pi[k] * mvn.pdf(x, mu[k,:], sigma[k])
        
    return A/(np.sum(A,axis=1).reshape((N, 1)))


def m_step(x, a):
    """
    @brief      Fix assignments, update parameters
    @param      x     Samples represented as a numpy array of shape (n, d)
    @param      a     Soft assignments of each sample, represented as a numpy
                      array of shape (n, k)
    @return     A tuple (pi, mu, sigma), where
                - pi is the mixing coefficient represented as a numpy array of
                shape (k,)
                - mu is the mean of each distribution represented as a numpy
                array of shape (k, d)
                - sigma is the covariance matrix of each distribution
                represented as a numpy array of shape (k, d, d)
    """
    N,K = a.shape
    D = x.shape[1]
    
    pi = np.mean(a, axis=0)
    mu = (a.T @ x)/((N*pi).reshape((K, 1)))

    sigma = np.zeros((K,D,D))
    
    for k in range(K):
        x_center = x-mu[k]
        num = a[:,k] * x_center.T @ x_center
        sigma[k] = num/(N*pi[k])
    
    return pi, mu, sigma

