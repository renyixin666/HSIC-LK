from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
import scipy.stats as stats
from scipy.stats import gamma
from Test.kernels import rbf_kernel, laplace_kernel, kernel_midwidth_rbf, kernel_midwidth_lap

"""
Module containing statistical independence tests with fixed kernels.
"""

__author__ = 'ryx'

class IndpTest(object):
    """
    An abstract class for independence test with learnable kernels. 
    The test requires a paired dataset specified by giving X, Y (torch tensors) such that X.shape[0] = Y.shape[0] = n. 
    """
    def __init__(self, X, Y, alpha=0.05):
        """ 
        X: Torch tensor of size n x dx
        Y: Torch tensor of size n x dy
        alpha: significance level of the test
        """
        self.X = X
        self.Y = Y
        self.alpha = alpha

    @abstractmethod
    def perform_test(self):
        """
        Perform the independence test and return values computed in a
        dictionary:
        {
            alpha: 0.05, 
            thresh: 1.0, 
            test_stat: 2.3, 
            h0_rejected: True, 
        }

        All values in the returned dictionary. 
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self):
        """
        Compute the test statistic. 
        Return a scalar value.
        """
        raise NotImplementedError()

class IndpTest_naive(IndpTest):
    """
    Independence test with fixed kernels (midwidth parameters).
    Gaussian/Laplace kernels are implemented!
    
    This test runs in O(n^2 (dx+dy)) time
    n: the sample size
    dx,dy: the dimension of x,y

    H0: x and y are independence 
    H1: x and y are not independence

    """

    def __init__(self, X, Y, alpha=0.05, n_permutation=100, kernel_type="Gaussian", null_gamma = True):
        """
        alpha: significance level 
        n_permutation: The number of times to simulate from the null distribution
            by permutations. Must be a positive integer. Default: 100. 
        kernel_type: "Gaussian" or "Laplace"
        null_gamma: if using gamma approximate. Default: True.
        """
        super(IndpTest_naive, self).__init__(X, Y, alpha)
        self.n_permutation = n_permutation
        self.kernel_type = kernel_type
        self.null_gamma = null_gamma
        
        if self.kernel_type not in ["Gaussian", "Laplace"]:
            raise NotImplementedError()
    
    def perform_test(self):
        """
        Perform the independence test and return values computed in a dictionary.
        """
        if self.kernel_type == "Gaussian":
            wx, wy = self.midwidth_rbf(self.X, self.Y)
            if self.null_gamma == True:
                K, L = self.cal_kernels(self.X, self.Y, wx, wy)
                Kc = K - torch.mean(K,0)
                Lc = L - torch.mean(L,1)
                testStat = self.compute_stat(K, L, Kc, Lc)
                thresh = self.cal_thresh_gamma(K, L, Kc, Lc)
            else:
                K, L = self.cal_kernels(self.X, self.Y, wx, wy)
                Kc = K - torch.mean(K,0)
                Lc = L - torch.mean(L,1)
                testStat = self.compute_stat(K, L, Kc, Lc)
                thresh = self.cal_thresh_pm(self.X, self.Y, wx, wy)  
                
        elif self.kernel_type == "Laplace":
            wx, wy = self.midwidth_lap(self.X, self.Y)
            if self.null_gamma == True:
                K, L = self.cal_kernels(self.X, self.Y, wx, wy)    
                Kc = K - torch.mean(K,0)
                Lc = L - torch.mean(L,1)
                testStat = self.compute_stat(K, L, Kc, Lc)
                thresh = self.cal_thresh_gamma(K, L, Kc, Lc)
            else:
                K, L = self.cal_kernels(self.X, self.Y, wx, wy)
                Kc = K - torch.mean(K,0)
                Lc = L - torch.mean(L,1)
                testStat = self.compute_stat(K, L, Kc, Lc)
                thresh = self.cal_thresh_pm(self.X, self.Y, wx, wy)
        
        h0_rejected = (testStat>thresh)
        
        results_all = {}
        results_all["alpha"] = self.alpha
        results_all["thresh"] = thresh
        results_all["test_stat"] = testStat
        results_all["h0_rejected"] = h0_rejected

        return results_all
    
    def cal_thresh_pm(self, X, Y, wx, wy):
        ind = []
        for _ in range(self.n_permutation):
            p = np.random.permutation(len(X))
            Xp = X[p]
            K, L = self.cal_kernels(Xp, Y, wx, wy)
            Kc = K - torch.mean(K,0)
            Lc = L - torch.mean(L,1)
            s_p = self.compute_stat(K, L, Kc, Lc)
            ind.append(s_p)
        sort_statistic = np.sort(ind)
        ls = len(sort_statistic)
        thresh_p = sort_statistic[int((1-self.alpha)*ls)+1]
        return thresh_p
    
    def cal_thresh_gamma(self, K,L,Kc,Lc):
        """
        Compute the test thresh. 
        """
        n = len(K)

        varHSIC = (Kc * Lc / 6)**2
        varHSIC = (torch.sum(varHSIC) - torch.trace(varHSIC)) / n / (n-1)
        varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

        K = K - torch.diag(torch.diag(K))
        L = L - torch.diag(torch.diag(L))

        muX = torch.sum(K) / n / (n-1)
        muY = torch.sum(L) / n / (n-1)

        mHSIC = (1 + muX * muY - muX - muY) / n

        al = (mHSIC**2 / varHSIC).detach().numpy()
        bet = (varHSIC*n / mHSIC).detach().numpy()

        thresh = gamma.ppf(1-self.alpha, al, scale=bet)

        return thresh
    
    def compute_stat(self, K, L, Kc, Lc):
        """
        Compute the test statistic. 
        """
        n = len(K)

        S = 0
        for i in range(n):
            S += torch.dot(Kc[i,:],Lc[:,i])

        testStat = S / n

        return testStat
        
    def cal_kernels(self, X, Y, kernel_width_x, kernel_width_y):
        """
        Calculate kernels
        """
        if self.kernel_type == "Gaussian":
            K = rbf_kernel(X, X, kernel_width_x)
            L = rbf_kernel(Y, Y, kernel_width_y)
        elif self.kernel_type == "Laplace":
            K = laplace_kernel(X, X, kernel_width_x)
            L = laplace_kernel(Y, Y, kernel_width_y)
        
        return K, L
        
    def midwidth_rbf(self, X, Y):
        """
        Calculate midwidth of Gaussian kernels 
        Return wx_mid, wy_mid
        """
        wx_mid, wy_mid, _, _ = kernel_midwidth_rbf(X, Y)
        
        return wx_mid, wy_mid
    
    def midwidth_lap(self, X, Y):
        """
        Calculate midwidth of Laplace kernels 
        Return wx_mid, wy_mid
        """
        wx_mid, wy_mid, _, _ = kernel_midwidth_lap(X, Y)
        
        return wx_mid, wy_mid