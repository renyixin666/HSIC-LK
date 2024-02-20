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

class IndpTest_kss(IndpTest):
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
        super(IndpTest_kss, self).__init__(X, Y, alpha)
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
            wx_mid, wy_mid = self.midwidth_rbf(self.X, self.Y)
            
            wx, wy = self.grid_search_init(self.X, self.Y, wx_mid, wy_mid)
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
    
    def grid_search_init(self, X, Y, wx_mid, wy_mid):
        """
        Using grid_search (log_scale) to init the widths (just the same as fsic)
        """
        n_gwidth_cand = 10
        gwidth_factors = 2.0**np.linspace(-1, 1, n_gwidth_cand) 

        width_pair = []
        J_pair = []

        for facx in gwidth_factors:
            for facy in gwidth_factors:
                wx = facx*wx_mid
                wy = facy*wy_mid

                K, L = self.cal_kernels(X, Y, wx, wy)
                Kc = K - torch.mean(K, 0)
                Lc = L - torch.mean(L, 1)

                thresh, al, bet = self.cal_thresh(K, L, Kc, Lc)
                testStat, sigma_estimate_reg = self.J_maxpower_term(K, L, Kc, Lc, lamb_reg = 1e-10)

                width_pair.append((wx,wy))
                J_pair.append((testStat - thresh)/sigma_estimate_reg)

        J_array = np.array(J_pair)
        indm = np.argmax(J_array)

        return width_pair[indm]
    
    def J_maxpower_term(self, K, L, Kc, Lc, lamb_reg = 1e-10):
        """
        Compute the terms for power criterion. 
        """
        n = len(K)

        S = 0
        for i in range(n):
            S += torch.dot(Kc[i,:],Lc[:,i])

        testStat = S / n

        A = torch.sum(K*L,0).reshape(-1,1)
        B = torch.sum(K,1).reshape(-1,1)
        C = torch.sum(L,1).reshape(-1,1)
        D = B*C
        h_i = 1/2*((n**2)*A+n*torch.sum(A)+torch.sum(C)*B+torch.sum(B)*C-n*(D+K@C+L@B)-torch.sum(D))/(n**3)

        var_estimate = 16*(torch.sum((h_i)**2)/n-((testStat)/n)**2)
        var_estimate_reg = var_estimate+lamb_reg

        sigma_estimate_reg = torch.sqrt(var_estimate_reg)

        return testStat, sigma_estimate_reg
    
    def cal_thresh(self, K, L, Kc, Lc):
        """
        Compute the test thresh and parameter (of gamma distribution). 
        """
        n = len(K)

        Kcc = Kc - torch.mean(Kc,1).reshape(-1,1)
        Lcc = Lc - torch.mean(Lc,1).reshape(-1,1)

        varHSIC = (Kcc * Lcc / 6)**2
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

        return thresh, al, bet
    
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

        Kcc = Kc - torch.mean(Kc,1).reshape(-1,1)
        Lcc = Lc - torch.mean(Lc,1).reshape(-1,1)

        varHSIC = (Kcc * Lcc / 6)**2
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