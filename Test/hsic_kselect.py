import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
import scipy.stats as stats
from scipy.stats import gamma
from Test.hsic_naive import IndpTest

import cupy as cp
import cupyx.scipy 

from Test.kernels import rbf_kernel, laplace_kernel, kernel_midwidth_rbf, kernel_midwidth_lap

"""
Module containing statistical independence tests with selecting kernels (predefined width parameters).
25 pairs gaussian kernels with fixed different bandwidths are implemented. (You can try more settings).
"""
__author__ = 'ryx'

class IndpTest_KSelect(IndpTest):
    """
    Independence test with selecting Gaussian/Laplace kernels (learnable width parameters).
    This test runs in O(T*n^2*(dx+dy)) time.
    T: the number to perform gradient descent
    n: the sample size
    dx,dy: the dimension of x,y
    
    H0: x and y are independence 
    H1: x and y are not independence
    """

    def __init__(self, X, Y, device, alpha=0.05, n_permutation=100, null_gamma = True, split_ratio = 0.5):
        """
        alpha: significance level 
        n_permutation: The number of times to simulate from the null distribution
            by permutations. Must be a positive integer.
        split_ratio: split ratio of samples (Train/all)
        """
        super(IndpTest_KSelect, self).__init__(X, Y, alpha)
        self.n_permutation = n_permutation
        self.null_gamma = null_gamma
        self.split_ratio = split_ratio
        self.device = device
    
    def split_samples(self):
        """
        split datasets into train/test datasets
        """
        n = len(self.X)
        p = np.random.permutation(n)
        tr_size = int(n*self.split_ratio)
        ind_train = p[:tr_size]
        ind_test = p[tr_size:]
        
        Xtr = self.X[ind_train]
        Ytr = self.Y[ind_train]
        Xte = self.X[ind_test]
        Yte = self.Y[ind_test]
        
        if len(Xtr.size())==1:
            Xtr = Xtr.reshape(-1,1)
        if len(Ytr.size())==1:
            Ytr = Ytr.reshape(-1,1)
        if len(Xte.size())==1:
            Xte = Xte.reshape(-1,1)
        if len(Yte.size())==1:
            Yte = Yte.reshape(-1,1)
        
        return Xtr, Ytr, Xte, Yte

    def perform_test(self, debug = -1, n_gwidth_cand = 10):
        """
        Perform the independence test and return values computed in a dictionary.
        if_grid_search: if use grid_search for widths to init before perform optimize.
        debug: if >0: then print details of the optimization trace. 
        """
        
        ### split the datasets ###
        Xtr, Ytr, Xte, Yte = self.split_samples()
        
        ### predefined kernel list###
        wx_mid_rbf, wy_mid_rbf, _, _ = self.midwidth_rbf(Xtr, Ytr)
        gwidth_factors = 2.0**np.linspace(-1, 1, n_gwidth_cand) 

        wx_list = []
        wy_list = []

        for fac in gwidth_factors:
            wx_list.append(fac*wx_mid_rbf)
            wy_list.append(fac*wy_mid_rbf)
        
        Xtr = Xtr.to(self.device)    
        Ytr = Ytr.to(self.device)
        
        bx_tensor, by_tensor = self.kernel_selection(Xtr, Ytr, wx_list, wy_list, debug = debug) 
        bx = bx_tensor.detach().cpu()
        by = by_tensor.detach().cpu()
        
        if self.null_gamma == True:
            K, L = self.cal_kernels(Xte, Yte, wx_list, wy_list, bx, by)
            Kc = K - torch.mean(K,0)
            Lc = L - torch.mean(L,1)
            testStat,_ = self.J_maxpower_term(K, L, Kc, Lc)
            thresh,_,_ = self.cal_thresh(K, L, Kc, Lc)
        else:
            K, L = self.cal_kernels(Xte, Yte, wx_list, wy_list, bx, by)
            Kc = K - torch.mean(K,0)
            Lc = L - torch.mean(L,1)
            testStat,_ = self.J_maxpower_term(K, L, Kc, Lc)
            thresh = self.cal_thresh_pm(Xte, Yte, wx_list, wy_list, bx, by)  
        
        h0_rejected = (testStat>thresh)
        
        results_all = {}
        results_all["alpha"] = self.alpha
        results_all["thresh"] = thresh
        results_all["test_stat"] = testStat
        results_all["h0_rejected"] = h0_rejected
        
        return results_all
    
    def kernel_selection(self, X, Y, wx_list, wy_list, lr = 0.05, \
                            delta_estimate_grad=1e-3, iter_steps = 100, debug = -1):
        
        size1 = len(wx_list)
        size2 = len(wy_list)
        
        use_gpu = False
        if X.device.type == "cuda":
            use_gpu = True
            device = X.device

        if use_gpu:
            betax = torch.tensor([0.0] * size1, requires_grad=True, device = device)
            betay = torch.tensor([0.0] * size2, requires_grad=True, device = device)
        else:
            betax = torch.tensor([0.0] * size1, requires_grad=True)
            betay = torch.tensor([0.0] * size2, requires_grad=True)

        optimizer = optim.Adam([betax, betay], lr=lr)

        delta = delta_estimate_grad

        # path = np.zeros((iter_steps,3))

        for st in range(iter_steps):
            optimizer.zero_grad()

            weightx = torch.exp(betax)
            weighty = torch.exp(betay)
            bx = weightx/torch.sum(weightx)
            by = weighty/torch.sum(weighty)
            
            K = torch.zeros([len(X), len(X)], device = X.device)
            L = torch.zeros([len(Y), len(Y)], device = Y.device)
            
            for iwx in range(len(wx_list)):
                bxi = bx[iwx]
                wx_rbf = wx_list[iwx]
                K_rbf = rbf_kernel(X, X, wx_rbf)
                K += bxi* K_rbf
                
            for iwy in range(len(wy_list)):
                byi = by[iwy]
                wy_rbf = wy_list[iwy]
                L_rbf = rbf_kernel(Y, Y, wy_rbf) 
                L += byi* L_rbf

            Kc = K - torch.mean(K,0)
            Lc = L - torch.mean(L,1)

            testStat, sigma_estimate_reg = self.J_maxpower_term(K, L, Kc, Lc, lamb_reg = 1e-10)

            #--------------calculate thresh ----------------#
            al, bet = self.cal_thresh_param(K,L,Kc,Lc)
            if use_gpu:
                r0, al_detach, bet_detach, grad_al, grad_bet = self.cal_thresh_gamma_gpu(al, bet, if_grad = True)
            else:
                r0, al_detach, bet_detach, grad_al, grad_bet = self.cal_thresh_gamma(al, bet, if_grad = True)

            #--------------calculate power criterion ----------------#
            J0 = (testStat-r0)/sigma_estimate_reg
            J0_value = -J0.detach()
            sigma_estimate_reg_detach = sigma_estimate_reg.detach()
            
            #--------------calculate the grad of power criterion ----------------#
            J = J0 - (grad_al*al+ grad_bet*bet)/sigma_estimate_reg_detach

            (-J).backward()

            if debug > 0:
                if st%debug==0:
                    print(J0_value.item(), testStat.item(),r0.item())
                    print("beta", [bx,by])
                    
            optimizer.step()

        return bx, by

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

        return thresh, al, bet
    
    def cal_thresh_param(self, K, L, Kc, Lc):
        """
        Compute the parameter (of gamma distribution). 
        """
        n = len(K)

        varHSIC = (Kc * Lc / 6)**2
        varHSIC = (torch.sum(varHSIC) - torch.trace(varHSIC)) / n / (n-1)
        varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

        K_nondiag = K - torch.diag(torch.diag(K))
        L_nondiag = L - torch.diag(torch.diag(L))

        muX = torch.sum(K_nondiag) / n / (n-1)
        muY = torch.sum(L_nondiag) / n / (n-1)

        mHSIC = (1 + muX * muY - muX - muY) / n

        al = (mHSIC**2 / varHSIC)
        bet = (varHSIC*n / mHSIC)

        return al, bet

    def cal_thresh_gamma(self, al, bet, if_grad = False, delta = 1e-6):
        """
        Compute the thresh and parameter (of gamma distribution). 
        if_grad: if need to obtain the gradient of thresh.
        """
        al = al.detach().numpy()
        bet = bet.detach().numpy()

        thresh = gamma.ppf(1-self.alpha, al, scale=bet)

        if if_grad == True:
            thresh_al = gamma.ppf(1-self.alpha, al+delta, scale=bet) 
            thresh_bet = gamma.ppf(1-self.alpha, al, scale=bet+delta) 
            grad_al = (thresh_al - thresh)/delta
            grad_bet = (thresh_bet - thresh)/delta

            return thresh, al, bet, grad_al, grad_bet

        return thresh, al, bet

    def cal_thresh_gamma_gpu(self, al, bet, if_grad = False, delta = 1e-6):
        """
        For GPU (cupyx is needed)
        Compute the thresh and parameter (of gamma distribution). 
        if_grad: if need to obtain the gradient of thresh.
        """
        al = al.detach()
        bet = bet.detach()

        thresh = torch.tensor(cupyx.scipy.special.gammaincinv(al,1-self.alpha))*bet

        if if_grad == True:
            thresh_al = torch.tensor(cupyx.scipy.special.gammaincinv(al+delta,1-self.alpha))*bet
            thresh_bet = torch.tensor(cupyx.scipy.special.gammaincinv(al,1-self.alpha))*(bet+delta)
            grad_al = (thresh_al - thresh)/delta
            grad_bet = (thresh_bet - thresh)/delta

            return thresh, al, bet, grad_al, grad_bet

        return thresh, al, bet
    
    def cal_kernels(self, X, Y, wx_list, wy_list, bx, by):
        """
        Calculate kernels
        """
        K = torch.zeros([len(X), len(X)], device = X.device)
        L = torch.zeros([len(Y), len(Y)], device = Y.device)

        for iwx in range(len(wx_list)):
            bxi = bx[iwx]
            wx_rbf = wx_list[iwx]
            K_rbf = rbf_kernel(X, X, wx_rbf)
            K += bxi* K_rbf

        for iwy in range(len(wy_list)):
            byi = by[iwy]
            wy_rbf = wy_list[iwy]
            L_rbf = rbf_kernel(Y, Y, wy_rbf) 
            L += byi* L_rbf
        
        return K, L
    
    def midwidth_lap(self, X, Y):
        """
        Calculate midwidth of Laplace kernels 
        (also return maxwidth that can be used to limit the range in learning kernels)
        
        Return 
        wx_mid, wy_mid, wx_max, wy_max
        """
        wx_mid, wy_mid, wx_max, wy_max = kernel_midwidth_lap(X, Y)
        
        return wx_mid, wy_mid, wx_max, wy_max
    
    def midwidth_rbf(self, X, Y):
        """
        Calculate midwidth of Laplace kernels 
        (also return maxwidth that can be used to limit the range in learning kernels)
        
        Return 
        wx_mid, wy_mid, wx_max, wy_max
        """
        wx_mid, wy_mid, wx_max, wy_max = kernel_midwidth_rbf(X, Y)
        
        return wx_mid, wy_mid, wx_max, wy_max
    
    def cal_thresh_pm(self, X, Y, wx_list, wy_list, bx, by):
        ind = []
        for _ in range(self.n_permutation):
            p = np.random.permutation(len(X))
            Xp = X[p]
            K, L = self.cal_kernels(Xp, Y, wx_list, wy_list, bx, by)
            Kc = K - torch.mean(K,0)
            Lc = L - torch.mean(L,1)
            s_p = self.compute_stat(K, L, Kc, Lc)
            ind.append(s_p)
        sort_statistic = np.sort(ind)
        ls = len(sort_statistic)
        thresh_p = sort_statistic[int((1-self.alpha)*ls)+1]
        return thresh_p
    
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