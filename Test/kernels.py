import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy

### kernel ###
def rbf_kernel(pattern1, pattern2, kernel_width):
    size1 = pattern1.size()
    size2 = pattern2.size()

    G = torch.sum(pattern1*pattern1, 1).reshape(size1[0],1)
    H = torch.sum(pattern2*pattern2, 1).reshape(size2[0],1)

    Q = torch.tile(G, (1, size2[0]))
    R = torch.tile(H.T, (size1[0], 1))

    H = Q + R - 2* (pattern1@pattern2.T)
    H = torch.exp(-H/2/(kernel_width**2))

    return H

def laplace_kernel(pattern1, pattern2, kernel_width):
    size1 = pattern1.size()
    size2 = pattern2.size()
    
    H = torch.cdist(pattern1, pattern2, p=1)
    H = torch.exp(-H/kernel_width)

    return H

def rbf_kernel_weight(pattern1, pattern2, kernel_width, weight):
    
    size1 = pattern1.size()
    size2 = pattern2.size()
    
    pattern1_weight = weight*pattern1
    pattern2_weight = weight*pattern2
    
    G = torch.sum(pattern1_weight*pattern1_weight, 1).reshape(size1[0],1)
    H = torch.sum(pattern2_weight*pattern2_weight, 1).reshape(size2[0],1)

    Q = torch.tile(G, (1, size2[0]))
    R = torch.tile(H.T, (size1[0], 1))

    H = Q + R - 2* (pattern1_weight@pattern2_weight.T)
    H = torch.exp(-H/2/(kernel_width**2))

    return H

def lap_kernel_weight(pattern1, pattern2, kernel_width, weight):
    
    size1 = pattern1.size()
    size2 = pattern2.size()
    
    pattern1_weight = weight*pattern1
    pattern2_weight = weight*pattern2

    H = torch.cdist(pattern1_weight, pattern2_weight, p=1)
    H = torch.exp(-H/2/(kernel_width**2))

    return H

def kernel_midwidth_rbf(X,Y):
    
    n = len(X)
    # ----- width of X -----
    Xmed = X

    G = torch.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = torch.tile(G, (1, n) )
    R = torch.tile(G.T, (n, 1) )

    dists = Q + R - 2* (Xmed@Xmed.T)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = torch.sqrt( 0.5 * torch.median(dists[dists>0]))    
    width_x_max = torch.sqrt( 0.5 * torch.max(dists[dists>0]))

    # ----- width of Y -----
    Ymed = Y

    G = torch.sum(Ymed*Ymed, 1).reshape(n,1)
    Q = torch.tile(G, (1, n) )
    R = torch.tile(G.T, (n, 1) )

    dists = Q + R - 2* (Ymed@Ymed.T)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_y = torch.sqrt( 0.5 * torch.median(dists[dists>0]))
    width_y_max = torch.sqrt( 0.5 * torch.max(dists[dists>0]))
    
    return width_x, width_y, width_x_max, width_y_max

def kernel_midwidth_lap(X,Y):
    
    n = len(X)
    # ----- width of X -----
    Xmed = X

    dists = torch.cdist(Xmed,Xmed,p=1)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = torch.median(dists[dists>0])   
    width_x_max = torch.max(dists[dists>0])

    # ----- width of Y -----
    Ymed = Y

    dists = torch.cdist(Ymed,Ymed,p=1)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_y = torch.median(dists[dists>0])
    width_y_max = torch.max(dists[dists>0])
    
    return width_x, width_y, width_x_max, width_y_max