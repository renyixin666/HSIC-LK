from scipy.stats import ortho_group
import numpy

def generate_ISA(n,d,sigma_normal,alpha):
    
    x = np.concatenate((np.random.normal(-1, sigma_normal, n//2), np.random.normal(1, sigma_normal, n//2)))
    y = np.concatenate((np.random.normal(-1, sigma_normal, n//2), np.random.normal(1, sigma_normal, n//2)))
    p = np.random.permutation(n)
    y_p = y[p]

    D = np.zeros([2,n])
    D[0,:] = x
    D[1,:] = y_p

    theta = np.pi/4*alpha
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    D_R = R@D
    X_mix = D_R[0,:].reshape(-1,1)
    Y_mix = D_R[1,:].reshape(-1,1)

    X_z = np.random.randn(n,d-1)
    Y_z = np.random.randn(n,d-1)

    X_con = np.concatenate((X_mix,X_z), axis=1)
    Y_con = np.concatenate((Y_mix,Y_z), axis=1)

    m_x = ortho_group.rvs(dim=d)
    m_y = ortho_group.rvs(dim=d)

    X = (m_x@X_con.T).T
    Y = (m_y@Y_con.T).T
    
    return X,Y

def Sinusoid(x, y, w):
    return 1 + np.sin(w*x)*np.sin(w*y)

# n = 2000
# w = 4
def Sinusoid_Generator(n,w):
    i = 0
    output = np.zeros([n,2])
    while i < n:
        U = np.random.rand(1)
        V = np.random.rand(2)
        x0 = -np.pi + V[0]*2*np.pi
        x1 = -np.pi + V[1]*2*np.pi
        if U < 1/2 * Sinusoid(x0,x1,w):
            output[i, 0] = x0
            output[i, 1] = x1
            i = i + 1
    return output[:,0], output[:,1]

# n = 1000
# d = 5
def sinedependence(n,d):
    mean = np.zeros(d)
    cov = np.eye(d)
    X = np.random.multivariate_normal(mean, cov, n)
    Z = np.random.randn(n)
    Y = 20*np.sin(4*np.pi*(X[:,0]**2 + X[:,1]**2))+Z 
    return X,Y

# n = 1000
# d = 5
def GSign(n,d):
    mean = np.zeros(d)
    cov = np.eye(d)
    X = np.random.multivariate_normal(mean, cov, n)
    sign_X = np.sign(X)
    Z = np.random.randn(n)
    Y = np.abs(Z)*np.prod(sign_X,1)
    return X,Y
