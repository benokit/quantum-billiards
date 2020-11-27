import numpy as np
import scipy
from . import Utils as ut
import matplotlib.pyplot as plt




def plot_matrix(A, cmap='RdBu'):
    ax = plt.gca() #plt.axes(xlim=(-1-eps-0.05, 1+eps+0.05), ylim=(-1-eps, 1+eps))
    #ax.axis('off')
    #indices = [i for i in range(n)]
    ax.set_aspect('equal', 'box')
    absmax = np.max(np.abs(A))
    ax.imshow(A, cmap = cmap, vmin = -absmax, vmax = absmax)

def GSVD(A, B, eps = False):
    n, p = np.shape(A)
    m, p = np.shape(B)
    #print("n=%s , m=%s, p=%s "%(n,m,p))
    #concatenate rows
    D = np.concatenate((A,B))
    #print(np.shape(D))
    #qr decomp of D
    Q, R = np.linalg.qr(D)
    if eps:
        Q1 = Q[:n]
        Q2 = Q[n:]
        Ur,Sr,tVr =  scipy.linalg.svd(R, full_matrices=True, lapack_driver='gesvd')
        #V = tVr.T
        #Vr1 = V[Sr>eps]
        ind = (Sr / np.max(Sr)) > eps 
        Ur1 = Ur[ :, ind ]
        #print(Sr)
        #print(ind)
        #print(np.shape(Ur))
        #print(np.shape(Ur1))
        Q1 = np.matmul(Q1,Ur1)
        #print(np.shape(Q1))
        Q2 = np.matmul(Q2,Ur1)
    #split Q 
    else:
        Q1 = Q[:n]
        #print(np.shape(Q1))
        Q2 = Q[n:]
    U, S, iX1 = scipy.linalg.svd(Q1, full_matrices=True, lapack_driver='gesvd')
    W, C, iX2 = scipy.linalg.svd(Q2, full_matrices=True, lapack_driver='gesvd')
    X = scipy.linalg.inv(iX2) #propper ordering?
    return  S[::-1], C, X #reverse S to get correct ordering

def particular_solutions_method(k, basis, bnd_pts, int_pts, eps = False, return_vector = False, show_matrix=False):
    """bnd_pts - boundary points (x[], y[]), 
       int_pts - interior points (x[], y[]) """
    #evaluate points on boundary
    bnd_x, bnd_y = bnd_pts.x, bnd_pts.y
    int_x, int_y = int_pts.x, int_pts.y   
    #construct fredholm matrix
    B = basis.evaluate_basis(k, bnd_x, bnd_y).T
    Bi = basis.evaluate_basis(k, int_x, int_y).T
    #T = weights * B
    #F = np.matmul(T, np.transpose(B))
    if show_matrix:
        plt.subplot(1,2,1)
        plot_matrix(B)
        plt.subplot(1,2,2)
        plot_matrix(Bi)
        plt.tight_layout()

    #generalized singular value decomposition
    S,C,X = GSVD(B, Bi, eps = eps)
    tension = (S[0]/C[0])**2
    if return_vector:
        vec = X[0]
        print (np.shape(X))
        return tension, vec
    else:
        return tension


def decomposition_method(k, basis, bnd_pts, length, area, eps = 0.5e-15, return_vector = False, show_matrix=False):
    #evaluate points on boundary
    bnd_x, bnd_y, nx, ny, weights = bnd_pts.x, bnd_pts.y, bnd_pts.nx, bnd_pts.ny, bnd_pts.ds
    #weights = ut.integration_weights(bnd_s, billiard.length)
    rn = (bnd_x * nx + bnd_y * ny)
    u_weights = weights/length * rn /(2*k**2) #changed *area !!!
    
    #construct fredholm matrix
    B = basis.evaluate_basis(k, bnd_x, bnd_y)
    T = weights * B
    F = np.matmul(T, np.transpose(B))

    #construct normalization matrix
    U = basis.evaluate_u(k, bnd_x, bnd_y, nx, ny)  #Transposed boundary function
    TU = u_weights * U      #apply weights

    G = np.matmul(TU, np.transpose(U))  #Normalization Matrix, boundary method

    if show_matrix:
        plt.subplot(1,2,1)
        plot_matrix(F)
        plt.subplot(1,2,2)
        plot_matrix(G)
        plt.tight_layout()
    # eigenvalues and eigenvectors of F
    d, S = np.linalg.eigh(F)
    if eps:
        # indices of relevant eigenvectors
        ind = (d / np.max(d)) > eps 
        q = 1 / np.sqrt(d[ind])
        #print((n, n - q.size))
        #print(q)
        S = S[:,ind]
        S = q * S
    G = np.transpose(S).dot(G).dot(S)


    if return_vector:
        mu, Z = np.linalg.eigh(G)
        lam0 = mu[-1]
        #print(lam0)
        #eigenvector of lam0
        Z = Z[:,-1]
        #transform back to original basis
        X = S.dot(Z)
        tension = 1/lam0
        vec = X/np.sqrt(lam0)
        return tension, vec
    else:
        mu= np.linalg.eigvalsh(G)
        tension = 1/mu[-1]
    return tension

def scaling_method(k, dk, basis, bnd_pts, eps = 0.5e-15, return_vector = False, show_matrix=False):
    #evaluate points on boundary
    bnd_x, bnd_y, nx, ny, weights = bnd_pts.x, bnd_pts.y, bnd_pts.nx, bnd_pts.ny, bnd_pts.ds
    rn = (bnd_x * nx + bnd_y * ny)
    sm_weights = weights / rn
    
    #construct fredholm matrix
    B = basis.evaluate_basis(k, bnd_x, bnd_y)
    T = sm_weights * B
    F = np.matmul(T, np.transpose(B))
    

    DB = basis.evaluate_df_dk(k, bnd_x, bnd_y)
    Fk = np.matmul(T, np.transpose(DB))
    Fk = Fk + np.transpose(Fk)
    if show_matrix:
        plt.subplot(1,2,1)
        plot_matrix(F)
        plt.subplot(1,2,2)
        plot_matrix(Fk)
        plt.tight_layout()

    if return_vector:
        d, S = np.linalg.eigh(F)
        # indices of relevant eigenvectors
        if eps:
            ind = (d / np.max(d)) > eps
            q = 1 / np.sqrt(d[ind])
            #print((n, n - q.size))
            S = S[:,ind]
            S = q * S
        Fk = np.transpose(S).dot(Fk).dot(S)
        mu, Z = np.linalg.eigh(Fk)
        ks = k - 2 / mu
        
        ind = np.abs(ks - k) <= dk
        ks = ks[ind]
        ten = 2*(2 / mu[ind])**2
        Z = Z[:,ind]
        ind = np.argsort(ks)
        ks = ks[ind]
        ten = ten[ind]
        Z = Z[:,ind]
        X = S.dot(Z)
        return ks, ten, X
    else:
        d, S = np.linalg.eigh(F)
        # indeces of relevant eigenvectors
        if eps:
            ind = (d / np.max(d)) > eps 
            q = 1 / np.sqrt(d[ind])
            S = S[:,ind]
            S = q * S
        Fk = np.transpose(S).dot(Fk).dot(S)
        mu = np.linalg.eigvalsh(Fk)
        ks = k - 2 / mu
        ind = np.abs(ks - k) <= dk
        ks = ks[ind]
        ten = 2*(2 / mu[ind])**2
        ind = np.argsort(ks)
        ks = ks[ind]
        ten = ten[ind]
        return ks, ten