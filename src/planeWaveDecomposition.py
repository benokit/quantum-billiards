import numpy as np
import math as m

def baseAngles(a, b, n):
    alpha = np.linspace(a, b, (n + 1))
    alpha = alpha + (b - a) / (2 * n)
    return alpha[0:n]


##########################
# no reflection symmetry #
##########################

def fg_2pi(n, k0, w, wg, x, y, nx, ny):
    """
    Plane wave decomposition F and G for basis with no reflection symetry
    - x and y are arrays of coordinates of points on a boundary
    - w is a vector of integration weights area/n
    - wg is a vector of integration weights length/n * normal*r 		
    - k0 is a wave number
    - n is number of plane wave directions (alpha)
    """
    alpha = baseAngles(0, m.pi, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    arg = np.outer(vx, x) + np.outer(vy, y)
    S = np.sin(k0 * arg)
    C = np.cos(k0 * arg)
    B = np.concatenate((S, C))
    vx = np.concatenate((vx, vx))
    vy = np.concatenate((vy, vy))
    T = w * B
    F = np.matmul(T, np.transpose(B)) #tension matrix
    #print(F.shape)
    DB = np.concatenate(( C, - S))
    #print(len(DB))
    
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = DB * VX
    VY = np.reshape(np.repeat(vy, x.size), (vx.size, x.size))
    DBVY = DB * VY
    U = DBVX * nx + DBVY * ny  #Transposed boundary function
    TU = wg/(k0**2) * U      #apply weights

    G = np.matmul(TU, np.transpose(U))  #Normalization Matrix, boundary method
    return F, G

###################################
# reflection symmetry over x axis #
###################################

# even parity

def fg_pi_sym(n, k0, w, wg, x, y, nx, ny):
    """
    Plane wave decomposition F and G for basis with reflection symetry over x axis
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights (area/n)
        - wg is a vector of integration weights n*r 		
        - k0 is a wave number
        - n is number of plane wave directions (alpha)
    """
    alpha = baseAngles(0, m.pi/2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k0 * argP)
    SM = np.sin(k0 * argM)
    CP = np.cos(k0 * argP)
    CM = np.cos(k0 * argM)
    B = np.concatenate((SP - SM, CP - CM))
    vx = np.concatenate((vx, vx))
    vy = np.concatenate((vy, vy))
    T = w * B
    F = np.matmul(T, np.transpose(B)) #tension matrix
    #print(F.shape)
    DB = np.concatenate((CP + CM, -SP - SM))
    vx = np.concatenate((vx, vx))
    vy = np.concatenate((vy, vy))
    #print(len(DB))
    
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = DB * VX
    VY = np.reshape(np.repeat(vy, x.size), (vx.size, x.size))
    DBVY = DB * VY
    U = DBVX * nx + DBVY * ny  #Transposed boundary function
    TU = wg/(k0**2) * U      #apply weights

    G = np.matmul(TU, np.transpose(U))  #Normalization Matrix, boundary method
    return F, G

# two reflection symmetry axis versions
def fg_pi2_sym_sym(n, k0, w, wg, x, y, nx, ny):
    """
    Plane wave decomposition
    F and G for basis with reflection symetry over x and y axis
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights (area/n)
        - wg is a vector of integration weights n*r 		
        - k0 is a wave number
        - n is number of plane wave directions (alpha)
    """
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k0 * argP)
    SM = np.sin(k0 * argM)
    CP = np.cos(k0 * argP)
    CM = np.cos(k0 * argM)
    
    B = CP + CM
    T = w * B
    F = np.matmul(T, np.transpose(B)) #tension matrix
    
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = SP * VX + SM * VX
    VY = np.reshape(np.repeat(vy, y.size), (vy.size, y.size))
    DBVY = SP * VY - SM * VY
    U = DBVX * nx + DBVY * ny  #Transposed boundary function
    TU = wg/(k0**2) * U      #apply weights

    G = np.matmul(TU, np.transpose(U))  #Normalization Matrix boundary method
    return F, G


def fg_pi2_sym_asym(n, k0, w, wg, x, y, nx, ny):
    """
        Plane wave decomposition
    F and G for basis with reflection symetry over x and y axis
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights (2pi/n) 
        - k0 is a wave number
        - n is number of plane wave directions (alpha)
    """
    alpha = baseAngles(0, m.pi / 2, n)
    #print("BasisDim=%s"%n)
    #print("PointsDim=%s"%len(w))
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k0 * argP)
    SM = np.sin(k0 * argM)
    CP = np.cos(k0 * argP)
    CM = np.cos(k0 * argM)
    
    B = SP - SM
    #print("B=%sx%s"%B.shape)
    T = w * B
    F = np.matmul(T, np.transpose(B)) #tension matrix
    
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = -CP * VX + CM * VX
    #print("DBVX=%sx%s"%DBVX.shape)
    
    VY = np.reshape(np.repeat(vy, y.size), (vy.size, y.size))
    DBVY = -CP * VY - CM * VY
    #print("DBVY=%sx%s"%DBVY.shape)
    U = DBVX * nx + DBVY * ny  #Transposed boundary function
    
    #print("U=%sx%s"%U.shape)
    #print("wg=%s"%wg.shape)
    TU = wg/(k0**2) * U      #apply weights

    G = np.matmul(TU, np.transpose(U))  #Normalization Matrix boundary method
    return F, G


def fg_pi2_asym_sym(n, k0, w, wg, x, y, nx, ny):
    """
        Plane wave decomposition
    F and G for basis with reflection symetry over x and y axis
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights (2pi/n) 
        - k0 is a wave number
        - n is number of plane wave directions (alpha)
    """
    alpha = baseAngles(0, m.pi / 2, n)
    #print("BasisDim=%s"%n)
    #print("PointsDim=%s"%len(w))
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k0 * argP)
    SM = np.sin(k0 * argM)
    CP = np.cos(k0 * argP)
    CM = np.cos(k0 * argM)
    
    B = SP + SM
    #print("B=%sx%s"%B.shape)
    T = w * B
    F = np.matmul(T, np.transpose(B)) #tension matrix
    
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = -CP * VX - CM * VX
    #print("DBVX=%sx%s"%DBVX.shape)
    
    VY = np.reshape(np.repeat(vy, y.size), (vy.size, y.size))
    DBVY = -CP * VY + CM * VY
    #print("DBVY=%sx%s"%DBVY.shape)
    U = DBVX * nx + DBVY * ny  #Transposed boundary function
    
    #print("U=%sx%s"%U.shape)
    #print("wg=%s"%wg.shape)
    TU = wg/(k0**2) * U      #apply weights

    G = np.matmul(TU, np.transpose(U))  #Normalization Matrix boundary method
    return F, G

def fg_pi2_asym_asym(n, k0, w, wg, x, y, nx, ny):
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k0 * argP)
    SM = np.sin(k0 * argM)
    CP = np.cos(k0 * argP)
    CM = np.cos(k0 * argM)
    
    B = CP - CM
    T = w * B
    F = np.matmul(T, np.transpose(B)) #tension matrix
    
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = -SP * VX + SM * VX
    VY = np.reshape(np.repeat(vy, y.size), (vy.size, y.size))
    DBVY = -SP * VY - SM * VY
    U = DBVX * nx + DBVY * ny  #Transposed boundary function
    TU = wg/(k0**2) * U      #apply weights

    G = np.matmul(TU, np.transpose(U))  #Normalization Matrix boundary method
    return F, G

def eigvalsPWD(k0, F, G):
    """
        PWD tension eigenvalues
    """
    # eigenvalues and eigenvectors of F
    d, S = np.linalg.eigh(F)
    # indeces of relevant eigenvectors
    ind = (d / np.max(d)) > 0.5e-16 
    q = 1 / np.sqrt(d[ind])
    #print((n, n - q.size))
    S = S[:,ind]
    S = q * S
    G = np.transpose(S).dot(G).dot(S)
    mu= np.linalg.eigvalsh(G)
    ks = 1/mu[-1]
    return ks


def eigPWD(k0, F, G):
    """
        PWD tension eigenvectors
    """
    # eigenvalues and eigenvectors of F
    d, S = np.linalg.eigh(F)
    # indices of relevant eigenvectors
    ind = (d / np.max(d)) > 0.5e-16 
    q = 1 / np.sqrt(d[ind])
    #print((n, n - q.size))
    #print(q)
    S = S[:,ind]
    S = q * S
    G = np.transpose(S).dot(G).dot(S)
    mu, Z = np.linalg.eigh(G)
    lam0 = mu[-1]
    #print(lam0)
	#eigenvector of lam0
    Z = Z[:,-1]
	#transform back to original basis
    X = S.dot(Z)
    return X/np.sqrt(lam0)            
    
