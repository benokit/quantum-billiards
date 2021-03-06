import numpy as np
import math as m

def baseAngles(a, b, n):
    alpha = np.linspace(a, b, (n + 1))
    alpha = alpha + (b - a) / (2 * n)
    return alpha[0:n]

def psi_pi2_sym_asym(k, vec, x, y):
    """
        Value of a wave function composed from a 
        basis with a discrete symmetry over x and y axis (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = vec.size
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k * argP)
    SM = np.sin(k * argM)
    B = SP - SM
    psi = np.transpose(B).dot(vec)
    return psi

def psi_pi2_asym_sym(k, vec, x, y):
    """
        Value of a wave function composed from a 
        basis with a discrete symmetry over x and y axis (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = vec.size
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k * argP)
    SM = np.sin(k * argM)
    B = SP + SM
    psi = np.transpose(B).dot(vec)
    return psi

def psi_pi2_asym_asym(k, vec, x, y):
    """
        Value of a wave function composed from a 
        basis with a discrete symmetry over x and y axis (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = vec.size
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    CP = np.cos(k * argP)
    CM = np.cos(k * argM)
    B = CP - CM
    psi = np.transpose(B).dot(vec)
    return psi

def psi_pi2_sym_sym(k, vec, x, y):
    """
        Value of a wave function composed from a 
        basis with a discrete symmetry over x and y axis (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = vec.size
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    CP = np.cos(k * argP)
    CM = np.cos(k * argM)
    B = CP + CM
    psi = np.transpose(B).dot(vec)
    return psi

def psi_pi_asym(k, vec, x, y):
    """
        Value of a wave function composed from a 
        basis with a discrete symmetry over x axis (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = int(vec.size / 2)
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k * argP)
    SM = np.sin(k * argM)
    CP = np.cos(k * argP)
    CM = np.cos(k * argM)
    B = np.concatenate((SP - SM, CP - CM))
    psi = np.transpose(B).dot(vec)
    return psi

def psi_pi_sym(k, vec, x, y):
    """
        Value of a wave function composed from a 
        basis with a discrete symmetry over x axis (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = int(vec.size / 2)
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k * argP)
    SM = np.sin(k * argM)
    CP = np.cos(k * argP)
    CM = np.cos(k * argM)
    B = np.concatenate((SP + SM, CP + CM))
    psi = np.transpose(B).dot(vec)
    return psi
    

def psi_2pi(k, vec, x, y):
    """
        Value of a wave function composed from a 
        basis without a discrete symmetry (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = int(vec.size / 2)
    alpha = baseAngles(0, m.pi, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    arg = np.outer(vx, x) + np.outer(vy, y)
    S = np.sin(k * arg)
    C = np.cos(k * arg)
    B = np.concatenate((S, C))
    psi = np.transpose(B).dot(vec)
    return psi

def grad_psi_pi2_sym_sym(k, vec, x, y):
    """
        Gradient of a wave function composed from a 
        basis with a discrete symmetry over x and y axis (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = vec.size
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k * argP)
    SM = np.sin(k * argM)
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = SP * VX + SM * VX
    dpsi_x = np.transpose(DBVX).dot(vec)
    VY = np.reshape(np.repeat(vy, y.size), (vy.size, y.size))
    DBVY = SP * VY - SM * VY
    dpsi_y = np.transpose(DBVY).dot(vec)
    return dpsi_x, dpsi_y

def grad_psi_pi_asym(k, vec, x, y):
    """
        Gradient of a wave function composed from a 
        basis with a discrete symmetry over x axis (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = int(vec.size / 2)
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k * argP)
    SM = np.sin(k * argM)
    CP = np.cos(k * argP)
    CM = np.cos(k * argM)
    DB = np.concatenate((CP - CM, -SP + SM))
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = DB * VX
    dpsi_x = np.transpose(DBVX).dot(vec)
    VY = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVY = DB * VY
    dpsi_y = np.transpose(DBVY).dot(vec)
    return dpsi_x, dpsi_y

def grad_psi_pi_sym(k, vec, x, y):
    """
        Gradient of a wave function composed from a 
        basis with a discrete symmetry over x axis (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = int(vec.size / 2)
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    SP = np.sin(k * argP)
    SM = np.sin(k * argM)
    CP = np.cos(k * argP)
    CM = np.cos(k * argM)
    DB = np.concatenate((CP + CM, -SP - SM))
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = DB * VX
    dpsi_x = np.transpose(DBVX).dot(vec)
    VY = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVY = DB * VY
    dpsi_y = np.transpose(DBVY).dot(vec)
    return dpsi_x, dpsi_y

def grad_psi_2pi(k, vec, x, y):
    """
        Gradient of a wave function composed from a 
        basis without a discrete symmetry (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - vec is an array of base coefficients
        - k is a wave number
    """
    n = int(vec.size / 2)
    alpha = baseAngles(0, m.pi, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    arg = np.outer(vx, x) + np.outer(vy, y)
    S = np.sin(k * arg)
    C = np.cos(k * arg)
    DB = np.concatenate((C, -S))
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = DB * VX
    dpsi_x = np.transpose(DBVX).dot(vec)
    VY = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVY = DB * VY
    dpsi_y = np.transpose(DBVY).dot(vec)
    return dpsi_x, dpsi_y

def ffk_pi2_sym_asym(n, k0, w, x, y):
    """
        F and Fk for basis with reflection symmetry over x and y axis
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights 
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
    B = SP - SM
    DB = argP * CP - argM * CM
    T = w * B
    F = np.matmul(T, np.transpose(B))
    Fk = np.matmul(T, np.transpose(DB))
    Fk = Fk + np.transpose(Fk)
    return F, Fk

def ffk_pi2_asym_sym(n, k0, w, x, y):
    """
        F and Fk for basis with reflection symmetry over x and y axis
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights 
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
    B = SP + SM
    DB = argP * CP + argM * CM
    T = w * B
    F = np.matmul(T, np.transpose(B))
    Fk = np.matmul(T, np.transpose(DB))
    Fk = Fk + np.transpose(Fk)
    return F, Fk

def ffk_pi2_asym_asym(n, k0, w, x, y):
    """
        F and Fk for basis with reflection symmetry over x and y axis
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights 
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
    B = CP - CM
    DB = -argP * SP + argM * SM
    T = w * B
    F = np.matmul(T, np.transpose(B))
    Fk = np.matmul(T, np.transpose(DB))
    Fk = Fk + np.transpose(Fk)
    return F, Fk

def ffk_pi2_sym_sym(n, k0, w, x, y):
    """
        F and Fk for basis with reflection symmetry over x and y axis
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights 
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
    DB = -argP * SP - argM * SM
    T = w * B
    F = np.matmul(T, np.transpose(B))
    Fk = np.matmul(T, np.transpose(DB))
    Fk = Fk + np.transpose(Fk)
    return F, Fk

def ffk_pi_asym(n, k0, w, x, y):
    """
        F and Fk for basis with reflection symmetry over x axis
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights 
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
    B = np.concatenate((SP - SM, CP - CM))
    DB = np.concatenate((argP * CP - argM * CM, -argP * SP + argM * SM))
    T = w * B
    F = np.matmul(T, np.transpose(B))
    Fk = np.matmul(T, np.transpose(DB))
    Fk = Fk + np.transpose(Fk)
    return F, Fk

def ffk_pi_sym(n, k0, w, x, y):
    """
        F and Fk for basis with reflection symmetry over x axis
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights 
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
    B = np.concatenate((SP + SM, CP + CM))
    DB = np.concatenate((argP * CP + argM * CM, -argP * SP - argM * SM))
    T = w * B
    F = np.matmul(T, np.transpose(B))
    Fk = np.matmul(T, np.transpose(DB))
    Fk = Fk + np.transpose(Fk)
    return F, Fk

def ffk_2pi(n, k0, w, x, y):
    """
        F and Fk for basis without a discrete symmetry (sin(alpha), cos(alpha))
        - x and y are arrays of coordinates of points on a boundary
        - w is a vector of integration weights 
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
    DB = np.concatenate((arg * C, -arg * S))
    T = w * B
    F = np.matmul(T, np.transpose(B))
    Fk = np.matmul(T, np.transpose(DB))
    Fk = Fk + np.transpose(Fk)
    return F, Fk

def eigvals(k0, dk, F, Fk):
    """
        calculates eigenvalues on the interval [k0-dk, k0+dk] by
        solving generalized eigenvalue problem Fk - mu * F = 0
        originating from the vergini-saraceno method
    """
    # dimension
    n, _ = F.shape
    # eigenvalues and eigenvectors of F
    d, S = np.linalg.eigh(F)
    # indeces of relevant eigenvectors
    ind = (d / np.max(d)) > 1e-16 
    q = 1 / np.sqrt(d[ind])
    S = S[:,ind]
    S = q * S
    Fk = np.transpose(S).dot(Fk).dot(S)
    mu = np.linalg.eigvalsh(Fk)
    ks = k0 - 2 / mu;
    ind = np.abs(ks - k0) <= dk
    ks = ks[ind]
    ks = np.sort(ks)
    return ks

def eig(k0, dk, F, Fk):
    """
        calculates eigenvalues and eigenvectors on the interval [k0-dk, k0+dk] by
        solving generalized eigenvalue problem Fk - mu * F = 0
        originating from the vergini-saraceno method
    """
    # dimension
    n, _ = F.shape
    # eigenvalues and eigenvectors of F
    d, S = np.linalg.eigh(F)
    # indices of relevant eigenvectors
    ind = (d / np.max(d)) > 0.5e-16 
    q = 1 / np.sqrt(d[ind])
    print((n, n - q.size))
    S = S[:,ind]
    S = q * S
    Fk = np.transpose(S).dot(Fk).dot(S)
    mu, Z = np.linalg.eigh(Fk)
    ks = k0 - 2 / mu
    ind = np.abs(ks - k0) <= dk
    ks = ks[ind]
    Z = Z[:,ind]
    ind = np.argsort(ks)
    ks = ks[ind]
    Z = Z[:,ind]
    X = S.dot(Z)
    return ks, X
    
    
    
