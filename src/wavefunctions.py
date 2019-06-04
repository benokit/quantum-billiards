import numpy as np
import math as m

def baseAngles(a, b, n):
    alpha = np.linspace(a, b, (n + 1))
    alpha = alpha + (b - a) / (2 * n)
    return alpha[0:n]

##################################
#  wavefunctions and gradients   #
##################################

##########################
# no reflection symmetry #
##########################

def psi_2pi(k, vec, x, y):
    """
    Value of a wave function composed from a 
    basis without a discrete symmetry (sin(kr), cos(kr))
    - x and y are arrays of coordinates of evaluation points
    - vec is an array of basis expansion coefficients
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

def grad_psi_2pi(k, vec, x, y):
    """
    Gradient of a wave function composed from a 
    basis without a discrete symmetry (sin(kr), cos(kr))
    - x and y are arrays of coordinates of evaluation points
    - vec is an array of basis expansion coefficients
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
    vx = np.concatenate((vx, vx))
    vy = np.concatenate((vy, vy)) 
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = DB * VX
    dpsi_x = np.transpose(DBVX).dot(vec)
    VY = np.reshape(np.repeat(vy, x.size), (vy.size, x.size))
    DBVY = DB * VY
    dpsi_y = np.transpose(DBVY).dot(vec)
    return dpsi_x, dpsi_y

###################################
# reflection symmetry over x axis #
###################################

#even parity

def psi_pi_sym(k, vec, x, y):
    """
    Value of a wave function composed from a basis with
    even parity against reflection over x axis
    - x and y are arrays of coordinates of evaluation points 
    - vec is an array of basis expansion coefficients
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

def grad_psi_pi_sym(k, vec, x, y):
    """
    Gradient of a wave function composed from a basis with 
    even parity against reflection over x axis
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
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
    vx = np.concatenate((vx, vx))
    vy = np.concatenate((vy, vy)) #check if correct
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = DB * VX
    dpsi_x = np.transpose(DBVX).dot(vec)
    VY = np.reshape(np.repeat(vy, y.size), (vy.size, y.size))
    DBVY = DB * VY
    dpsi_y = np.transpose(DBVY).dot(vec)
    return dpsi_x, dpsi_y

# odd parity

def psi_pi_asym(k, vec, x, y):
    """
    Value of a wave function composed from a basis with
    odd parity against reflection over x axis
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
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

def grad_psi_pi_asym(k, vec, x, y):
    """
    Gradient of a wave function composed from a basis with
    odd parity against reflection over x axis
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
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
    vx = np.concatenate((vx, vx)) #expand to correct size
    vy = np.concatenate((vy, vy))
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = DB * VX
    dpsi_x = np.transpose(DBVX).dot(vec)
    VY = np.reshape(np.repeat(vy, y.size), (vy.size, y.size))
    DBVY = DB * VY
    dpsi_y = np.transpose(DBVY).dot(vec)
    return dpsi_x, dpsi_y

##############################################
# reflection symmetry over x axis and y axis #
##############################################

# even, even parity (with respect to x and y axis)

def psi_pi2_sym_sym(k, vec, x, y):
    """
    Value of a wave function composed from a basis with
    even parity against reflection over x axis and
    even parity against reflection over y axis  
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
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

def grad_psi_pi2_sym_sym(k, vec, x, y):
    """
    Gradient of a wave function composed from a basis with
    even parity against reflection over x axis and
    even parity against reflection over y axis  
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
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

# even, odd parity (with respect to x and y axis)

def psi_pi2_sym_asym(k, vec, x, y):
    """
    Value of a wave function composed from a basis with
    even parity against reflection over x axis and
    odd parity against reflection over y axis  
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
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

def grad_psi_pi2_sym_asym(k, vec, x, y):
    """
    Gradient of a wave function composed from a basis with
    even parity against reflection over x axis and
    odd parity against reflection over y axis  
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
    - k is a wave number
    """
    n = vec.size
    alpha = baseAngles(0, m.pi / 2, n)
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    argP = np.outer(vx, x) + np.outer(vy, y)
    argM = np.outer(vx, x) - np.outer(vy, y)
    #SP = np.sin(k * argP)
    #SM = np.sin(k * argM)
    CP = np.cos(k * argP)
    CM = np.cos(k * argM)
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = -CP * VX + CM * VX
    dpsi_x = np.transpose(DBVX).dot(vec)
    VY = np.reshape(np.repeat(vy, y.size), (vy.size, y.size))
    DBVY = -CP * VY - CM * VY
    dpsi_y = np.transpose(DBVY).dot(vec)
    return dpsi_x, dpsi_y # -grad psi

# odd, even parity (with respect to x and y axis)

def psi_pi2_asym_sym(k, vec, x, y):
    """
    Value of a wave function composed from a basis with
    odd parity against reflection over x axis and
    even parity against reflection over y axis  
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
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

def grad_psi_pi2_asym_sym(k, vec, x, y):
    """
    Gradient of a wave function composed from a basis with
    odd parity against reflection over x axis and
    even parity against reflection over y axis  
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
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
    VX = np.reshape(np.repeat(vx, x.size), (vx.size, x.size))
    DBVX = -CP * VX - CM * VX
    dpsi_x = np.transpose(DBVX).dot(vec)
    VY = np.reshape(np.repeat(vy, y.size), (vy.size, y.size))
    DBVY = -CP * VY + CM * VY
    dpsi_y = np.transpose(DBVY).dot(vec)
    return dpsi_x, dpsi_y

# odd, odd parity (with respect to x and y axis)

def psi_pi2_asym_asym(k, vec, x, y):
    """
    Value of a wave function composed from a basis with
    odd parity against reflection over x axis and
    odd parity against reflection over y axis  
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
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

def grad_psi_pi2_asym_asym(k, vec, x, y):
    """
    Gradient of a wave function composed from a basis with
    odd parity against reflection over x axis and
    odd parity against reflection over y axis  
    - x and y are arrays of coordinates of evaluation points  
    - vec is an array of basis expansion coefficients
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
    DBVX = -SP * VX + SM * VX
    dpsi_x = np.transpose(DBVX).dot(vec)
    VY = np.reshape(np.repeat(vy, y.size), (vy.size, y.size))
    DBVY = -SP * VY - SM * VY
    dpsi_y = np.transpose(DBVY).dot(vec)
    return dpsi_x, dpsi_y	

