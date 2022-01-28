import numpy as np

import sys
sys.path.append("..")
from ..cython_functions import Sin, Cos
from ..CoreModules import BasisFunction as bf
from ..CoreModules import Basis as ba

def RealPlaneWave(i,k,x,y, dphi = 0.1, angle = 0, phase = 0):
    alpha = angle + i*dphi
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    arg = vx*x + vy*y
    return Sin(k*arg + phase)

def RPW_gradient(i,k,x,y, dphi = 0.1, angle = 0, phase = 0):
    alpha = angle + i*dphi
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    arg = vx*x + vy*y
    grd = Cos(k*arg + phase)
    
    return k*vx*grd, k*vy*grd

def RPW_dk(i,k,x,y, dphi = 0.1, angle = 0, phase = 0):
    alpha = angle + i*dphi
    vx = np.cos(alpha)
    vy = np.sin(alpha)
    arg = vx*x + vy*y
    return arg*Cos(k*arg + phase)

#symmetry axis - x

def RPW_sym(i,k,x,y, dphi = 0.1, sym_x = "odd", sym_y = "odd"):
    alpha = (i+1)*dphi
    vx = np.cos(alpha)
    vy = np.sin(alpha)

    if sym_x == "odd" and sym_y == "odd":
        return Sin(k*vx*x)* Sin(k*vy*y)
        
    if sym_x == "odd" and sym_y == "even":
        return Cos(k*vx*x)* Sin(k*vy*y)

    if sym_x == "even" and sym_y == "odd":
        return Sin(k*vx*x)* Cos(k*vy*y)
        
    if sym_x == "even" and sym_y == "even":
        return Cos(k*vx*x)* Cos(k*vy*y)

def RPW_sym_grad(i,k,x,y, dphi = 0.1, sym_x = "odd", sym_y = "odd"):
    alpha = (i+1)*dphi
    vx = np.cos(alpha)
    vy = np.sin(alpha)

    if sym_x == "odd" and sym_y == "odd":
        dx = k*vx* Cos(k*vx*x)* Sin(k*vy*y)
        dy = k*vy* Sin(k*vx*x)* Cos(k*vy*y)
        return dx, dy
        
    if sym_x == "odd" and sym_y == "even":
        dx = -k*vx* Sin(k*vx*x)* Sin(k*vy*y)
        dy = k*vy* Cos(k*vx*x)* Cos(k*vy*y)
        return dx, dy

    if sym_x == "even" and sym_y == "odd":
        dx = k*vx* Cos(k*vx*x)* Cos(k*vy*y)
        dy = -k*vy* Sin(k*vx*x)* Sin(k*vy*y)
        return dx, dy
        
    if sym_x == "even" and sym_y == "even":
        dx = -k*vx* Sin(k*vx*x)* Cos(k*vy*y)
        dy = -k*vy* Cos(k*vx*x)* Sin(k*vy*y)
        return dx, dy

   

def RPW_sym_dk(i,k,x,y, dphi = 0.1, sym_x = "odd", sym_y = "odd"):
    alpha = (i+1)*dphi
    vx = np.cos(alpha)
    vy = np.sin(alpha)

    if sym_x == "odd" and sym_y == "odd":
        return vx*x*Cos(k*vx*x)*Sin(k*vy*y) + vy*y*Sin(k*vx*x)*Cos(k*vy*y)
     
    if sym_x == "odd" and sym_y == "even":
        return -vx*x*Sin(k*vx*x)*Sin(k*vy*y) + vy*y*Cos(k*vx*x)*Cos(k*vy*y)

    if sym_x == "even" and sym_y == "odd":
        return vx*x*Cos(k*vx*x)*Cos(k*vy*y) + vy*y*Sin(k*vx*x)*Sin(k*vy*y)

    if sym_x == "even" and sym_y == "even":
        return -vx*x*Sin(k*vx*x)*Cos(k*vy*y) - vy*y*Cos(k*vx*x)*Sin(k*vy*y)


#symmetry axis - x and y

def RPW_angles(i,n):
    dphi = np.pi/(n+1)
    return {"dphi":dphi}

def RPW_angles_sym_xy(i,n):
    dphi = np.pi/(2*(n+1))
    return {"dphi":dphi}

def make_RPW_basis(sym_x = None, sym_y = None, dist_fun = None):
    if sym_x == None and sym_y == None:
        if dist_fun == None:
            par_fun = RPW_angles
        else:
            par_fun = dist_fun
        sin = bf.basis_function(RealPlaneWave, { "phase" : 0, "angle" : 0}, 
                                par_fun = par_fun, gradient = RPW_gradient,
                                k_derivative = RPW_dk)
        cos = bf.basis_function(RealPlaneWave, { "phase" : np.pi/2, "angle" : 0}, 
                                par_fun = par_fun, gradient = RPW_gradient,
                                k_derivative = RPW_dk)
        basis_functions = [sin, cos]
        return ba.basis(basis_functions)
        
    # only x symmetry
    if sym_x == "odd" and sym_y == None:
        if dist_fun == None:
            par_fun = RPW_angles
        else:
            par_fun = dist_fun
        sin = bf.basis_function(RPW_sym, {"sym_x": "odd", "sym_y": "odd"}, 
                            par_fun = par_fun, gradient = RPW_sym_grad,
                            k_derivative = RPW_sym_dk)
        cos = bf.basis_function(RPW_sym, {"sym_x": "odd", "sym_y": "even"}, 
                            par_fun = par_fun, gradient = RPW_sym_grad,
                            k_derivative = RPW_sym_dk)
        basis_functions = [sin, cos]
        return ba.basis(basis_functions)
        
    if sym_x == "even" and sym_y == None:
        if dist_fun == None:
            par_fun = RPW_angles
        else:
            par_fun = dist_fun
        sin = bf.basis_function(RPW_sym, {"sym_x": "even", "sym_y": "odd"}, 
                            par_fun = par_fun, gradient = RPW_sym_grad,
                            k_derivative = RPW_sym_dk)
        cos = bf.basis_function(RPW_sym, {"sym_x": "even", "sym_y": "even"}, 
                            par_fun = par_fun, gradient = RPW_sym_grad,
                            k_derivative = RPW_sym_dk)
        basis_functions = [sin, cos]
        return ba.basis(basis_functions)
        

    # only y symmetry
    if sym_y == "odd" and sym_x == None:
        if dist_fun == None:
            par_fun = RPW_angles
        else:
            par_fun = dist_fun
        sin = bf.basis_function(RPW_sym, {"sym_x": "odd", "sym_y": "odd"}, 
                            par_fun = par_fun, gradient = RPW_sym_grad,
                            k_derivative = RPW_sym_dk)
        cos = bf.basis_function(RPW_sym, {"sym_x": "even", "sym_y": "odd"}, 
                            par_fun = par_fun, gradient = RPW_sym_grad,
                            k_derivative = RPW_sym_dk)
        basis_functions = [sin, cos]
        return ba.basis(basis_functions)
        

    if sym_y == "even" and sym_x == None:
        if dist_fun == None:
            par_fun = RPW_angles
        else:
            par_fun = dist_fun
        sin = bf.basis_function(RPW_sym, {"sym_x": "odd", "sym_y": "even"}, 
                            par_fun = par_fun, gradient = RPW_sym_grad,
                            k_derivative = RPW_sym_dk)
        cos = bf.basis_function(RPW_sym, {"sym_x": "even", "sym_y": "even"}, 
                            par_fun = par_fun, gradient = RPW_sym_grad,
                            k_derivative = RPW_sym_dk)
        basis_functions = [sin, cos]
        return ba.basis(basis_functions)
        
    # both x and y symmetry
    if sym_x == "odd" and sym_y == "odd":
        if dist_fun == None:
            par_fun = RPW_angles_sym_xy
        else:
            par_fun = dist_fun
        bas = bf.basis_function(RPW_sym, {"sym_x": "odd", "sym_y": "odd"}, 
                            par_fun = par_fun, gradient = RPW_sym_grad,
                            k_derivative = RPW_sym_dk)
        basis_functions = [bas]
        return ba.basis(basis_functions)
        
    if sym_x == "odd" and sym_y == "even":
        if dist_fun == None:
            par_fun = RPW_angles_sym_xy
        else:
            par_fun = dist_fun
        bas = bf.basis_function(RPW_sym, {"sym_x": "odd", "sym_y": "even"}, 
                    par_fun = par_fun, gradient = RPW_sym_grad,
                    k_derivative = RPW_sym_dk)
        basis_functions = [bas]
        return ba.basis(basis_functions)        

    if sym_x == "even" and sym_y == "odd":
        if dist_fun == None:
            par_fun = RPW_angles_sym_xy
        else:
            par_fun = dist_fun
        bas = bf.basis_function(RPW_sym, {"sym_x": "even", "sym_y": "odd"}, 
                    par_fun = par_fun, gradient = RPW_sym_grad,
                    k_derivative = RPW_sym_dk)
        basis_functions = [bas]
        return ba.basis(basis_functions)
        

    if sym_x == "even" and sym_y == "even":
        if dist_fun == None:
            par_fun = RPW_angles_sym_xy
        else:
            par_fun = dist_fun
        bas = bf.basis_function(RPW_sym, {"sym_x": "even", "sym_y": "even"}, 
                    par_fun = par_fun, gradient = RPW_sym_grad,
                    k_derivative = RPW_sym_dk)
        basis_functions = [bas]
        return ba.basis(basis_functions)
            

