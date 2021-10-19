import numpy as np
from scipy import special
import sys
sys.path.append("..")
from ..CoreModules import BasisFunction as bf
from ..CoreModules import Basis as ba

# circular waves

def fb_0(i,k,x,y,x0=0,y0=0):
    X = x-x0
    Y = y-y0
    r = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y,X) 
    idx = i
    return special.jv(idx, k*r)*np.cos(idx*angle)

def fb_1(i,k,x,y,x0=0,y0=0):
    X = x-x0
    Y = y-y0
    r = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y,X)
    idx = i+1 
    return special.jv(idx, k*r)*np.sin(idx*angle)

def fb_0_grad(i,k,x,y,x0=0,y0=0):
    X = x-x0
    Y = y-y0
    r = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y,X) 
    idx = i
    j = special.jv(idx, k*r)
    dj = special.jvp(idx, k*r)
    c = np.cos(idx*angle)
    s = np.sin(idx*angle)
    dx = dj*k*X/r*c + j*s*Y/(X**2+Y**2)*idx
    dy = dj*k*Y/r*c - j*s*X/(X**2+Y**2)*idx
    return dx, dy

def fb_1_grad(i,k,x,y,x0=0,y0=0):
    X = x-x0
    Y = y-y0
    r = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y,X) 
    idx = i+1
    j = special.jv(idx, k*r)
    dj = special.jvp(idx, k*r)
    c = np.cos(idx*angle)
    s = np.sin(idx*angle)
    dx = dj*k*X/r*s - j*c*Y/(X**2+Y**2)*idx
    dy = dj*k*Y/r*s + j*c*X/(X**2+Y**2)*idx
    return dx, dy

def fb_0_dk(i,k,x,y,x0=0,y0=0):
    X = x-x0
    Y = y-y0
    r = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y,X) 
    idx = i
    return r*special.jvp(idx, k*r)*np.cos(idx*angle)

def fb_1_dk(i,k,x,y,x0=0,y0=0):
    X = x-x0
    Y = y-y0
    r = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y,X) 
    idx = i+1
    return r*special.jvp(idx, k*r)*np.sin(idx*angle)

def make_FB_basis(pars = {"x0":0, "y0" : 0}):
    basis_functions = []
    fb_cos = bf.basis_function(fb_0, pars, 
                        gradient = fb_0_grad,
                        k_derivative = fb_0_dk)
    basis_functions.append(fb_cos)
    fb_sin = bf.basis_function(fb_1, pars, 
                        gradient = fb_1_grad,
                        k_derivative = fb_1_dk)
    basis_functions.append(fb_sin)
    return ba.basis(basis_functions)


def fb_ca(i,k,x,y, nu=1, x0=0, y0 = 0, phi0 = 0, sym = None):
    X = x-x0
    Y = y-y0
    r = np.sqrt(X**2 + Y**2)
    #r[r==0] = 1e-16
    angle = np.arctan2(Y,X) - phi0
    if sym == None:
        idx = i+1
    if sym == "odd":
        idx = 2*i+1
    if sym == "even":
        idx = 2*i + 2
    #angle = np.arcaos(np.sign(Y)*X/r)
    return special.jv(nu*idx, k*r)*np.sin(nu*idx*angle)

def fb_ca_grad(i,k,x,y, nu=1, x0=0, y0 = 0, phi0 = 0, sym = None):
    X = x-x0
    Y = y-y0
    r = np.sqrt(X**2 + Y**2)
    #r[r==0] = 1e-16
    angle = np.arctan2(Y,X) - phi0
    if sym == None:
        idx = i+1
    if sym == "odd":
        idx = 2*i+1
    if sym == "even":
        idx = 2*i + 2
    j = special.jv(idx*nu, k*r)
    dj = special.jvp(idx*nu, k*r)
    c = np.cos(nu*idx*angle)
    s = np.sin(nu*idx*angle)
    dx = dj*k*X/r*s - j*c*Y/(X**2+Y**2)*nu*idx
    dy = dj*k*Y/r*s + j*c*X/(X**2+Y**2)*nu*idx
    return dx, dy


def fb_ca_dk(i,k,x,y, nu=1, x0=0, y0 = 0, phi0 = 0, sym = None):
    X = x-x0
    Y = y-y0
    r = np.sqrt(X**2 + Y**2)
    #r[r==0] = 1e-16
    angle = np.arctan2(Y,X) - phi0
    if sym == None:
        idx = i+1
    if sym == "odd":
        idx = 2*i+1
    if sym == "even":
        idx = 2*i + 2
    s = np.sin(nu*idx*angle)
    return r*special.jvp(nu*idx, k*r)*s


def make_FBca_basis(par_list = [{"nu":1, "x0":0, "y0" : 0, "phi0" : 0 , "sym" : None}]):
    basis_functions = []
    for pars in par_list:
        fb_function = bf.basis_function(fb_ca, pars, 
                            gradient = fb_ca_grad,
                            k_derivative = fb_ca_dk)
        basis_functions.append(fb_function)
    return ba.basis(basis_functions)