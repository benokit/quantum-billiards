import numpy as np
import sys
sys.path.append("..")
from ..CoreModules import Utils as ut
from ..CoreModules import Curve as cv
from ..CoreModules import Billiard as bil
from ..CoreModules import Spectrum as sp
from ..CoreModules import Wavefunctions as wf
from ..BasisModules import FourierBessel_cy as fb
from ..BasisModules import RealPlaneWaves_cy as rpw
from . import Geometry as geo



##def make_lemon_quarter(a, virtual_axes = True ):
## original definition -- for stadium
#def make_lemon_quarter(eps, virtual_axes = True ):    
#    params_1 = {"x0" : eps, "y0" : 1, "x1" : 0, "y1": 1 }
#    line = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params_1)
#    params_2 = {"angle" : np.pi/2, "x0" : eps}
#    circle = cv.curve(geo.circle_r, geo.circle_n, geo.circle_arc, **params_2)
#    params_3 = {"x0" : 0, "y0" : 1, "x1" : 0, "y1": 0 }
#    y_axis = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params_3, virtual=virtual_axes, symmetry=virtual_axes)
#    params_4 = {"x0" : 0, "y0" : 0, "x1" : 1+eps, "y1": 0 }
#    x_axis = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params_4, virtual=virtual_axes, symmetry=virtual_axes)
#    curves = [circle, line, y_axis, x_axis]
#    area = 0.25 * np.pi + eps
#
#    return bil.billiard(curves, area)


# lemon -- circle arc
def lemon_circle_r(t, R = 1, eps = 0.2, x0 = 0, y0 = 0, angle = 2*np.pi, shift = 0):
    shift = np.pi/2 - np.arccos(eps/R)
    x = x0 + R * np.cos(t*angle + shift)
    y = y0 + R * np.sin(t*angle + shift)
    return x, y

def lemon_circle_n(t, R = 1, eps = 0.2, angle = 2*np.pi, shift = 0, **kwargs):
    shift = np.pi/2 - np.arccos(eps/R)
    nx = np.cos(t*angle + shift)
    ny = np.sin(t*angle + shift)
    return nx, ny

def lemon_circle_arc(t, R = 1, angle = 2*np.pi, **kwargs):
    return  R * angle *t

def lemon_arc_end_y0(eps, R = 1):
    return (R - eps)

def lemon_arc_end_x0(eps, R = 1):
    return R * np.sin(np.arccos(eps/R))

def lemon_area(eps, R = 1):
    theta = np.arccos(eps/R)
    #return  0.5 * (R**2 * np.arccos(eps/R) - eps * R * np.sin(np.arccos(eps/R)))
    return 0.25 * R**2*(2*theta - np.sin(2*theta))

#used for weyl formula
def sym_length(eps, sym_x, sym_y):
    """Length of inner virtual curves taking into account the boundary conditions
    + for Dirichlet
    - for Neumann
    """
    L = 0
    if sym_x == "odd":
        L = L + lemon_arc_end_x0(eps)
    if sym_x == "even":
        L = L - lemon_arc_end_x0(eps)
    if sym_y == "odd":
        L = L + lemon_arc_end_y0(eps)
    if sym_y == "even":
        L = L - lemon_arc_end_y0(eps)
    return L

##redefined for lemon --- def make_lemon_quarter(a, virtual_axes = True ):
def make_lemon_quarter(eps, R = 1, virtual_axes = True ):    
##    params_1 = {"x0" : eps, "y0" : 1, "x1" : 0, "y1": 1 }
##    line = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params_1)
    params_2 = {"angle" : np.arccos(eps/R),"eps": eps, "y0": -eps}
    ## params_2 = {"angle" : np.pi/2, "x0" : eps}
    circle = cv.curve(lemon_circle_r, lemon_circle_n, lemon_circle_arc, **params_2)
    y0up = lemon_arc_end_y0(eps, R)
    params_3 = {"x0" : 0, "y0" : y0up, "x1" : 0, "y1": 0 }
    y_axis = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params_3, virtual=virtual_axes, symmetry=virtual_axes)
    x0right = lemon_arc_end_x0(eps, R)
    params_4 = {"x0" : 0, "y0" : 0, "x1" : x0right, "y1": 0 }
    x_axis = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params_4, virtual=virtual_axes, symmetry=virtual_axes)
    curves = [circle, y_axis, x_axis]
    area = lemon_area(eps, R)

    return bil.billiard(curves, area)


def make_solvers(eps, sym_x = "odd", sym_y = "odd", basis_type = "rpw", scale_basis = 3, min_size = 200):
    
    billiard = make_lemon_quarter(eps, virtual_axes=True)
    
    if basis_type == "rpw":
        basis = rpw.make_RPW_basis(sym_x = sym_x, sym_y = sym_y)
    elif basis_type == "fb":
        basis = fb.make_FBq_basis(sym_x, sym_y)
    evp = sp.spectrum(billiard, basis)
    waf = wf.wavefunctions(billiard, basis, scale_basis=scale_basis, sym_x=sym_x, sym_y=sym_y)
    return evp, waf
