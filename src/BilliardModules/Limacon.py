import numpy as np
import sys
from scipy import special
sys.path.append("..")
from ..CoreModules import Utils as ut
from ..CoreModules import Curve as cv
from ..CoreModules import Billiard as bil
from ..CoreModules import Spectrum as sp
from ..CoreModules import Wavefunctions as wf
from ..BasisModules import FourierBessel_cy as fb
from ..BasisModules import RealPlaneWaves_cy as rpw
from . import Geometry as geo


def limacon_r(t, lam = 0, x0 = 0.0):
    phi = t * np.pi
    x = np.cos(phi) * (1 + lam * np.cos(phi)) - x0
    y = np.sin(phi) * (1 + lam * np.cos(phi))
    return x, y

def limacon_n(t, lam = 0, x0 = 0.0):
    phi = t * np.pi
    #tangent
    tx = -np.sin(phi) - lam*np.sin(2*phi)
    ty = np.cos(phi) + lam*np.cos(2*phi)
    norm = np.sqrt(tx**2+ty**2)
    return ty/norm, -tx/norm

def limacon_arc(t, lam = 0, x0 = 0.0):
    phi = t * np.pi
    a = lam 
    return 2*(1+a)*special.ellipeinc(phi/2,4*a/(1+a)**2 )


def make_limacon_half(lam, virtual_axes = True, centered = True ):
    x0,y0 = limacon_r(0, lam = lam)
    x1,y1 = limacon_r(1, lam = lam)
    if centered:
        x_c = (x0 + x1)/2
        params_1 = {"lam" : lam, "x0" : x_c }
    else:
        x_c = 0.0
        params_1 = {"lam" : lam, "x0" : 0.0 }
    limacon_curve = cv.curve(limacon_r, limacon_n, limacon_arc, **params_1)

    params_4 = {"x0" : x0-x_c, "y0" : 0, "x1" : x1-x_c, "y1": 0 }
    x_axis = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params_4, virtual=virtual_axes, symmetry=virtual_axes)

    curves = [limacon_curve, x_axis]
    area = 0.5 * (1 + 0.5 * lam * lam) * np.pi
    
    return bil.billiard(curves, area)

def make_solvers(lam, sym_x, basis_type = "rpw", scale_basis = 3, centered = True):
    billiard = make_limacon_half(lam, virtual_axes = True, centered = centered )
    if basis_type == "rpw":
        basis = rpw.make_RPW_basis(sym_x = sym_x, sym_y = None)
       
    evp = sp.spectrum(billiard, basis)
    waf = wf.wavefunctions(billiard, basis, scale_basis=scale_basis, sym_x=sym_x, sym_y=None)
    return evp, waf