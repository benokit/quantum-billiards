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


def make_stadium_quarter(eps, virtual_axes = True ):
    params_1 = {"x0" : eps, "y0" : 1, "x1" : 0, "y1": 1 }
    line = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params_1)
    params_2 = {"angle" : np.pi/2, "x0" : eps}
    circle = cv.curve(geo.circle_r, geo.circle_n, geo.circle_arc, **params_2)
    params_3 = {"x0" : 0, "y0" : 1, "x1" : 0, "y1": 0 }
    y_axis = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params_3, virtual=virtual_axes, symmetry=virtual_axes)
    params_4 = {"x0" : 0, "y0" : 0, "x1" : 1+eps, "y1": 0 }
    x_axis = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params_4, virtual=virtual_axes, symmetry=virtual_axes)
    curves = [circle, line, y_axis, x_axis]
    area = 0.25 * np.pi + eps

    return bil.billiard(curves, area)

def make_solvers(eps, sym_x, sym_y, basis_type = "rpw", scale_basis = 3):
    stadium = make_stadium_quarter(eps, virtual_axes=True)
    if basis_type == "rpw":
        basis = rpw.make_RPW_basis(sym_x = sym_x, sym_y = sym_y)
    elif basis_type == "fb":
        basis = fb.make_FBq_basis(sym_x, sym_y)
    
    evp = sp.spectrum(stadium, basis)
    waf = wf.wavefunctions(stadium, basis, scale_basis=scale_basis, sym_x=sym_x, sym_y=sym_y)
    return evp, waf