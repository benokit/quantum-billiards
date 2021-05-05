import numpy as np
import sys
sys.path.append("..")
from ..CoreModules import Utils as ut
from ..CoreModules import Curve as cv
from ..CoreModules import Billiard as bil
from . import Geometry as geo


def make_lemon_quarter(a, virtual_axes = True ):
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