import numpy as np
import sys
sys.path.append("..")
from ..CoreModules import Utils as ut
from ..CoreModules import Curve as cv
from ..CoreModules import Billiard as bil
from . import Geometry as geo

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def make_polygon(x, y, x0 = 0, y0 = 0, virtual = None):
    sz = len(x)
    #print(sz)
    curves = []
    area = poly_area(x,y)
    for i in range(sz):
        #print(i)
        if i == sz-1:
            xA = x[i]
            xB = x[0]
            yA = y[i]
            yB = y[0]
        else:
            xA = x[i]
            xB = x[i+1]
            yA = y[i]
            yB = y[i+1]
            
        params = {"x0": xA - x0, "x1": xB-x0, "y0": yA-y0, "y1": yB-y0}
        if virtual is not None:
            line = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params, virtual = virtual[i])
        else:
            line = cv.curve(geo.line_r, geo.line_n, geo.line_arc, **params, virtual = False)
        curves.append(line)
    #print(len(curves))
    
    return bil.billiard(curves, area)