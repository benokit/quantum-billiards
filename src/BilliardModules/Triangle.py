import numpy as np
import sys
sys.path.append("..")
from ..CoreModules import Utils as ut
from ..CoreModules import Curve as cv
from ..CoreModules import Billiard as bil
from ..CoreModules import Spectrum as sp
from ..CoreModules import Wavefunctions as wf
from ..BasisModules import FourierBessel_cy as fb
#from ..BasisModules import RealPlaneWaves_cy as rpw
from . import Geometry as geo
from . import Polygon as poly


def triangle_corrners(gamma, chi , x0 = 0, y0 = 0, h = 1):
    #gamma is the obtuse angle
    #chi is the ratio between the other angles 
    alpha = (np.pi-gamma)/(1+chi)
    beta = alpha*chi
    #print(alpha)
    #print(beta)
    #print(np.pi-alpha/2-beta)
    
    xB, yB = - h/np.tan(beta+alpha)-x0, h-y0
    xA, yA = h/np.tan(alpha) + xB-x0, 0-y0
    xC, yC =  0-x0, 0-y0
    x = [xA, xB, xC]
    y = [yA, yB, yC]
    
    print(r"%s, %s, %s"%(alpha, beta, gamma))
    return np.array(x), np.array(y)


def make_solvers(gamma, chi, h = 1, basis_type = "fb", scale_basis = 1.5, min_size = 200, virual_edges_waf = False):
    x, y = triangle_corrners(gamma, chi, h = h)

    def angle(u,v):
        "Returns angle between vectors"
        x1, y1 = u
        x2, y2 = v
        dot = x1*x2 + y1*y2      # dot product between [x1, y1] and [x2, y2]
        det = x1*y2 - y1*x2 
        return np.arctan2(det, dot)
    
    def rotate(phi, x, y):
        x1 = x*np.cos(phi) - y*np.sin(phi)
        y1 = x*np.sin(phi) + y*np.cos(phi)
        return x1, y1
    
    idx = 2
    x = x - x[idx]
    y = y - y[idx]
    x = np.roll(x,-idx)
    y = np.roll(y,-idx)
    u = np.array([x[1],y[1]])
    v = np.array([x[2],y[2]])
    
    theta = angle(u,v)
    nu = np.pi/theta
    
    x_axis = np.array([1,0])
    phi0 = angle(u,x_axis)
    #print(x)
    x, y = rotate(phi0, x, y)
    fb_basis = fb.make_FBca_basis(par_list=[{"x0" : 0, "y0" : 0, "nu" : nu, "phi0" :0}])
    
    if virual_edges_waf:
        triangle_waf = poly.make_polygon(x,y, virtual = [True, False, True])
    else:
        triangle_waf = poly.make_polygon(x,y, virtual = [False, False, False])
    waf = wf.wavefunctions(triangle_waf, fb_basis, scale_basis=scale_basis)
    triangle_evp = poly.make_polygon(x,y, virtual = [True, False, True])
    evp = sp.spectrum(triangle_evp, fb_basis)
    return evp, waf