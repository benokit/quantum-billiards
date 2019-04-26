import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from ..src import billiardClass as bc
from ..src import curveClass as cv
from ..src import plottingFunctions as pf

# Our weird billiard will be constructed using line segments and circle arcs

# We first define functions that define the arc of a circle.
# The functions must be defined on the interval [0,1].
# Additional parameters are pased as key-word arguments.

# Three functions are required.
# The first returns the coordinate vector (x,y) of the curve at t.
def circle_r(t, R = 1, x0 = 0, y0 = 0, angle = 2*np.pi, shift = 0):
    x = x0 + R * np.cos(t*angle + shift)
    y = y0 + R * np.sin(t*angle + shift)
    return x, y

# The second returns the normal vector (nx,ny) of the curve at t.
def circle_n(t, angle = 2*np.pi, shift = 0, **kwargs):
    nx = np.cos(t*angle + shift)
    ny = np.sin(t*angle + shift)
    return nx, ny

# The third returns the arc length of the curve up to parameter value t.
def circle_arc(t, R = 1, angle = 2*np.pi, **kwargs):
    return  R * angle *t

# We define the equivalent functions for line segments.
def line_r(t, x0 = 0, y0 = 0, x1 = 1, y1 = 1):
    x = (x1-x0) * t + x0 
    y = (y1-y0) * t + y0
    return x, y

def line_n(t, x0 = 0, y0 = 0, x1 = 1, y1 = 1):
    l = np.sqrt((x1 - x0)**2 + (y1 -y0)**2)
    nx = (y1 - y0)/l
    ny = (x0 - x1)/l
    return nx, ny

def line_arc(t, x0 = 0, y0 = 0, x1 = 1, y1 = 1):
    l = np.sqrt((x1 - x0)**2 + (y1 -y0)**2)
    return t * l

# We construct 6 curves that define the border of the billiard.
# Parameters are passed as dictionaries of key-word arguments.
params_1 = {"angle" : np.pi/2}
arc_1 = cv.curve(circle_r, circle_n, circle_arc, **params_1)

params_2 = {"x0": 0, "x1": -1, "y0": 1, "y1": 0}
line_2 = cv.curve(line_r, line_n, line_arc, **params_2)

params_3 = {"x0": -1, "x1": -1, "y0": 0, "y1": -1}
line_3 = cv.curve(line_r, line_n, line_arc, **params_3)

params_4 = {"x0": -1, "x1": 0.5, "y0": -1, "y1": -1}
line_4 = cv.curve(line_r, line_n, line_arc, **params_4)

params_5 = {"R" : 0.5, "x0": 0.5, "y0" : -0.5 , "angle" : np.pi/2, "shift" : 3*np.pi/2}
arc_5 = cv.curve(circle_r, circle_n, circle_arc, **params_5)

params_6 = {"x0": 1, "x1": 1, "y0": -0.5, "y1": 0}
line_6 = cv.curve(line_r, line_n, line_arc, **params_6)

# We may check the construction of any of the curves 
# by using the plot_curve function from the plotting functions module
pf.plot_curve(arc_1)
plt.show()

# The curves are placed into a list with the correct ordering to form the border of the billiard.
border = [arc_1, line_2, line_3, line_4, arc_5, line_6]
#We calculate the surface area of the billiard.
area = np.pi/4 + 1/2 + 1.5 + np.pi *0.5**2 + 0.5**2 

# An optional parameter defines the point density scaling with k on each curve.
# The default value is 10. 
densities = [15 for i in border] 
weirdBilliard = bc.billiard(border, area, point_densities=densities)

# We can plot the billiard border and normal directions 
# using the plot_boundary function from plotting functions 
# to check if input is correct.
pf.plot_boundary(weirdBilliard)
plt.show()

# We can now calculate the ground state wavevector 
# using the Vergini-Saraceno scaling method,
k0 = 2.4
dk = 0.05
N = 200
VSresult = weirdBilliard.scaling_eigenvalues(N, k0, dk)
print(VSresult)
# or the plane wave decomposition method
PWDresult = weirdBilliard.PWD_eigenvalue(N, k0, dk)
print(PWDresult)

# We may plot the PWD tension profile using the folowing function
k0 = 2.44
dk = 0.05
pf.plot_tension(weirdBilliard, k0 -dk, k0+dk, N = 200, grid = 400)
plt.show()

# The grond state probbability may be ploted using the plot_probability function
pf.plot_probability(weirdBilliard, PWDresult.x)
plt.show()

# We may countinue to calculate exited states.
k1 = 3.5
k2 = 4.5
pf.plot_tension(weirdBilliard, k1, k2, N = 200, grid = 400)
plt.show()

k0 = 3.7
dk = 0.25
N = 500
PWDresult = weirdBilliard.PWD_eigenvalue(N, k0, dk)
print(PWDresult)
pf.plot_probability(weirdBilliard, PWDresult.x)
plt.show()

k0 = 4
dk = 0.25
N = 500
PWDresult = weirdBilliard.PWD_eigenvalue(N, k0, dk)
print(PWDresult)
pf.plot_probability(weirdBilliard, PWDresult.x)
plt.show()