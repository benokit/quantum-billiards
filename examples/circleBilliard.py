import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from ..src import billiardClass as bc
from ..src import plottingFunctions as pf

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

# We construct a curve that defines the border of the billiard.
# Default key-word parameters are used in this case.
circle = bc.curve(circle_r, circle_n, circle_arc)

# The border of the billiard is defined by a list of curves. 
border = [circle]
# We calculate the area of the billiard.
area = np.pi
# We construct a billiard.
circleBilliard = bc.billiard(border, area)

# we can plot the billiard border and normal directions 
# using the plot_boundary function from plotting functions 
# to check if input is correct
pf.plot_boundary(circleBilliard)
plt.show()

# we can now calculate the ground state wavevector 
# using the Vergini-Saraceno scaling method
k0 = 2.4
dk = 0.05
N = 200
VSresult  = circleBilliard.scaling_eigenvalues(N, k0, dk)
print(VSresult)
# or the plane wave decomposition method
PWDresult = circleBilliard.PWD_eigenvalue(N, k0, dk)
print(PWDresult)
# and compare with the analytical result
print(special.jn_zeros(0,1))
# we may plot the PWD tension profile using the folowing function
pf.plot_tension(circleBilliard, k0 -dk, k0+dk, grid = 200)

# The grond state probbability may be ploted using the plot_probability function
pf.plot_probability(circleBilliard, PWDresult.x)