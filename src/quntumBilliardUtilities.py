
import numpy as np
import math as m


def baseAngles(a, b, n):
    alpha = np.linspace(a, b, (n + 1))
    alpha = alpha + (b - a) / (2 * n)
    return alpha[0:n]


