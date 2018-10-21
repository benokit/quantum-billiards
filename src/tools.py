#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 14:16:36 2017

@author: benjamin
"""

import numpy as np

def ecdf(z):
    zz = np.sort(z)
    n = 1 + zz.size
    f = lambda x: np.count_nonzero(zz <= x) / n
    return np.vectorize(f)
    