# -*- coding: utf-8 -*-
"""
Authors: Michael Baosv, Shlomo Gulayev, Micha Briskman
ID: 315223156, 318757382, 208674713
"""
import numpy as np
from scipy.optimize import curve_fit
import time


def sin_func(x, a, b):
    """
    Returns Y value of a given sin function
    """
    return a * np.sin(b * x)


X = np.array([-5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5,
              1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
Y = np.array([-2.16498306, -1.53726731, 1.67075645, 2.47647932, 4.49579917,
                   1.14600963, 0.15938811, -3.09848048, -3.67902427, -1.84892687,
                   -0.11705947, 3.14778203, 4.26365256, 2.49120585, 0.55300516,
                   -2.105836 , -2.68898773, -2.39982575, -0.50261972, 
                   1.40235643,2.15371399])
start_time = time.time()
popt, pcov = curve_fit(sin_func, X, Y)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: %.2f milliseconds" % (elapsed_time * 1000))
print('a: ', popt[0], ' b: ', popt[1])