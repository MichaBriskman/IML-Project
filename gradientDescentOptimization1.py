# -*- coding: utf-8 -*-
"""
Authors: Michael Baosv, Shlomo Gulayev, Micha Briskman
ID: 315223156, 318757382, 208674713
"""
import matplotlib.pyplot as plt
import numpy as np


def linear_func(x, a, b):
    """
    Returns Y value of a given linear function
    """
    return a * x + b


def single_error(x, a, b, y):
    """
    Compute error of a single given value
    """
    return (a * x - b - y)**2


def error_func(x, a, b, y):
    """
    Define an error function of a given linear function
    """
    Y = linear_func(x, a, b)
    return np.sum((Y - y)**2)


def gradient_calc_func(x, a, b, y):
    """
    Calculate gradient descent for a given linear function
    """
    Y = linear_func(x, a, b)
    da = 2 * np.sum((Y - y) * x)
    db = 2 * np.sum(Y - y)
    return da, db


# Q1_a
x = np.array([-3.0, -2.0, 0.0, 1.0, 3.0, 4.0])
y = np.array([-1.5, 2.0, 0.7, 5.0, 3.5, 7.5])
error = []
a_history = []
b_history = []


lr = 0.0001
a = 1
b = 1

for i in range(1000):
    a_history.append(a)
    b_history.append(b)
    da, db = gradient_calc_func(x, a, b, y)
    a = a - lr * da
    b = b - lr * db
    error.append(error_func(x, a, b, y))
    
print('a: ', a, ' b: ', b)


fig, axes = plt.subplots()
axes.scatter(x, y)
X = np.array(x)
Y = a * x + b
axes.plot(X, Y)

fig, axes = plt.subplots()
plt.xlabel("iterations")
plt.ylabel("loss rate")

axes.plot(range(1000), error)

z = [] 
surf_x = np.arange(-5, 5, 0.1)
surf_y = np.arange(-5, 5, 0.1)
surf_z = single_error(surf_x, a, b, surf_y)

X, Y = np.meshgrid(surf_x, surf_y)
Z = single_error(X, a, b, Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.scatter(surf_x, surf_y, surf_z, color='r')

plt.show()
