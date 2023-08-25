# -*- coding: utf-8 -*-
"""
Authors: Michael Baosv, Shlomo Gulayev, Micha Briskman
ID: 315223156, 318757382, 208674713
"""
import numpy as np
from sympy import symbols, lambdify, sin, diff
import matplotlib.pyplot as plt


def single_error(x, a, b, y):
    """
    Compute error of a single given value
    """
    return (a * x - b - y)**2


X = np.array([-5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5,
              1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
Y = np.array([-2.16498306, -1.53726731, 1.67075645, 2.47647932, 4.49579917,
                   1.14600963, 0.15938811, -3.09848048, -3.67902427, -1.84892687,
                   -0.11705947, 3.14778203, 4.26365256, 2.49120585, 0.55300516,
                   -2.105836 , -2.68898773, -2.39982575, -0.50261972, 
                   1.40235643,2.15371399])
error = []

x, y, a, b = symbols('x, y, a, b')
f = a * sin(b * x)
F = lambdify([x, a, b], f, 'numpy')

e = (f - y)**2
E = lambdify([x, a, b, y], e, 'numpy')

a0 = 3
b0 = 1.1
lr = 0.0001

dfa = diff(e, a)
DFA = lambdify([x, a, b, y], dfa, 'numpy')
dfb = diff(e, b)
DFB = lambdify([x, a, b, y], dfb, 'numpy')

for i in range(1000):
    da = DFA(X, a0, b0, Y)
    db = DFB(X, a0, b0, Y)
    
    a0 = a0 - lr * sum(da)
    b0 = b0 - lr * sum(db)
    loss_rate = E(X, a0, b0, Y)
    error.append(sum(loss_rate))

print('a: ', a0, ' b: ', b0)

fig, axes = plt.subplots()
axes.scatter(X, Y)
X_line = np.array(X)
Y_line = a0 * X_line + b0
axes.plot(X_line, Y_line)

fig, axes = plt.subplots()
plt.xlabel("iterations")
plt.ylabel("loss rate")
axes.plot(range(1000), error)

z = [] 
surf_x = np.arange(-5, 5, 0.1)
surf_y = np.arange(-5, 5, 0.1)
surf_z = single_error(surf_x, a0, b0, surf_y)

X, Y = np.meshgrid(surf_x, surf_y)
Z = single_error(X, a0, b0, Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.scatter(surf_x, surf_y, surf_z, color='r')

plt.show()
