# -*- coding: utf-8 -*-
"""
Authors: Michael Baosv, Shlomo Gulayev, Micha Briskman
ID: 315223156, 318757382, 208674713
"""
import numpy as np
from sympy import symbols, lambdify, diff, sin, cos
import matplotlib.pyplot as plt
import time


X = np.array([-5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5,
              1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
Y = np.array([-2.16498306, -1.53726731, 1.67075645, 2.47647932, 4.49579917,
                   1.14600963, 0.15938811, -3.09848048, -3.67902427, -1.84892687,
                   -0.11705947, 3.14778203, 4.26365256, 2.49120585, 0.55300516,
                   -2.105836 , -2.68898773, -2.39982575, -0.50261972, 
                   1.40235643,2.15371399])
error = []

x, y, a, b, c = symbols('x, y, a, b, c')
f = a * sin(b * x) * cos(c * x)
F = lambdify([x, a, b, c], f, 'numpy')

e = (f - y)**2
E = lambdify([x, a, b, c, y], e, 'numpy')

a0 = 4
b0 = 1.7
c0 = 0.7
lr = 0.0001

dfa = diff(e, a)
DFA = lambdify([x, a, b, c, y], dfa, 'numpy')
dfb = diff(e, b)
DFB = lambdify([x, a, b, c, y], dfb, 'numpy')
dfc = diff(e, c)
DFC = lambdify([x, a, b, c, y], dfc, 'numpy')
start_time = time.time()
for i in range(1000):
    da = DFA(X, a0, b0, c0, Y)
    db = DFB(X, a0, b0, c0, Y)
    dc = DFC(X, a0, b0, c0, Y)
    
    a0 = a0 - lr * sum(da)
    b0 = b0 - lr * sum(db)
    c0 = c0 - lr * sum(dc)
    loss_rate = E(X, a0, b0, c0, Y)
    error.append(sum(loss_rate))
end_time = time.time()
print('a: ', a0, ' b: ', b0, ' c: ', c0)
print(error[-1])
elapsed_time = end_time - start_time
print("Elapsed time: %.2f milliseconds" % (elapsed_time * 1000))

fig, axes = plt.subplots()
axes.scatter(X, Y)
X_line = np.array(X)
Y_line = a0 * X_line + b0
axes.plot(X_line, Y_line)

fig, axes = plt.subplots()
plt.xlabel("iterations")
plt.ylabel("loss rate")
axes.plot(range(1000), error)
