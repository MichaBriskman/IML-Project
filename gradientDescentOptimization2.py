# -*- coding: utf-8 -*-
"""
Authors: Michael Baosv, Shlomo Gulayev, Micha Briskman
ID: 315223156, 318757382, 208674713
"""

import numpy as np
from sympy import symbols, lambdify, diff

X = np.array([-3.0, -2.0, 0.0, 1.0, 3.0, 4.0])
Y = np.array([-1.5, 2.0, 0.7, 5.0, 3.5, 7.5])
error = []

x, y, a, b = symbols('x, y, a, b')

f = a * x + b
F = lambdify([x, a, b], f, 'numpy')

e = (f - y)**2
E = lambdify([x, a, b, y], e, 'numpy')

a0 = 1
b0 = 1
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
    
print(a0, b0)
