"""A simple example to show the Runge phenomenon."""

from scipy.interpolate import lagrange
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from matplotlib import pyplot as plt

def func(__x):
    return 1 / (1+25 * __x ** 2)

x_pnt = np.linspace(-1, 1, 100)
y_pnt = func(x_pnt)

x_sample = np.linspace(-1, 1, 11)
y_sample = func(x_sample)

poly = lagrange(x_sample, y_sample)
y_ppnt = Polynomial(poly.coef[::-1])(x_pnt)

plt.scatter(x_sample, y_sample)
plt.plot(x_pnt, y_pnt)
plt.plot(x_pnt, y_ppnt)
plt.show()
