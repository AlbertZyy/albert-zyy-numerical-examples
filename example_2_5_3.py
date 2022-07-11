"""An example for cubic spline interpolation"""

from scipy.interpolate import CubicSpline
import numpy as np
from matplotlib import pyplot as plt


x_pnt = np.linspace(0, 3, 100)
x_sample = np.array([0, 1, 2, 3])
y_sample = np.array([0, 0, 0, 0])
cs = CubicSpline(x_sample, y_sample, bc_type=((1, 1), (1, 0)))
y_pnt = cs(x_pnt)

plt.plot(x_sample, y_sample, "o")
plt.plot(x_pnt, y_pnt, "-")
plt.show()
