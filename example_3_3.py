"""Reveal the meaning of "intersection points"."""
from matplotlib import pyplot as plt
import numpy as np

func = lambda x: np.cos(5 * np.pi * x)

x_sample = np.array([1/5, 2/5, 3/5, 4/5, 1])
x_pnt = np.linspace(0, 1, 100)
plt.plot(x_sample, func(x_sample), "o")
plt.plot(x_pnt, func(x_pnt), "--")
plt.show()
