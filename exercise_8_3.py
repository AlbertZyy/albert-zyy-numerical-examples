"""Exercise 8.3

Use Monte Carlo method to approximately calculate
the sum of 1, 2, ..., 10000.

Show how the error between approximate and
exact values changes through different sampling scales."""

from typing import SupportsIndex
import numpy as np
from matplotlib import pyplot as plt

EXACT_VALUE = 50005000

def accumulative(__x:SupportsIndex, points:SupportsIndex = 10000):
    __x = int(__x)
    points = int(points)
    sample_x = np.random.random((1, points)) * __x + 1
    sample_y = np.random.random((1, points)) * (__x + 1)

    flag = sample_y < sample_x - 0.5

    return np.sum(flag) / points * (__x + 1) * __x


if __name__ == "__main__":
    res = []

    points_list = [1e4, 1e5, 1e6]

    for points in points_list:
        res.append([accumulative(10000, points=points) - EXACT_VALUE for _ in range(10)])

    print(res)

    variances = [np.var(res_of_points) for res_of_points in res]
    plt.plot(points_list, variances, "b*--")
    plt.show()
