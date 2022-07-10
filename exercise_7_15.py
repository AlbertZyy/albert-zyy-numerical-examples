"""Exercise 7.15

Use difference method with h = 1 to solve the problem
y'' - xy' + y = 1, y(0) = y(2) = 0
"""

from typing import Callable

import numpy as np
from scipy.linalg import solve_banded
from matplotlib import pyplot as plt

_F = Callable[[np.ndarray], float]

N = 20
"""Number of points - 1"""

def make_mat(start:float, cond1:float, end:float, cond2:float,\
    r_m:_F, q_m:_F, p_m:_F, ab_mode=False):
    """Construct the matrix and vector."""
    x_val = np.linspace(start, end, N + 1)[1:N]
    interval = (end - start) / N
    a_m = - (1 + interval*p_m(x_val)/2)
    d_m = 2 + interval**2*q_m(x_val)
    c_m = - (1 - interval*p_m(x_val)/2)
    b_m = - interval**2*r_m(x_val)

    # construct matrix
    if ab_mode:
        result_mat = np.zeros((3, N-1))
        result_mat[0, 1:] = c_m[:N-2]
        result_mat[1, :] = d_m[:]
        result_mat[2, :N-2] = a_m[1:]
    else:
        result_mat = np.zeros((N-1, N-1))
        result_mat[0, 0], result_mat[0, 1] = d_m[0], c_m[0]

        for i in range(1, N-2):
            result_mat[i, i-1], result_mat[i, i], result_mat[i, i+1] = a_m[i], d_m[i], c_m[i]

        result_mat[N-2, N-3], result_mat[N-2, N-2] = a_m[N-2], d_m[N-2]

    # construct vector
    result_vec = np.zeros((N-1,))
    result_vec[0] = b_m[0] - a_m[0] * cond1

    result_vec[1: N-2] = b_m[1: N-2]

    result_vec[N-2] = b_m[N-2] - c_m[N-2] * cond2

    return result_mat, result_vec

a_mat, b_vec = make_mat(
    0, 0, 2, 0,
    lambda x:np.ones(x.shape),
    lambda x:-np.ones(x.shape),
    lambda x:x,
    True
)

print(a_mat, b_vec, sep="\n")
result = solve_banded((1, 1), a_mat, b_vec)
print(result)

# Draw the result
y_val = np.zeros((N+1,))
y_val[1:N] = result
plt.plot(np.linspace(0, 2, N + 1), y_val)
plt.show()
