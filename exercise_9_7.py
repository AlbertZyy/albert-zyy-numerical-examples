"""Linear programming."""

from itertools import combinations
from numpy.typing import ArrayLike, NDArray
import numpy as np
from scipy.linalg import solve, LinAlgError


def feasible(a_mat: ArrayLike, b_vec: ArrayLike, positive_only: bool=False):
    a_mat = np.array(a_mat)
    b_vec = np.array(b_vec)

    if len(a_mat.shape) != 2:
        raise ValueError
    n_res, n_var = a_mat.shape
    b_vec = b_vec.reshape((n_res, 1))



    for combs in combinations(range(n_var), n_var - n_res):
        try:
            the_solve: NDArray = solve(a_mat[:, combs], b_vec)
        except LinAlgError:
            continue
        if positive_only and any(the_solve < 0):
            continue
        ret = np.zeros((n_var, 1))
        ret[combs, :] = the_solve.reshape((n_res, 1))
        yield ret.reshape(n_var)


def func(x_array: NDArray):
    return 2 * x_array[0] - x_array[1] + 3 * x_array[2] - 5 * x_array[3]


for the_solve in feasible(
    [
        [1, 1, 0, 0, 1, 0, 0, 0],
        [1, 2, 4, -1, 0, 1, 0, 0],
        [2, 3, -1, 4, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 0, 1]
    ],
    [3, 6, 12, 4],
    True
):
    print(func(the_solve), the_solve)

input("Press <Enter> to quit ...")
