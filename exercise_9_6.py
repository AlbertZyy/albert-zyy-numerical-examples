"""Exercise 9.6

Use the steepest descend method to solve

$$\min x_1^2 - 2x_1x_2 + 4x_2^2 + x_1 - 3x_2$$

where $$x^{(0)} = (1, 1)^T$$, iterate twice."""

from typing import Literal, Tuple
import numpy as np
from numpy.typing import NDArray, ArrayLike
from numpy.linalg import norm



def st_dcnt(
    a_mat: NDArray,
    b_vec: NDArray,
    init: ArrayLike,
    tol: float = 1e-5,
    n_max: int = 500
) -> Tuple[NDArray[np.floating], Literal[0, 1], int]:
    a_shape = a_mat.shape
    if len(a_shape) != 2 or a_shape[0] != a_shape[1]:
        raise ValueError
    b_vec = np.squeeze(b_vec)
    b_shape = b_vec.shape
    b_vec = b_vec.reshape((b_shape[0], 1))
    if len(b_shape) == 1:
        if b_shape[0] != a_shape[0]:
            raise ValueError
    else:
        raise ValueError
    init_point = np.array(init)

    x_point = init_point.reshape((b_shape[0], 1))
    step = 0
    while step < n_max:
        d_step: NDArray = - b_vec - np.matmul(a_mat, x_point)
        if norm(d_step, 2) < tol:
            return x_point, 0, step
        alpha_step: float = np.matmul(np.transpose(d_step), d_step)\
            / np.matmul(np.transpose(np.matmul(a_mat, d_step)), d_step)

        x_point = x_point + alpha_step * d_step
        step += 1
    return x_point, 1, step


mat_A = np.array(
    [
        [2, -2],
        [-2, 8]
    ]
)

vec_B = np.array(
    [1, -3]
)

print(st_dcnt(mat_A, vec_B, [1, 1], 1e-8))
input("Press <Enter> to quit ...")
