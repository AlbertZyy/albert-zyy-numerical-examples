
from numpy.linalg import norm
import numpy as np

# Create an array
vec1 = np.array([1.0, -2.0, 3.0])

# 2-norm
ret = np.sqrt(np.sum(vec1 ** 2))
print(ret)
print(norm(vec1, 2))

# inf-norm
ret = np.max(np.abs(vec1))
print(ret)
print(norm(vec1, np.inf))

# Create a matrix
mat1 = np.array(
    [
        [4, 2, 1],
        [2, 1, 3],
        [1, 3, 0]
    ]
)

# frobenius_norm
squ = np.sum(np.square(mat1))
ret = np.sqrt(squ)
print(ret)
print(norm(mat1, "fro"))

# Create two arrays
vec_a = np.array([1, 2, 3])
vec_b = np.array([-1, 2, -3])

# angle
cos = np.inner(vec_a, vec_b) / (norm(vec_a, 2) * norm(vec_b, 2))
ret = np.arccos(cos)
print(ret)
