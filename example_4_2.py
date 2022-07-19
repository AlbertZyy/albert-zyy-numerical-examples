"""Examples for gaussian elimination."""
import numpy as np

def gauss_elimination(A, b):
    n = len(A)
    G = np.hstack((A, b.reshape(-1, 1))).astype(dtype=np.float64)
    for k in range(n):
        for i in range(k + 1, n):
            G[i] = G[i] - G[i][k]/G[k][k]*G[k]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (G[i][n] - np.matmul(G[i, :-1], x)) / G[i][i]
    return x

A = np.array([[2, 2, 2],
              [2, 3, 4],
              [-1, 3, 2]])
b = np.array([6, 4, 8])

# ret = gauss_elimination(A, b)
# print(ret)


#################################################
### Column principal elimination method
#################################################

def principal_elimination(A, b):
    n = len(A)
    G = np.hstack((A, b.reshape(-1, 1))).astype(dtype=np.float64)
    for k in range(n):
        max_i = np.argmax(np.abs(G[k:, k])) + k

        temp_row = np.array(G[k])
        G[k] = G[max_i]
        G[max_i] = temp_row

        for i in range(k + 1, n):
            G[i] = G[i] - G[i][k]/G[k][k]*G[k]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (G[i][n] - np.matmul(G[i, :-1], x)) / G[i][i]
    return x

A = np.array([[0, 2, 1],
             [1, 1, 0],
             [2, 3, 2]])
b = np.array([5, 3, 0])

# ret = principal_elimination(A, b)
# print(ret)


#################################################
### LU decomposition
#################################################

### An example

a_mat = np.array(
    [
        [1, 2, 1],
        [2, 2, 3],
        [-1, -3, 0]
    ]
)

### Doolittle decomposition

def doolittle(mat_a):
    u = np.array(mat_a, dtype=np.float64)
    n = u.shape[0]
    l = np.identity(n)
    for i in range(1, n):
        for k in range(i):
            l[i, k] = u[i, k] / u[k, k]
            u[i] = u[i] - l[i, k] * u[k]
    return l, u

# mat_l, mat_u = doolittle(a_mat)
# print("L = \n", mat_l)
# print("U = \n", mat_u)

### LDU decomposition

def LDU(A):
    n = len(A)
    l = np.identity(n)
    u = np.array(A, dtype=np.float64)
    for k in range(n):
        for i in range(k + 1, n):
            l[i,k] = u[i,k] / u[k,k]
            u[i] = u[i] - l[i,k] * u[k]

    d = np.diag(np.diag(u))
    u /= np.diag(u).reshape(-1, 1)

    return l, d, u

# mat_l, mat_d, mat_u = LDU(a_mat)
# print("L = \n", mat_l)
# print("D = \n", mat_d)
# print("U = \n", mat_u)

### PLU decomposition

# from scipy.linalg import lu

# mp, ml, mu = lu(a_mat)
# print(mp, ml, mu, sep="\n")


#################################################
### Cholesky decomposition
#################################################

from scipy.linalg import cholesky

A = np.array([[3, 3, 5],
              [3, 5, 9],
              [5, 9, 17]])

L = cholesky(A)
print("L = \n", L)
print("L^TL = \n", np.matmul(L.transpose(), L))

# D1 = np.diag(np.diag(L))
# L1 = L / np.diag(L)[None, :]
# U = L.T
# D2 = np.diag(np.diag(U))
# U1 = U / np.diag(U)[:, None]
# D = np.dot(D1, D2)
