"""Newton method and Newton downhill method."""

import numpy as np
from matplotlib import pyplot as plt

def func(x):
    return 0.5 * x**2 - x

def dfunc(x):
    return x - 1

x_pnt = np.linspace(1, 4.5, 100)
x_sample = np.array([4.0, 2.667, 2.133])

y_pnt = func(x_pnt)
y_sample = func(x_sample)

plt.plot(x_pnt, y_pnt)

for i in range(x_sample.shape[0]):
    plt.plot((x_sample[i], x_sample[i]), (0, y_sample[i]))
    plt.text(x_sample[i], 0, f"$x_{i}$")

for i in range(x_sample.shape[0], 1, -1):
    plt.plot((x_sample[i-1], x_sample[i-2]), (0, y_sample[i-2]))

plt.grid()
plt.show()


def newton(f, df, init, tol, m = 1, N = 100):
    if N <= 0:
        print("Method failed")
        return None
    p = init - m * f(init)/df(init)
    print(p)
    if abs(p-init)/abs(p) < tol and f(p) < tol:
        return p
    else:
        return newton(f, df, p, tol, m, N-1)


def newton2(f, df, ddf, init, tol, N = 100):
    if N <= 0:
        print("Method failed")
        return None
    fval = f(init)
    dfval = df(init)
    p = init - fval*dfval/(dfval**2 - fval*ddf(init))
    print(p)
    if abs(p-init)/abs(p) < tol and f(p) < tol:
        return p
    else:
        return newton2(f, df, ddf, p, tol, N-1)


def newton_downhill(f, df, init, lambda_, tol, N = 100, f_x = None, df_x = None):
    if N <= 0:
        print("Method failed")
        return None
    fval = f_x if f_x else f(init)
    dfval = df_x if df_x else df(init)
    p = init - lambda_ * fval/dfval
    fpval = f(p)
    if abs(p-init)/abs(p) < tol and fpval < tol:
        return p
    if fpval < fval:
        print(p, lambda_, fpval)
        return newton_downhill(f, df, p, 1, tol, N-1)
    else:
        return newton_downhill(f, df, init, lambda_/2, tol, N, fval, dfval)
