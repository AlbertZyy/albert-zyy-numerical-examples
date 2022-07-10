"""Exercise 8.4

Use the Monte Carlo method to approximately calculate
the integrations below:

(1) $$int_0^{2\pi} e^{x^2} \sin^3(x) dx$$

(2) $$\int_1^3\int_0^{2x} x^3y + e^y\cos x dydx$$

(3) $$\int\int_{x^2+y^2\le 1} \frac{e^{xy}}{x+y+3} dxdy"""
# This has not been finished.

from time import time
from multiprocessing import Pool, cpu_count
from typing import Callable, SupportsFloat, Tuple
import numpy as np


def monte_carlo_int_1d(__f: Callable[[float], float], __interval: Tuple[SupportsFloat], points: int = 10000) -> float:
    edge_a = float(__interval[0])
    edge_b = float(__interval[1])
    points = int(points)

    if edge_b < edge_a:
        raise ValueError

    span = edge_b - edge_a

    sample_x = np.random.random((points, )) * span + edge_a
    sample_y = __f(sample_x) * span

    return sample_y


def multi_process(func: Callable[[float], float], method: Callable, interval_var: Tuple, points: int = 10000):
    points = int(points)
    n_cpu = cpu_count()
    remain = points % n_cpu
    span = points // n_cpu
    split = [span] * n_cpu
    for i in range(remain):
        split[i] += 1

    print(
f"""Build {n_cpu} processes, the number of points assigned to them are:
{split}."""
    )

    the_pool = Pool(n_cpu)
    pool_data = []

    for point_num in split:
        res = the_pool.apply_async(method, args=(func, interval_var, point_num))
        pool_data.append(res)

    the_pool.close()
    the_pool.join()

    result = np.concatenate([x.get() for x in pool_data], axis=0)

    return np.mean(result)


def func1(__x: float):
    return np.exp(__x ** 2) * np.sin(__x) ** 3


if __name__ == "__main__":

    t1 = time()
    ret = multi_process(func1, monte_carlo_int_1d, (0, np.pi/2), points=1e6)
    # ret = monte_carlo_int_1d(func1, (0, np.pi/2), points=1e5)
    t2 = time()

    print(ret)
    print(f"calculation costs {t2 - t1:.3f} sec.")
