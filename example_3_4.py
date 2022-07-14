"""Show some chebyshev polynomials."""
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev

ch_1 = Chebyshev((0, 1, ))
ch_2 = Chebyshev((0, 0, 1, ))
ch_3 = Chebyshev((0, 0, 0, 1))
ch_4 = Chebyshev((0, 0, 0, 0, 1))

fig, ax = plt.subplots(2, 2)
x_pnt = np.linspace(-1, 1, 100)

ax[0, 0].plot(x_pnt, ch_1(x_pnt))
ax[0, 1].plot(x_pnt, ch_2(x_pnt))
ax[1, 0].plot(x_pnt, ch_3(x_pnt))
ax[1, 1].plot(x_pnt, ch_4(x_pnt))

plt.show()
