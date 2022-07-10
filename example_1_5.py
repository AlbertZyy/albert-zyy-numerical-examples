from typing import Literal, SupportsIndex, Union
from numpy.typing import NDArray
from numpy.linalg import norm
import numpy as np


def p_norm(__x: NDArray, __ord: Union[SupportsIndex, Literal["inf"]] = 2) -> float:
    if __ord == "inf":
        return np.max(np.abs(__x))
    __ord = int(__ord)
    return np.sum(np.power(__x, __ord)) ** (1 / __ord)

def frobenius_norm(__x: NDArray) -> float:
    squ = np.sum(np.square(__x))
    return np.sqrt(squ)

def angle(__x: NDArray, __y: NDArray) -> float:
    tan = np.inner(__x, __y) / (norm(__x, 2), norm(__y, 2))
    return np.arctan(tan)

a = np.array(
    [
        [1, 2, 3]
    ]
)

print(p_norm(a))
