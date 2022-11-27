from typing import Tuple

from numpy.typing import NDArray

import numba as nb


@nb.jit(nopython=True)
def numba_extremes_index_default(array: NDArray) -> Tuple[int, int]:
    """
    a method for calculating min and max of an array in a single array pass
    :return: min and max of an array
    """
    array_length = len(array)
    if array_length < 1:
        raise ValueError("cannot calculate extremes of an empty array!")

    min_element = array[0]
    max_element = array[0]
    min_index = 0
    max_index = 0
    for i in range(1, array_length):
        if array[i] < min_element:
            min_element = array[i]
            min_index = i
        elif array[i] > max_element:
            max_element = array[i]
            max_index = i
    return min_index, max_index


def numba_extremes_default(array: NDArray) -> Tuple[float, float]:
    min_index, max_index = numba_extremes_index_default(array)
    return array[min_index], array[max_index]
