import numpy as np

from unittest import TestCase

from test.test_functions.numba_function_default_anotation import numba_extremes_index_default, numba_extremes_default


class TestNumbaExtremesDefault(TestCase):

    def test_empty_array_raises_value_error(self):
        array = np.array([])
        with self.assertRaises(ValueError):
            numba_extremes_index_default(array)

        with self.assertRaises(ValueError):
            numba_extremes_default(array)

    def test_returns_right_output(self):
        array = np.array([1, 1.5,  2, 3, 2])
        min_index, max_index = numba_extremes_index_default(array)
        min_value, max_value = numba_extremes_default(array)
        self.assertAlmostEqual(min_index, 0)
        self.assertAlmostEqual(max_index, 3)

        self.assertAlmostEqual(min_value, 1)
        self.assertAlmostEqual(max_value, 3)