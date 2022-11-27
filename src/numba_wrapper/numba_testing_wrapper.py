import sys

import numba as nb


def testable_jit(nopython=True):
    is_test = "unittest" in sys.modules

    def wrapper(func):
        if is_test:

            @nb.jit(nopython=False)
            def wrapped_function(*args):
                return func(*args)

            return wrapped_function

        else:
            @nb.jit(nopython=nopython)
            def wrapped_function(*args):
                return func(*args)
            return wrapped_function

    return wrapper
