"""Useful tools for dealing with cffi."""
from fast_green_kernel import ffi

import numpy as np

def as_double_ptr(arr):
    """Turn to a double ptr."""
    return ffi.cast("double*", arr.ctypes.data)


def as_float_ptr(arr):
    """Turn to a float ptr."""
    return ffi.cast("float*", arr.ctypes.data)


def as_usize(num):
    """Cast number to usize."""
    return ffi.cast("unsigned long", num)


def align_data(arr, dtype=None):
    """Make sure that an array has the right properties."""

    if dtype is None:
        dtype = arr.dtype

    return np.require(arr, dtype=dtype, requirements=["C", "A"])
