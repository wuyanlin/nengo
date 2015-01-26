"""
Extra functions to extend the capabilities of Numpy.
"""
from __future__ import absolute_import

import numpy as np

from nengo.utils.magic import FunctionWrapper

maxint = np.iinfo(np.int32).max


def compare(a, b):
    return 0 if a == b else 1 if a > b else -1 if a < b else None


def broadcast_shape(shape, length):
    """Pad a shape with ones following standard Numpy broadcasting."""
    n = len(shape)
    if n < length:
        return tuple([1] * (length - n) + list(shape))
    else:
        return shape


def array(x, dims=None, min_dims=0, **kwargs):
    y = np.array(x, **kwargs)
    dims = max(min_dims, y.ndim) if dims is None else dims

    if y.ndim < dims:
        shape = np.ones(dims, dtype='int')
        shape[:y.ndim] = y.shape
        y.shape = shape
    elif y.ndim > dims:
        raise ValueError(
            "Input cannot be cast to array with %d dimensions" % dims)

    return y


def expm(A, n_factors=None, normalize=False):
    """Simple matrix exponential to replace Scipy's matrix exponential

    This just uses a recursive (factored) version of the Taylor series,
    and is not as good as Scipy (which uses Pade approximants). The hard
    part with this implementation is choosing the length of the Taylor
    series. A longer series is generally needed for a matrix with a larger
    eigenvalues, but I'm not exactly sure how these relate. I'm using
    a heuristic based on the matrix norm, since this is kind-of related
    to the size of eigenvalues, though for larger norms the function
    becomes inaccurate no matter the length of the series.

    This function is mostly intended for use in `filter_design`, where
    the matrices should be small, both in dimensions and norm.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Argument must be a square matrix")

    a = np.linalg.norm(A)
    if normalize:
        a = int(a)
        A = A / float(a)

    if n_factors is None:
        n_factors = 20 if normalize else max(20, int(a))

    Y = np.zeros_like(A)
    for i in range(n_factors, 0, -1):
        Y = np.dot(A / float(i), Y)
        np.fill_diagonal(Y, Y.diagonal() + 1)  # add identity matrix

    return np.linalg.matrix_power(Y, a) if normalize else Y


def norm(x, axis=None, keepdims=False):
    """Euclidean norm

    Parameters
    ----------
    x : array_like
        Array to compute the norm over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. `None` sums all axes. See `np.sum`.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See `np.sum` in
        newer versions of Numpy (>= 1.7).
    """
    y = np.sqrt(np.sum(x**2, axis=axis))
    return np.expand_dims(y, axis=axis) if keepdims else y


def meshgrid_nd(*args):
    args = [np.asarray(a) for a in args]
    s = len(args) * (1,)
    return np.broadcast_arrays(*(
        a.reshape(s[:i] + (-1,) + s[i + 1:]) for i, a in enumerate(args)))


def rms(x, axis=None, keepdims=False):
    """Root-mean-square amplitude

    Parameters
    ----------
    x : array_like
        Array to compute RMS amplitude over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. `None` sums all axes. See `np.sum`.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See `np.sum` in
        newer versions of Numpy (>= 1.7).
    """
    y = np.sqrt(np.mean(x**2, axis=axis))
    return np.expand_dims(y, axis=axis) if keepdims else y


def rmse(x, y, axis=None, keepdims=False):
    """Root-mean-square error amplitude

    Equivalent to rms(x - y, axis=axis, keepdims=keepdims).

    Parameters
    ----------
    x, y : array_like
        Arrays to compute RMS amplitude over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. `None` sums all axes. See `np.sum`.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See `np.sum` in
        newer versions of Numpy (>= 1.7).
    """
    return rms(x - y, axis=axis, keepdims=keepdims)


if hasattr(np.fft, 'rfftfreq'):
    rfftfreq = np.fft.rfftfreq
else:
    def rfftfreq(n, d=1.0):
        return np.abs(np.fft.fftfreq(n=n, d=d)[:n // 2 + 1])


class array_like(object):
    """Decorator to apply `np.asarray` to function arguments

    We actually use `np.array` with the `copy` argument defaulting
    to false, since this is more configurable than `np.asarray`.

    Parameters
    ----------
    *args : strings
        The names of the arguments to ensure are arrays.
    **kwargs
        Keyword arguments to pass to `np.array`. `copy` defaults to False,
        so that it acts like `asarray`.

    Example
    -------
    A function that adds a scaled version of an array to another array:

    >>> @array_like('A', 'B')
    >>> def A_plus_scaled_B(A, beta, B):
    >>>     return A + beta * B

    This function will work as expected even if `A` and `B` are lists.
    """

    def __init__(self, *args, **kwargs):
        self.arguments = args  # the arguments to run `asarray` on
        self.array_kwargs = dict(copy=False)
        self.array_kwargs.update(kwargs)

    def __call__(self, wrapped):
        import inspect

        # check that self.arguments are valid
        argspec = inspect.getargspec(wrapped)
        for arg in self.arguments:
            if arg not in argspec.args:
                raise ValueError("'%s' is not an argument of '%s'" %
                                 (arg, wrapped.__name__))

        # make the wrapper
        def wrapper(_wrapped, _instance, args, kwargs):
            callargs = inspect.getcallargs(_wrapped, *args, **kwargs)
            for arg in self.arguments:
                callargs[arg] = np.array(callargs[arg], **self.array_kwargs)
            return wrapped(**callargs)

        return FunctionWrapper(wrapped, wrapper)
