"""
This file implements a few common operators, useful in the development of PFW for the LASSO problem.
"""

import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "L1NormPositivityConstraint",
]

# class NonNegativeOrthant(pyco.ProxFunc):
#     """
#     Indicator function of the non-negative orthant (positivity constraint).
#     """
#
#     def __init__(self, shape: pyct.OpShape):
#         super().__init__(shape=shape)
#
#     @pycrt.enforce_precision(i="arr")
#     def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
#         xp = pycu.get_array_module(arr)
#         if arr.ndim <= 1:
#             res = 0 if xp.all(arr >= 0) else xp.inf
#             return xp.r_[res].astype(arr.dtype)
#         else:
#             res = xp.zeros(arr.shape[:-1])
#             res[xp.any(arr < 0, axis=-1)] = xp.infty
#             return res.astype(arr.dtype)
#
#     @pycrt.enforce_precision(i="arr")
#     def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
#         xp = pycu.get_array_module(arr)
#         res = xp.copy(arr)
#         res[res < 0] = 0
#         return res
#
#     def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
#         pass


class L1NormPositivityConstraint(pyco.ProxFunc):
    def __init__(self, shape: pyct.OpShape):
        super().__init__(shape=shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if arr.ndim <= 1:
            res = arr.sum() if xp.all(arr >= 0) else xp.inf
            return xp.r_[res].astype(arr.dtype)
        else:
            res = xp.full(arr.shape[:-1], xp.inf)
            indices = xp.all(arr >= 0, axis=-1)
            res[indices] = arr[indices].sum(axis=-1)
            return res.astype(arr.dtype)

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        res = xp.zeros_like(arr)
        indices = arr > 0
        res[indices] = xp.fmax(0, arr[indices] - tau)
        return res


if __name__ == "__main__":
    N = 10
    # posIndicator = NonNegativeOrthant(shape=(1, None))
    #
    # a = np.random.normal(size=N)
    # b = posIndicator.prox(a, tau=1)
    # print(posIndicator(a))
    # print(b)
    # print(posIndicator(b))
    #
    # print("0-d input: {}".format(posIndicator(np.r_[-1])))
    # print("3-d input: {}".format(posIndicator(np.arange(24).reshape((2, 3, 4)) - 3)))

    print("\nPositivity constraint:")
    posL1Norm = L1NormPositivityConstraint(shape=(1, None))

    a = np.random.normal(size=N)
    b = posL1Norm.prox(a, tau=0.1)
    print(posL1Norm(a))
    print(b)
    print(posL1Norm(b))

    print("0-d input: {}".format(posL1Norm(np.r_[-1])))
    print("3-d input: {}".format(posL1Norm(np.arange(24).reshape((2, 3, 4)) - 3)))
