"""
Implements a new strategy for estimating the new candidate locations at each step.
It is based on looking for the local maxima of the dual certificate at each stepp.
To do so, we change the behavior of PFW with a Mixin class, that overwrites the m_step method.
We need different implementations for 1D and 2D data (higher dimensions not considered so far).
"""

import numpy as np
import scipy.signal as ss

import pyfwl

__all__ = ["LocalMaxPFW"]

class MixinLocalMax1D():
    def find_candidates(self, mgrad):
        """
        It will probably be needed to have a decreasing weight that is strictly above 1 in the search of the peaks.
        """
        if self._astate["positivity_c"]:
            new_indices = ss.find_peaks(mgrad, height=1.)[0]
        else:
            new_indices = ss.find_peaks(np.abs(mgrad), height=1.)[0]
        return new_indices

class LocalMaxPFW(MixinLocalMax1D, pyfwl.PFWLasso):
    pass


