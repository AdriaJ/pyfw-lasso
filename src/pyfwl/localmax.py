"""
Implements a new strategy for estimating the new candidate locations at each step.
It is based on looking for the local maxima of the dual certificate at each stepp.
To do so, we change the behavior of PFW with a Mixin class, that overwrites the m_step method.
We need different implementations for 1D and 2D data (higher dimensions not considered so far).
"""

import numpy as np
import scipy.signal as ss
import skimage.feature as skf

import pyfwl

__all__ = ["LocalMaxPFW", "LocalMaxPFW2D"]

#todo later: remove in the original PFW anything that is link to the initial multispikes trategy. Also decompose what
# is necessary for 2d and factorize the code.

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

class MixinLocalMax2D():
    def __init__(self, image_shape: tuple, *args, **kwargs):
        self.image_shape = image_shape  # must be a size 2 tuple of integers
        super().__init__(*args, **kwargs)

    def find_candidates(self, mgrad):
        """
        It will probably be needed to have a decreasing weight that is strictly above 1 in the search of the peaks.
        """
        mgrad_im = mgrad.reshape(self.image_shape)
        if self._astate["positivity_c"]:
            indices2d = skf.peak_local_max(mgrad_im, threshold_abs=1., min_distance=1, exclude_border=True)
        else:
            indices2d = skf.peak_local_max(np.abs(mgrad_im), threshold_abs=1., min_distance=1, exclude_border=True).T
        new_indices = np.ravel_multi_index((indices2d[0], indices2d[1]), self.image_shape)
        return new_indices

class LocalMaxPFW(MixinLocalMax1D, pyfwl.PFWLasso):
    pass

class LocalMaxPFW2D(MixinLocalMax2D, pyfwl.PFWLasso):
    pass


