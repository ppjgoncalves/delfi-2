import numpy as np

from delfi.kernel.BaseKernel import KernelBase


class Uniform(KernelBase):
    @staticmethod
    def kernel(u):
        if np.abs(u) <= 1:
            return 0.5
        else:
            return 0.
