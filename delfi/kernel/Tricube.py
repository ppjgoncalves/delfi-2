import numpy as np

from delfi.kernel.BaseKernel import KernelBase


class Tricube(KernelBase):
    @staticmethod
    def kernel(u):
        if np.abs(u) <= 1:
            return 70/81*(1-np.abs(u)**3)**3
        else:
            return 0.
