import numpy as np
import theano
import theano.tensor as tt

from theano.compile.ops import as_op

dtype = theano.config.floatX

if dtype == 'float32':
    itypes = [tt.fmatrix]
    otypes = [tt.fvector]
else:
    itypes = [tt.dmatrix]
    otypes = [tt.dvector]


def distribution_pyop(dist):
    """Build distribution op for graph

    Parameters
    ----------
    dist : Distribution or Mixture distribution instance
        Distribution which will be evaluated given input parameters

    Returns
    -------
    theano op : takes a parameter as input and calculates the probability
    of that parameter under the distribution (not log probability)
    """
    @as_op(itypes=itypes, otypes=otypes)
    def op(theta):
        return dist.eval(theta, log=False).astype(dtype).reshape(-1,)
    return op


def kernel_pyop(kernel):
    """Factory function to build kernel op for graph

    Parameters
    ----------
    kernel : Kernel instance
        Kernel which will be evaluated at particular point given input

    Returns
    -------
    theano op : takes a parameter as input and evaluates the kernel
    """
    @as_op(itypes=itypes, otypes=otypes)
    def op(x):
        return kernel.eval(x).astype(dtype).reshape(-1,)
    return op
