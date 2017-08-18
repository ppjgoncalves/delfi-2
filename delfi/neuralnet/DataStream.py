import abc
import numpy as np
import theano

dtype = theano.config.floatX


class DataStream(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def gen(self, N):
        """Generates a new data batch of size N"""
        raise NotImplementedError


class DataSubSampler(DataStream):
    """Given a data set, subsamples mini-batches from it"""

    def __init__(self, xs, seed=None):
        N = xs[0].shape[0]
        self.index_stream = IndexSubSampler(N, seed=seed)
        self.xs = [
            theano.shared(
                x.astype(dtype),
                name='data' +
                str(i)) for i,
            x in enumerate(xs)]

    def gen(self, N):
        """Generates a new data batch of size N from the data set"""
        n = self.index_stream.gen(N)
        return [x[n] for x in self.xs]


class IndexSubSampler(DataStream):
    def __init__(self, num_idx, seed=None):
        """Subsamples minibatches of indices"""
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

        self.num_idx = num_idx
        self.nn = list(range(num_idx))
        self.rng.shuffle(self.nn)
        self.i = 0

    def gen(self, N):
        """Generates a new index batch of size N from 0:num_idx-1"""
        j = self.i + N
        times = j // self.num_idx
        new_i = j % self.num_idx
        n = []

        for t in range(times):
            n += self.nn[self.i:]
            self.rng.shuffle(self.nn)
            self.i = 0

        n += self.nn[self.i:new_i]
        self.i = new_i

        return n
