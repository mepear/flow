import numpy as np

class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        # self._S = np.zeros(shape)
        self._STD = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            # self._S[...] = self._S + (x - oldM) * (x - self._M)

            old_std = self._STD.copy()
            self._STD[...] = np.sqrt(np.square(self._STD) + ((x - oldM) * (x - self._M) - np.square(old_std)) / (self._n - 1))
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    # @property
    # def var(self):
        # return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        # return np.sqrt(self.var)
        return self._STD
    @property
    def shape(self):
        return self._M.shape
