import numpy as np


class GP:
    """
    A full-rank GP based on a kernel matrix for internal use.
    Assumming zero-mean input targets.
    """

    def __init__(self, noise, signal=1):
        """ Store the noise parameters. """
        self.K = self.y = self.Ci = None
        self.noise = noise
        self.signal = signal
        self.bias = 0

    def fit(self, K, y):
        """ Store a kernel matrix and associated targets. """
        self.K = K
        self.y = y
        self.bias = y.mean()
        self.Ci = np.linalg.inv(self.signal * self.K + self.noise * np.eye(self.K.shape[0]))

    def predict(self, Kx, Kxx=None):
        """
        Predictive distribution, given evaluations of the kernel.
        :param Kx: Test set vs. training set (matrix).
        :param Kxx: Test set vs. test set (vector).
        """
        nt, dt = Kx.shape
        Kx, Kxx = self.signal * Kx, self.signal * Kxx
        assert dt == 0 or self.K is not None
        if dt > 0:
            assert nt == len(Kxx)
            m = Kx.dot(self.Ci).dot(self.y - self.bias).reshape((nt, 1)) + self.bias
            v = np.array([Kxx[i] - Kx[i, :].dot(self.Ci).dot(Kx[i, :].T) + self.noise
                          for i in range(nt)]).reshape((nt, 1))
        else:
            m = np.zeros((nt, 1)) + self.bias
            v = np.array([Kxx[i] + self.noise for i in range(nt)]).reshape((nt, 1))
        return m, v