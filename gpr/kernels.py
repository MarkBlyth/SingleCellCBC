import numpy as np
from abc import ABC, abstractmethod


class _Kernel(ABC):
    """
    Abstract class for managing kernels. Requires set_hyperparams to
    be defined, for checking then storing all relevant
    hyperparameters, and cov, for calculating the covariance between
    pairs of vectors.

    Requires the following variables:

        sigma_n : float
            Non-negative scalar representing the variance of any
            observation noise.
    """

    def _set_observation_noise(self, sigma_n):
        try:
            self.sigma_n = float(sigma_n)
        except TypeError:
            raise TypeError("sigma_n must be a float")
        if self.sigma_n < 0:
            raise ValueError("sigma_n must be >= 0")

    @abstractmethod
    def cov(self, x1, x2):
        """Abstract method. Get the covariance between two 1d array
        row vectors, x1 and x2. Must be implemented with a chosen
        covariance function.
        """
        pass

    @abstractmethod
    def set_hyperparams(self, **kwargs):
        """Abstract method for checking then storing hyperparameter
        values.
        """
        pass

    def get_cov_matrix(self, X1, X2, noise_term=False):
        """Construct a covariance matrix between array of row vectors
        X1, and X2.

            X1, X2 : float array
                Array of row vectors to evaluate covariance between.
        
            noise_term : bool
                True if we want to add a noise contribution to the
                covariance matrix, False otherwise.
        
        Passes up any exceptions.

        Returns a covariance matrix between each vector in X1, X2.
        """
        # Check and format into appropriate matrices
        X1mat = self._matrixify(X1)
        X2mat = self._matrixify(X2)
        n_cols = X1mat.shape[0]
        n_rows = X2mat.shape[0]
        noiseless_mat = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                noiseless_mat[i][j] = self.cov(X1mat[j], X2mat[i])
        if not noise_term:
            return noiseless_mat
        # add in a noise contribution
        noise_mat = self.sigma_n * np.eye(noiseless_mat.shape[0])
        return noiseless_mat + noise_mat

    def _matrixify(self, X):
        """
        Take a scalar, row vector, or 2d array of row vectors, check
        it, and return a 2d array of row vectors.
        
            X : float
                Scalar, 1d row vector, or 2d array of row vectors,
                representing the datapoints at which we want to
                evaluate covariances

        Raises the following:

            ValueError : if X cannot be turned into a 2d array

            ValueError : if the elements of X cannot be cast to float

        Returns a checked 2d array of floats
        """
        # If X is a scalar, turn it into a 1-element matrix
        xarr = np.array([[X]]) if np.isscalar(X) else X
        # If X is only a single datapoint (row vector), turn it into a matrix
        xmat = xarr.reshape((1, -1)) if len(xarr.shape) == 1 else xarr
        # Make sure we're working on a 2d array (list of 1d vectors)
        if len(xmat.shape) != 2 or xmat.size == 0:
            raise ValueError("Covariance must be calculated on an array of row vectors")
        # Make sure everything is a float
        try:
            return xmat.astype(float)
        except ValueError:
            raise ValueError("Could not cast input data to floats")

    def __call__(self, X1, X2, noise_term=False):
        return self.get_cov_matrix(X1, X2, noise_term)


class SEKernel(_Kernel):
    """Callable class for computing square-exponential kernels. Implements
    the _Kernel abstract class.
    """

    name = "SEKernel"

    def __init__(self, sigma_n, sigma_f, l):
        """Build a SEKernel object. 

            sigma_n : float
                Value >=0 representing the amount of noise in our
                measurements. sigma_n of zero means no noise was
                present in the measurements.

            sigma_f : float
                Value >=0 representing the signal variance

            l : 1d array of floats A list of characteristic distances.
                If set, there must be exactly one characteristic
                distance for each input data dimension. Each entry
                must be greater than zero.

        Passes up any exceptions.

        """
        # Set and check hyperparameters
        self.set_hyperparams(sigma_n, sigma_f, l)

    def cov(self, x1, x2):
        """Calculate the square-exponential covariance between two vectors
        x1, x2, for hyperparameters sigma_f, l. Assumes sigma_f, l
        have been checked as much as possible by set_hyperparams.
        
            x1, x2 : 1-by-n float array

        Returns the square-exponential covariance between x1, x2.
        Assumes everything has been checked by previous functions!
        """
        exp_arg = -0.5 * np.dot(np.dot((x2 - x1).T, self.l), x2 - x1)
        return self.sigma_f * np.exp(exp_arg)

    def set_hyperparams(self, sigma_n=None, sigma_f=None, l=None):
        """
        Store and check the hyperparameter values.

            sigma_n : float
                Value >=0 representing the amount of noise in our
                measurements. sigma_n of zero means no noise was
                present in the measurements.

            sigma_f : float
                Value >=0 representing the signal variance. Default 1.

            l : 1d array of floats
                A list of characteristic distances. If set, there must
                be exactly one characteristic distance for each input
                data dimension. If unset, distances default to 1. Each
                entry must be greater than zero.

        Hyperparameters that are passed as None are left unchanged.

        Raises the following:
        
            ValueError : if sigma_n is negative

            ValueError : if l is not a 1d array

            ValueError : if sigma_f is not greater than zero

            ValueError : if characteristic lengths are not all greater
                         than zero

            TypeError : if sigma_n is not castable to a float

            TypeError : if entries of l cannot be cast to a float

            TypeError : if sigma_f is not a scalar

            TypeError : if sigma_f cannot be cast to a float
        """
        # Check sigma_n if set
        if sigma_n is not None:
            try:
                self.sigma_n = float(sigma_n)
            except TypeError:
                raise TypeError("sigma_n must be castable to a float")
            if not self.sigma_n >= 0:
                raise ValueError("sigma_n must be non-negative")
            self.sigma_f = sigma_f
        # Check l if set
        if l is not None:
            self.l = np.array([l]) if np.isscalar(l) else np.array(l)
            # Check l is 1d
            if not len(self.l.shape) == 1:
                raise ValueError("Characteristic lengths l must be a 1d array")
            # Check l is floaty
            try:
                self.l = self.l.astype(float)
            except TypeError:
                raise TypeError("Entries of l must all be castable to type float")
            # Make sure l is all non-negative
            if np.any(self.l <= 0):
                raise ValueError("Characteristic lengths l must all be greater than 0")
        # Check sigma_f if set
        if sigma_f is not None:
            # Check sigma_f is a scalar
            if not np.isscalar(self.sigma_f):
                raise TypeError("sigma_f must be a scalar")
            # Check sigma_f is floaty
            try:
                self.sigma_f = float(sigma_f)
            except TypeError:
                raise TypeError("sigma_f must be castable to type float")
            # Make sure sigma_f is non-negative
            if self.sigma_f <= 0:
                raise ValueError("sigma_f must be greater than 0")


class _AbstractPeriodicKernel(SEKernel, ABC):
    """To make the Gaussian process periodic, we simply ensure the
    covariance is periodic. This abstract class subclasses SEKernel
    and ABC, to get an abstract implementation of an RBF kernel.
    """

    def __init__(self, period, sigma_n=0, sigma_f=1, l=None):
        super().__init__(sigma_n, sigma_f, l)
        # Check period is float > 0
        if not isinstance(period, (float, int)):
            raise TypeError("period must be float or int")
        if not period > 0:
            raise ValueError("period must be greater than zero")
        self._period = period

    def set_hyperparams(self, sigma_n=None, sigma_f=None, l=None, period=None):
        super().set_hyperparams(sigma_n, sigma_f, l)
        if period is not None:
            if not isinstance(period, (float, int)):
                raise TypeError("period must be float or int")
            if not period > 0:
                raise ValueError("period must be greater than zero")
            self._period = period


class PeriodicSEKernel(_AbstractPeriodicKernel):
    """Here we model periodic covariance by letting the correlation be
    given by sum[ m = -infty to infty] of cov(x1, x2+mT), for period
    T. The code here uses a modulo operator to approximate this.
    """
    name = "PeriodicSEKernel"

    def cov(self, x1, x2):
        # TODO Check x1, x2 are scalar
        p1 = np.mod(x2 - x1, self._period)
        p2 = np.mod(x1 - x2, self._period)
        phase = min(p1, p2)
        return super().cov(x1, x1 + phase)


class PeriodicKernel(_AbstractPeriodicKernel):
    """Here we model periodic covariance by letting the correlation be the
    SEKernel acting on the sine of the distance between two inputs.
    """
    name = "PeriodicKernel"

    def cov(self, x1, x2):
        dist = np.linalg.norm(x2-x1)
        exp_arg = np.sin(np.pi*dist/(self._period))**2 / self.l
        return self.sigma_f * np.exp(-exp_arg)


KERNEL_NAMES = {"SEKernel":SEKernel, "PeriodicKernel":PeriodicKernel, "PeriodicSEKernel":PeriodicSEKernel}
