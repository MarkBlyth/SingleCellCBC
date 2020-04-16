import numpy as np
from abc import ABC, abstractmethod
import scipy.linalg

"""
Code interface:
SISO:
  - Trainer:
    * accept 2d training x iff it has the same rows as training y has cols, reject otherwise
    * accept 1d training x iff it's the same shape as training y, reject otherwise; reshape to row vecs anyway
  - Predictor:
    * accept scalar inputs, reshape to 2d array of row vecs
    * accept 1d arrays and reshape to 2d array of row vecs
    * accept 2d array of 1-element row vecs
    * reject anything else

MISO:
  - Trainer:
    * accept 2d training x if it has the same rows as training y has cols
    * reject anything else
  - Predictor:
    * accept 1d array iff it is the same shape as the training input vecs
    * accept 2d array of row vecs iff each row vec is the same shape as input vecs
    * reject anything else
"""


class GPR:
    """
    Class implementing a Gaussian process regression. Finds a
    non-parametric model of the data, for a given kernel, training
    inputs (x values), and training outputs (y values). x values must
    be row vectors or scalars. y values must be scalars. Code
    internals treat X as a 2d array of row vectors, for both scalar
    and vector X. Prediction points can be scalar, 1d array, or 2d
    array of single-entry row vectors, for scalar training data; 1d
    array (single vector), or 2d array of row vectors, for vector
    training data.
    """

    def __init__(self, X_data, y_data, cov):
        """
        Constructor for GPR class. Vector X_data must be a 2d array,
        where each row is a row vector of training x points. Scalar
        X_data can either be a 1d array of x datapoints, or considered
        as a set of one-dimensional row vectors and passed as a 2d row
        vector array. y_data must be scalar, and passed as a 1d array
        containing one output for each X_data input.

            X_data : float array
                Array of row vectors at which the training data were
                sampled. Scalar inputs can be either a 1d array, or 2d
                array of single-entry row vectors. Vector inputs must
                be a 2d array of row vectors.

            y_data : 1d float array
                Array of measured outputs. Outputs must be scalars.
                y_data must be a 1d array even if we have only a
                single datapoint. There must be the same number of
                outputs y as inputs x.

            cov : _Kernel object
                Kernel to use.

        Raises the following:

            ValueError : if y_data is not a 1d array

            ValueError : if the number of scalars or vectors in x does
                         not match the number of scalars in y

            ValueError : if X_data is neither a 1d array of scalars,
                         nor a 2d array of row vectors

            TypeError : if X or y cannot be cast to float arrays

            TypeError : if cov is not a _Kernel object
        """
        X = np.array(X_data)
        y = np.array(y_data)
        # Check data type
        try:
            X = X.astype(float)
            y = y.astype(float)
        except TypeError:
            raise TypeError("X, y must be castable to floats")
        # Check y shape
        if not len(y.shape) == 1:
            raise ValueError("Training data y must be a 1d array of scalars")
        # If we have a 1d X array...
        if len(X.shape) == 1:
            # ...ensure it's the same shape as training y
            if not X.shape == y.shape:
                raise ValueError(
                    "A 1d array of scalar training X must contain the same number of entries as training y"
                )
            X = X.reshape((-1, 1))  # Reshape to row vecs
        # If we have a 2d X array...
        elif len(X.shape) == 2:
            # make sure there's the same number of row vecs as scalar outputs
            if not X.shape[0] == y.shape[0]:
                raise ValueError(
                    "Number of rows (input vectors) in X must equal number of elements (output scalars) in y"
                )
        # Reject anything else, eg scalar X, or 3d+ X
        else:
            raise ValueError(
                "X must be a 1d array of scalars or 2d array of row vectors"
            )
        # Check cov type
        if not isinstance(cov, _Kernel):
            raise TypeError("Covariance function must be a _Kernel object")

        # All checks passed!
        self.cov = cov
        self._X = X
        self._y = y
        self._fit()

    def predict(self, X):
        """
        Find the GP mean (predicted system output) at positions X. For
        scalar training data, X can be scalar, or a 1d array of
        scalars. For vector training data, X can be a single 1d array
        row vector, or a 2d array of several row vectors.

            X : float array
                2d array of row vectors at which we wish to evaluate
                the Gaussian process.

        Raises the following:

            ValueError : if X is not a scalar, 1d, or 2d array

            ValueError : if the vectors in X do not contain the same
                         number of elements as the training vectors

        Passes up any other exceptions.

        Returns a 1d array. Element i of the returned array gives the
        mean of the Gaussian process, evaluated at the i'th vector of
        X.
        """
        # Turn scalar inputs into 2d array
        if np.isscalar(X):
            X = np.array([[X]])
        X = np.array(X)
        # Turn 1d arrays into appropriate 2d arrays
        if len(X.shape) == 1:
            if self._X.shape[1] == 1:
                # Reshape list of scalars to 2d array of single-entry row vecs
                X = X.reshape((-1, 1))
            else:
                # Reshape a single row vector to a 2d array containing one row vec
                X = X.reshape((1, -1))
        # Reject anything that's not scalar, 1d, or 2d
        if len(X.shape) != 2:
            raise ValueError("X must be scalar, 1d array, or 2d array of row vecs")
        # Make sure test vec dimensions are the same as training vec dimensions
        if not X[0].shape == self._X[0].shape:
            raise ValueError(
                "Test points must be of the same dimension as training points"
            )
        Kss = self.cov(self._X, X)
        return np.dot(Kss, self._alpha)

    def set_covariance_func_and_compute(self, cov):
        """
        Sets the covariance function (kernel) used for computing the
        posterior Gaussian process. As using a different kernel will
        yield different predictions, the model is also recomputed
        automatically.
        
            cov : _Kernel obj
                New _Kernel object to use
        
        Raises the following:
        
            TypeError : if cov is not a _Kernel object
        """
        if not istype(cov, _Kernel):
            raise TypeError("Covariance function must be a _Kernel object")
        self.cov = cov
        self._fit()

    def _fit(self):
        """
        Fit a GPR model to the training data. Assumes the training
        data and kernel have already been set and checked by either
        __init__, add_data, or set_covariance_func_and_compute.
        """
        self.Kxx = self.cov(self._X, self._X, noise_term=True)
        try:
            self._L = scipy.linalg.cho_factor(self.Kxx)
            self._alpha = scipy.linalg.cho_solve(self._L, self._y)
            self.log_likelihood = (
                -0.5 * np.dot(self._y, self._alpha)
                - 0.5 * np.sum(np.log(np.diag(self._L[0])))
                - 0.5 * self._X.shape[0] * np.log(2 * np.pi)
            )
        except scipy.linalg.LinAlgError:
            # If the matrix is ill-conditioned for a Cholesky solution
            self._alpha = scipy.linalg.solve(self.Kxx, self._y)
            self.log_likelihood = (
                -0.5 * np.dot(self._y, self._alpha)
                - 0.5 * np.sum(np.log(np.diag(self.Kxx)))
                - 0.5 * self._X.shape[0] * np.log(2 * np.pi)
            )

    def __call__(self, X):
        return self.predict(X)

    def get_variance(self, X):
        raise NotImplementedError

    def add_data(self, data_X, data_y=None):
        raise NotImplementedError

    def remove_data(self, data_to_remove):
        raise NotImplementedError


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

    def __init__(self, sigma_n):
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

    def __init__(self, sigma_n=0, sigma_f=1, l=None):
        """
        Build a SEKernel object. 

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

        Passes up any exceptions.
        """
        # Set noise level
        super().__init__(sigma_n)
        # Set and check other hyperparameters
        self.set_hyperparams(sigma_f, l)

    def cov(self, x1, x2):
        return self._SE_covariance(x1, x2, self.sigma_f, self.l)

    def set_hyperparams(self, sigma_f=1, l=None):
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

        Raises the following:
        
            ValueError : if l is not a 1d array

            ValueError : if sigma_f is not greater than zero

            ValueError : if characteristic lengths are not all greater
                         than zero

            TypeError : if entries of l cannot be cast to a float

            TypeError : if sigma_f is not a scalar

            TypeError : if sigma_f cannot be cast to a float
        """
        self.sigma_f = sigma_f
        self.l = np.array([l]) if np.isscalar(l) else np.array(l)
        # Check l if set
        if l is not None:
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

    def _SE_covariance(self, x1, x2, sigma_f=1, l=None):
        """Calculate the square-exponential covariance between two vectors
        x1, x2, for hyperparameters sigma_f, l. Assumes sigma_f, l
        have been checked as much as possible by set_hyperparams.
        
            x1, x2 : 1-by-n float array

            sigma_f : float
                Value >=0 representing the signal variance. Default 1.

            l : 1d array of floats
                A list of characteristic distances. If set, there must
                be exactly one characteristic distance for each input
                data dimension. If unset, distances default to 1. Each
                entry must be greater than zero.

        Raises the following:
        
            TypeError : if x1 or x2 cannot be cast to a float

            ValueError : if x1 and x2 are not 1d arrays 

            ValueError : if x1 and x2 do not contain the same number
                         of entries

            ValueError : if l and x1, x2 do not contain the same
                         number of entries

        Returns the square-exponential covariance between x1, x2.
        """
        x1squeeze, x2squeeze = np.array(x1).squeeze(), np.array(x2).squeeze()
        try:
            x1squeeze = x1squeeze.astype(float)
            x2squeeze = x2squeeze.astype(float)
        except TypeError:
            raise TypeError("x1 and x2 must be castable to float arrays")
        try:
            # Make sure x1, x2 are vectors, not matrices
            if len(x1squeeze) != 1:
                raise ValueError("Expected vectors for x1, x2, but recieved matrices")
        except TypeError:
            # If xi are scalar, len will throw a type error
            # We needn't worry about it, since scalars are necessarily the same dimension
            x1squeeze, x2squeeze = np.array([x1squeeze]), np.array([x2squeeze])
        # Make sure x1, x2 are of the same dimensions
        if x1.shape != x2.shape:
            raise ValueError("Vectors x1, x2 must be the same dimension")
        # Set l to identity if not defined
        if l is None:
            l = np.eye(x1squeeze.shape[0])
        else:
            # Check l is the same dimension as our vectors
            # No need for other checks since l's type, sigma_f were checked by set_params
            if not l.shape == x1squeeze.shape:
                raise ValueError("l must be the same dimension as input vectors")
            # Turn l into a diagonal matrix
            l = np.diag(1 / l)
        # Calculate and return!
        exp_arg = -0.5 * np.dot(np.dot((x2 - x1).T, l), x2 - x1)
        return sigma_f * np.exp(exp_arg)
