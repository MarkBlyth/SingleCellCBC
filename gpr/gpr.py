import numpy as np
import scipy.linalg
import warnings
from . import kernels

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
        if not isinstance(cov, kernels._Kernel):
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
        if not istype(cov, kernels._Kernel):
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
        except scipy.linalg.LinAlgError as e:
            warnings.warn(
                "Cholesky decomposition failed with warning {0}. Reverting to non-Cholesky solution".format(
                    e
                )
            )
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
        X = np.array([[X]]) if np.isscalar(X) else np.array(X)
        # If X is 3d, 4d, ...
        if len(X.shape)>2:
            raise ValueError("Cannot handle {0} dimensional X".format(len(X.shape)))
        # If X is 1d
        if len(X.shape)==1:
            if len(self._X[0])==1:
                # List of scalars
                X = X.reshape((-1,1))
            else:
                # Single row vector
                X = X.reshape((1,-1))
        if not X[0].shape == self._X[0].shape:
            raise ValueError("Vectors in X must match the shape of vectors in the training data")
        # Data checks done
        if not "_L" in self.__dict__:
            # Variance only implemented for Cholesky solution
            raise NotImplementedError(
                "Variance has not been implemented for non-Cholesky solutions"
            )
        k_xstar_xstar = self.cov(X,X, True)
        kstar = self.cov(X, self._X)
        v = scipy.linalg.cho_solve(self._L, kstar)
        return k_xstar_xstar - np.dot(v.T, v)


    def add_data(self, data_X, data_y=None):
        raise NotImplementedError

    def remove_data(self, data_to_remove):
        raise NotImplementedError
