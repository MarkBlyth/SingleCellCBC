#!/usr/bin/env python3

import gpr
import numpy as np
import matplotlib.pyplot as plt

# Funcs for generating random functions
def random_function(xs, l, sigma_f):
    """
    Evaluate a random C-infinity smooth function at points specified
    by xs. Hyperparameter l specifies how quickly nearby points
    decorrelate, and sigma_f specifies the variance of the function.
    """
    out_rows = xs.shape[0]
    cov_matrix = np.zeros((out_rows, out_rows))
    for i, x1 in enumerate(xs):
        for j, x2 in enumerate(xs):
            lengths = np.diag(1 / l)
            exp_arg = (-1 / 2) * (x2 - x1) * lengths * (x2 - x1)
            cov = sigma_f * np.exp(exp_arg)
            cov_matrix[i][j] = cov
    mean = np.zeros(xs.shape)
    ys = np.random.multivariate_normal(mean, cov_matrix)
    return ys


# Generate some samples from the prior
#   This is treated as the underlying,
#   'true' process, that we wish to model
l = np.array([1])
sigma_f = 1
xs = np.linspace(0, 10)
ys = random_function(xs, l, sigma_f)

# Take a subset of points as training data
#   This is designed to replicate us only
#   having partial knowledege (few observations)
#   of the underlying process
xs = xs.reshape((-1, 1))
training_X = xs[:-1:5]
training_y = ys[:-1:5]

# Build a GPR model on the data
# (This is where the GPR happens!)
kernel = gpr.SEKernel(0, sigma_f, l)
regressor = gpr.GPR(training_X, training_y, kernel)
predictions = regressor(xs)

# Plot!
fig, ax = plt.subplots()
ax.plot(xs, ys, "--", label="Original process")
ax.scatter(training_X, training_y, label="Sampled training data")
ax.plot(xs, predictions, label="GPR estimate")
ax.legend()
plt.show()
