#!/usr/bin/env python3

import gpr
import numpy as np
import matplotlib.pyplot as plt
import  cProfile
import timeit

N_SAMPLES = 500

# Funcs for generating random functions
def random_function(xs, l, sigma_f):
    """
    Evaluate a random C-infinity smooth function at points specified
    by xs. Hyperparameter l specifies how quickly nearby points
    decorrelate, and sigma_f specifies the variance of the function.
    """
    out_rows = xs.shape[0]
    cov_matrix = np.zeros((out_rows, out_rows))
    lengths = np.diag(1 / l)
    def cov(x1,x2):
        exp_arg = (-1 / 2) * (x2 - x1) * lengths * (x2 - x1)
        return sigma_f * np.exp(exp_arg)
    n_rows = len(xs)
    cov_matrix = np.array([[cov(xs[j], xs[i]) for i in range(n_rows)] for j in range(n_rows)]).squeeze()
    mean = np.zeros(xs.shape)
    ys = np.random.multivariate_normal(mean, cov_matrix)
    return ys


# Generate some samples from the prior
#   This is treated as the underlying,
#   'true' process, that we wish to model
l = np.array([1])
sigma_f = 1
xs = np.linspace(0, 10, N_SAMPLES)
ys = random_function(xs, l, sigma_f)

kernel = gpr.kernels.SEKernel(0, sigma_f, l)

print("Fitting")
print(timeit.timeit(lambda: gpr.gpr.GPR(xs,ys, kernel), number=10)/10," per run")
cProfile.run("gpr.gpr.GPR(xs,ys, kernel)")

#regressor = gpr.gpr.GPR(xs,ys, kernel)

# print("Testing")
# predictions = regressor(xs)
