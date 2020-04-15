#!/usr/bin/env python3

import gpr
import numpy as np
import matplotlib.pyplot as plt

# To make the Gaussian process periodic, 
#   we simply ensure the covariance is
#   periodic. Here, we let the correlation
#   be given by sum[ m = -infty to infty]
#   of cov(x1, x2+mT), for period T:
class PeriodicSE(gpr.SEKernel):
    def __init__(self, period, sigma_n=0, sigma_f=1, l=None):
        super().__init__(sigma_n, sigma_f, l)
        # Check period is float > 0
        self._period = period

    def cov(self, x1, x2):
        # Check x1, x2 are scalar
        p1 = np.mod(x2-x1, self._period)
        p2 = np.mod(x1-x2, self._period)
        phase = min(p1,p2)
        return super().cov(x1, x1+phase)

# Build a periodic SEKernel
period = 0.5
l = (period/(2*np.pi))**2 # From Adler 1981 theorem 4.1.1
sigma_f = 1 # Guess
kernel = PeriodicSE(period,0, sigma_f, l)

# Generate a periodic signal to work from
ts = np.linspace(0, 100, 100)
xs = np.sin(2*np.pi*ts / period)

# Take a subset of points as training data
#   This is designed to replicate us only
#   having partial knowledege (few observations)
#   of the underlying process
ts = ts.reshape((-1, 1))
training_X = ts[:-1:5]
training_y = xs[:-1:5]

# Build a GPR model on the data
# (This is where the GPR happens!)
regressor = gpr.GPR(training_X, training_y, kernel)
predictions = regressor(ts)

# Plot!
fig, ax = plt.subplots()
ax.plot(ts, xs, "--", label="Original process")
ax.scatter(training_X, training_y, label="Sampled training data")
ax.plot(ts, predictions, label="GPR estimate")
ax.legend()
plt.show()
