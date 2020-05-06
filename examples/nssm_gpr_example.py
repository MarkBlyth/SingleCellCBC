#!/usr/bin/env python2

"""
Code based on that provided at https://github.com/sremes/nssm-gp. 
Implements a neural generalised non-stationary spectral kernel.


MIT License

Copyright (c) 2018 Sami Remes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import gc
import tempfile

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model_class import Model

import gpflow
import gpflow.utilities
from gpflow import Parameter
from gpflow import Parameter
from gpflow.models import VGP
from gpflow.conditionals import conditional
from gpflow.kernels import (
    Kernel,
    IsotropicStationary,
    Sum,
    Product,
)  # used to derive new kernels
from tensorflow_probability import bijectors

T_MAX = 200
N_COMPONENTS = 10
TARGET_ELBO = -180

float_type = gpflow.default_float()

# ----------------------------------------------------------------------------------------------------
# Neural spectral kernel definintion
# ----------------------------------------------------------------------------------------------------


class NeuralSpectralKernel(gpflow.kernels.Kernel):
    def __init__(self, input_dim, active_dims=None, Q=1, hidden_sizes=None):
        super().__init__(active_dims=active_dims)
        self.input_dim = input_dim
        self.Q = Q
        if hidden_sizes is None:
            hidden_sizes = (32, 32)
        self.num_hidden = len(hidden_sizes)
        for v, final_size in zip(["freq", "len", "var"], [input_dim, input_dim, 1]):
            self._create_nn_params(v, hidden_sizes, final_size)

    def _create_nn_params(self, prefix, hidden_sizes, final_size):
        for q in range(self.Q):
            input_dim = self.input_dim
            for level, hidden_size in enumerate(hidden_sizes):
                """name_W = '{prefix}_{q}_W_{level}'.format(prefix=prefix, q=q, level=level)
                name_b = '{prefix}_{q}_b_{level}'.format(prefix=prefix, q=q, level=level)
                params = _create_params(input_dim, hidden_size)
                setattr(self, name_W, params[0])
                setattr(self, name_b, params[1])"""
                name_W = "{prefix}_W_{level}".format(prefix=prefix, level=level)
                name_b = "{prefix}_b_{level}".format(prefix=prefix, level=level)
                if not hasattr(self, name_W):
                    params = _create_params(input_dim, hidden_size)
                    setattr(self, name_W, params[0])
                    setattr(self, name_b, params[1])
                # input dim for next layer
                input_dim = hidden_size
            params = _create_params(input_dim, final_size)
            setattr(self, "{prefix}_{q}_W_final".format(prefix=prefix, q=q), params[0])
            setattr(self, "{prefix}_{q}_b_final".format(prefix=prefix, q=q), params[1])

    def _nn_function(self, x, prefix, q, dropout=0.8, final_activation=tf.nn.softplus):
        for level in range(self.num_hidden):
            """W = getattr(self, '{prefix}_{q}_W_{level}'.format(prefix=prefix, q=q, level=level))
            b = getattr(self, '{prefix}_{q}_b_{level}'.format(prefix=prefix, q=q, level=level))"""
            W = getattr(self, "{prefix}_W_{level}".format(prefix=prefix, level=level))
            b = getattr(self, "{prefix}_b_{level}".format(prefix=prefix, level=level))
            x = tf.nn.selu(
                tf.compat.v1.nn.xw_plus_b(x, W, b)
            )  # self-normalizing neural network
            # if dropout < 1.0:
            #     x = tf.contrib.nn.alpha_dropout(x, keep_prob=dropout)
        W = getattr(self, "{prefix}_{q}_W_final".format(prefix=prefix, q=q))
        b = getattr(self, "{prefix}_{q}_b_final".format(prefix=prefix, q=q))
        return final_activation(tf.compat.v1.nn.xw_plus_b(x, W, b))

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        kern = 0.0
        for q in range(self.Q):
            # compute latent function values by the neural network
            freq, freq2 = (
                self._nn_function(X, "freq", q),
                self._nn_function(X2, "freq", q),
            )
            lens, lens2 = (
                self._nn_function(X, "len", q),
                self._nn_function(X2, "len", q),
            )
            var, var2 = self._nn_function(X, "var", q), self._nn_function(X2, "var", q)

            # compute length-scale term
            Xr = tf.expand_dims(X, 1)  # N1 1 D
            X2r = tf.expand_dims(X2, 0)  # 1 N2 D
            l1 = tf.expand_dims(lens, 1)  # N1 1 D
            l2 = tf.expand_dims(lens2, 0)  # 1 N2 D
            L = tf.square(l1) + tf.square(l2)  # N1 N2 D
            # D = tf.square((Xr - X2r) / L)  # N1 N2 D
            D = tf.square(Xr - X2r) / L  # N1 N2 D
            D = tf.reduce_sum(input_tensor=D, axis=2)  # N1 N2
            det = tf.sqrt(2 * l1 * l2 / L)  # N1 N2 D
            det = tf.reduce_prod(input_tensor=det, axis=2)  # N1 N2
            E = det * tf.exp(-D)  # N1 N2

            # compute cosine term
            muX = tf.reduce_sum(
                input_tensor=freq * X, axis=1, keepdims=True
            ) - tf.transpose(
                a=tf.reduce_sum(input_tensor=freq2 * X2, axis=1, keepdims=True)
            )
            COS = tf.cos(2 * np.pi * muX)

            # compute kernel variance term
            WW = tf.matmul(var, var2, transpose_b=True)  # w*w'^T

            # compute the q'th kernel component
            kern += WW * E * COS
        if np.all((X - X2) < 1e-6):
            return robust_kernel(kern, tf.shape(input=X)[0])
        else:
            return kern

    def K_diag(self, X):
        kd = gpflow.default_jitter()
        for q in range(self.Q):
            kd += tf.square(self._nn_function(X, "var", q))
        return tf.squeeze(kd)


def robust_kernel(kern, shape_X):
    eigvals = tf.linalg.eigvalsh(kern)
    min_eig = tf.reduce_min(input_tensor=eigvals)
    jitter = gpflow.default_jitter()

    def abs_min_eig():
        return tf.compat.v1.Print(
            tf.abs(min_eig), [min_eig], "kernel had negative eigenvalue"
        )

    def zero():
        return tf.constant(0, dtype=float_type)

    jitter += tf.cond(pred=tf.less(min_eig, 0.0), true_fn=abs_min_eig, false_fn=zero)
    return kern + jitter * tf.eye(shape_X, dtype=gpflow.default_float())


def _create_params(input_dim, output_dim):
    def initializer():
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        return np.random.uniform(-limit, +limit, (input_dim, output_dim))

    return (
        Parameter(initializer(), dtype=float_type),
        Parameter(np.zeros(output_dim), dtype=float_type),
    )


# ----------
# Initializers for Neural spectral kernels.
# ----------


def random_Z(x, N, M):
    inducing_idx = np.random.randint(N, size=M)
    Z = x[inducing_idx, :].copy()
    Z += 1e-3 * np.random.randn(*Z.shape)
    Z = np.sort(Z, 0)
    return Z


def init_neural(
    data_x,
    data_y,
    n_components,
    n_inits=1,
    noise_var=0.1,
    likelihood=None,
    hidden_sizes=None,
):
    print("Initializing neural spectral kernel...")
    best_loglik = -np.inf
    best_m = None
    N, input_dim = data_x.shape
    for k in range(n_inits):
        try:
            Z = random_Z(data_x, N, N)
            k = NeuralSpectralKernel(
                input_dim, Q=n_components, hidden_sizes=hidden_sizes
            )
            if likelihood is not None:
                likhood = likelihood
            else:
                likhood = gpflow.likelihoods.Gaussian(noise_var)
            model = VGP(data=(data_x, data_y), kernel=k, likelihood=likhood)
            loglik = model.elbo()
            if loglik > best_loglik:
                best_loglik = loglik
                best_m = model
            del model
            gc.collect()
        except tf.errors.InvalidArgumentError:  # cholesky fails sometimes (with really bad init?)
            pass
    print("Best init: %f" % best_loglik)
    print(best_m)
    # print(best_m)
    return best_m


def simple_training_loop(model):
    optimizer = tf.optimizers.Adam()
    epoch_id =0
    loss = -np.inf
    while loss < TARGET_ELBO:
        optimizer.minimize(model.training_loss, model.trainable_variables)
        loss = model.elbo()
        epoch_id += 1
        tf.print(f"Epoch {epoch_id}: ELBO (train) {loss}")


def fitzhugh_nagumo_neuron(x, t, pars):
    """Defines the RHS of the FH model"""
    v, w = x  # Unpack state
    v_dot = v - (v ** 3) / 3 - w + pars["I"]
    w_dot = 0.08 * (v + 0.7 - 0.8 * w)
    return [v_dot, w_dot]


def main():

    print("Simulating neuron...")
    FHNeuron = Model()
    FHNeuron["model"] = fitzhugh_nagumo_neuron
    FHNeuron["parvec"] = ["I"]
    FHNeuron["I"] = 1
    solution = FHNeuron.run_model([0, T_MAX], [-1, -1])  # t_span, initial_contition
    data_X = solution.t.reshape((-1, 1))
    data_y = solution.y[0].reshape((-1, 1))

    model = init_neural(data_X, data_y, n_components=N_COMPONENTS, n_inits=50)
    print("Optimizing model")
    simple_training_loop(model)

    # Evaluate model
    print("Started evaluation!")
    test_x = np.linspace(0, T_MAX, len(data_X)).reshape((-1,1))
    test_y, _ = model.predict_f(test_x)
    print("Done evaluating")
    fig, ax = plt.subplots()
    ax.plot(data_X, data_y, label="Training")
    ax.scatter(data_X, data_y, label="Training")
    ax.plot(test_x, test_y, label="Test")
    ax.scatter(test_x, test_y, label="Test")
    #ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
